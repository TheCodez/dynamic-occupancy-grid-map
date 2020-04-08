/*
MIT License

Copyright (c) 2019 Michael Kösel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "dogm/dogm.h"
#include "dogm/dogm_types.h"

#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

#define PI 3.14159265358979323846f

struct Vehicle
{
    Vehicle(const int width, const glm::vec2& pos, const glm::vec2& vel) : width(width), pos(pos), vel(vel) {}

    void move(float dt) { pos += vel * dt; }

    int width;
    glm::vec2 pos;
    glm::vec2 vel;
};

struct Simulator
{
	explicit Simulator(int num_measurements) : num_measurements(num_measurements) {}

    void addVehicle(const Vehicle& vehicle) { vehicles.push_back(vehicle); }

    std::vector<std::vector<float>> update(int steps, float dt)
    {
        std::vector<std::vector<float>> measurements;

        for (int i = 0; i < steps; i++)
        {
            std::vector<float> measurement(num_measurements, INFINITY);

            for (auto& vehicle : vehicles)
            {
                vehicle.move(dt);

				for (int j = 0; j < vehicle.width; ++j)
				{
					int index = static_cast<int>(vehicle.pos.x) + j;
					measurement[index] = vehicle.pos.y;
				}
			}

            measurements.push_back(measurement);
        }

        return measurements;
    }

    int num_measurements;
    std::vector<Vehicle> vehicles;
};

float pignistic_transformation(float free_mass, float occ_mass)
{
    return occ_mass + 0.5f * (1.0f - occ_mass - free_mass);
}

cv::Mat compute_measurement_grid_image(const dogm::DOGM& grid_map)
{
    cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        for (int x = 0; x < grid_map.getGridSize(); x++)
        {
            int index = y * grid_map.getGridSize() + x;

            const dogm::MeasurementCell& cell = grid_map.meas_cell_array[index];
            float occ = pignistic_transformation(cell.free_mass, cell.occ_mass);
            uchar temp = static_cast<uchar>(floor(occ * 255));
            grid_img.at<cv::Vec3b>(y, x) = cv::Vec3b(255 - temp, 255 - temp, 255 - temp);
        }
    }

    return grid_img;
}

cv::Mat compute_raw_measurement_grid_image(const dogm::DOGM& grid_map)
{
    cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        for (int x = 0; x < grid_map.getGridSize(); x++)
        {
            int index = y * grid_map.getGridSize() + x;
            const dogm::MeasurementCell& cell = grid_map.meas_cell_array[index];
            int red = static_cast<int>(cell.occ_mass * 255.0f);
            int green = static_cast<int>(cell.free_mass * 255.0f);
            int blue = 255 - red - green;

            grid_img.at<cv::Vec3b>(y, x) = cv::Vec3b(blue, green, red);
        }
    }

    return grid_img;
}

cv::Mat compute_raw_polar_measurement_grid_image(const dogm::DOGM& grid_map)
{
    cv::Mat grid_img(grid_map.getGridSize(), 100, CV_8UC3);
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        for (int x = 0; x < 100; x++)
        {
            int index = y * 100 + x;
            const dogm::MeasurementCell& cell = grid_map.polar_meas_cell_array[index];
            int red = static_cast<int>(cell.occ_mass * 255.0f);
            int green = static_cast<int>(cell.free_mass * 255.0f);
            int blue = 255 - red - green;

            grid_img.at<cv::Vec3b>(y, x) = cv::Vec3b(blue, green, red);
        }
    }

    return grid_img;
}

cv::Mat createCircularColorGradient(const int hue_offset)
{
    // Set linear gradient (180 levels). Needed because the OpenCV hue range is [0, 179], see
    // https://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html
    const int hue_steps = 180;
    cv::Mat hue_image{hue_steps, hue_steps, CV_8UC3, cv::Scalar(0)};
    const cv::Point image_center{hue_image.cols / 2, hue_image.rows / 2};
    cv::Mat hsv{1, 1, CV_8UC3, cv::Scalar(0, 255, 255)};
    cv::Mat rgb{1, 1, CV_8UC3};
    for (int hue = 0; hue < hue_image.rows; ++hue)
    {
        hsv.at<uchar>(0, 0) = (hue + hue_offset) % 180;
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
        hue_image.row(hue).setTo(rgb.at<cv::Vec3b>(0, 0));
    }
    cv::linearPolar(hue_image, hue_image, image_center, 255,
                    cv::INTER_CUBIC | cv::WARP_FILL_OUTLIERS | cv::WARP_INVERSE_MAP);
    return hue_image;
}

cv::Mat createCircleMask(const cv::Size2d& size)
{
    cv::Mat mask{size, CV_8UC3, cv::Scalar(0, 0, 0)};
    const cv::Point image_center{mask.cols / 2, mask.rows / 2};
    const float outer_radius = mask.rows / 2;
    const float inner_radius = mask.rows / 4;
    cv::circle(mask, image_center, outer_radius, cv::Scalar(1, 1, 1), -1);
    cv::circle(mask, image_center, inner_radius, cv::Scalar(0, 0, 0), -1);
    return mask;
}

cv::Mat createColorWheel(const int hue_offset = 0)
{
    cv::Mat circular_color_gradient = createCircularColorGradient(hue_offset);
    cv::Mat circle_mask = createCircleMask(circular_color_gradient.size());

    cv::Mat color_wheel{};
    circular_color_gradient.copyTo(color_wheel, circle_mask);
    return color_wheel;
}

void resizeRelativeToShorterEdge(const cv::Mat& original_image, const float relative_size, cv::Mat& destination_image)
{
    const float target_edge_length = std::min(original_image.cols, original_image.rows) * relative_size;
    cv::resize(destination_image, destination_image, cv::Size(target_edge_length, target_edge_length));
}

cv::Mat createFullSaturationFullValueAlphaMaskOf(const cv::Mat& img)
{
    cv::Mat mask{};
    cv::Mat img_hsv{};
    cv::cvtColor(img, img_hsv, cv::COLOR_RGB2HSV);
    cv::inRange(img_hsv, cv::Scalar(0, 254, 254), cv::Scalar(180, 255, 255), mask);
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2RGB);
    return mask;
}

void blendIntoBottomRightCorner(const cv::Mat& img_to_blend_in, cv::Mat& img)
{
    const cv::Rect roi{img.cols - img_to_blend_in.cols, img.rows - img_to_blend_in.rows, img_to_blend_in.cols,
                       img_to_blend_in.rows};
    cv::Mat img_roi = img(roi);
    cv::Mat mask = createFullSaturationFullValueAlphaMaskOf(img_to_blend_in);
    cv::multiply(cv::Scalar::all(1.0) - mask, img_roi, img_roi);
    cv::addWeighted(img_roi, 1.0F, img_to_blend_in, 1.0F, 0.0, img_roi);
}

void addColorWheelToBottomRightCorner(cv::Mat& img, const float relative_size = 0.2, const int hue_offset = 0)
{
    cv::Mat color_wheel = createColorWheel(hue_offset);
    resizeRelativeToShorterEdge(img, relative_size, color_wheel);
    blendIntoBottomRightCorner(color_wheel, img);
}

cv::Mat compute_dogm_image(const dogm::DOGM& grid_map, float occ_tresh = 0.7f, float m_tresh = 4.0f)
{
    cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        for (int x = 0; x < grid_map.getGridSize(); x++)
        {
            int index = y * grid_map.getGridSize() + x;

            const dogm::GridCell& cell = grid_map.grid_cell_array[index];
            float occ = pignistic_transformation(cell.free_mass, cell.occ_mass);
            uchar temp = static_cast<uchar>(floor(occ * 255));

            cv::Mat vel_img(2, 1, CV_32FC1);
            vel_img.at<float>(0) = cell.mean_x_vel;
            vel_img.at<float>(1) = cell.mean_y_vel;

            cv::Mat covar_img(2, 2, CV_32FC1);
            covar_img.at<float>(0, 0) = cell.var_x_vel;
            covar_img.at<float>(1, 0) = cell.covar_xy_vel;
            covar_img.at<float>(0, 1) = cell.covar_xy_vel;
            covar_img.at<float>(1, 1) = cell.var_y_vel;

            cv::Mat mdist = vel_img.t() * covar_img.inv() * vel_img;

            if (occ >= occ_tresh && mdist.at<float>(0, 0) >= m_tresh)
            {
                float angle = fmodf((atan2(cell.mean_y_vel, cell.mean_x_vel) * (180.0f / PI)) + 360, 360);

                // printf("Angle: %f\n", angle);

                // OpenCV hue range is [0, 179], see https://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html
                const auto hue_opencv = static_cast<uint8_t>(angle * 0.5F);
                cv::Mat hsv{1, 1, CV_8UC3, cv::Scalar(hue_opencv, 255, 255)};
                cv::Mat rgb{1, 1, CV_8UC3};
                cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
                grid_img.at<cv::Vec3b>(y, x) = rgb.at<cv::Vec3b>(0, 0);

                // printf("Vel Y: %f\n", cell.mean_y_vel);
            }
            else
            {
                grid_img.at<cv::Vec3b>(y, x) = cv::Vec3b(255 - temp, 255 - temp, 255 - temp);
            }
        }
    }

    addColorWheelToBottomRightCorner(grid_img);

    return grid_img;
}

cv::Mat compute_particles_image(const dogm::DOGM& grid_map)
{
    cv::Mat particles_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < grid_map.particle_count; i++)
    {
        const dogm::Particle& part = grid_map.particle_array[i];
        float x = part.state[0];
        float y = part.state[1];

        if ((x >= 0 && x < grid_map.getGridSize()) && (y >= 0 && y < grid_map.getGridSize()))
        {
            particles_img.at<cv::Vec3b>(static_cast<int>(y), static_cast<int>(x)) = cv::Vec3b(0, 0, 255);
        }
    }

    return particles_img;
}

std::vector<float2> load_measurement_from_image(const std::string& file_name)
{
    cv::Mat grid_img = cv::imread(file_name);

    // cv::namedWindow("img", cv::WINDOW_NORMAL);
    // cv::imshow("img", grid_img);
    // cv::waitKey(0);

    std::vector<float2> meas_grid(grid_img.cols * grid_img.rows);

    for (int y = 0; y < grid_img.rows; y++)
    {
        for (int x = 0; x < grid_img.cols; x++)
        {
            int index = y * grid_img.cols + x;

            cv::Vec3b color = grid_img.at<cv::Vec3b>(y, x);

            meas_grid[index] = make_float2(color[0] / 255.0f, color[1] / 255.0f);
        }
    }

    return meas_grid;
}

int main(int argc, const char** argv)
{
    std::vector<std::vector<float2>> mg_meas;

    for (int i = 0; i < 10; i++)
    {
        std::vector<float2> grid = load_measurement_from_image(cv::format("meas_grids/meas_grid%d.png", i));
        mg_meas.push_back(grid);
    }

// clang-format off
#if 0
	dogm::GridParams params;
	params.size = 128;
	params.resolution = 1.0f;
	params.particle_count = 2 * static_cast<int>(10e5);
	params.new_born_particle_count = 2 * static_cast<int>(10e4);
	params.persistence_prob = 0.99f;
	params.process_noise_position = 0.06f;
	params.process_noise_velocity = 2.4f;
	params.birth_prob = 0.02f;
	params.velocity_persistent = 12.0f;
	params.velocity_birth = 12.0f;

	dogm::LaserSensorParams laser_params;
	laser_params.fov = 120.0f;
	laser_params.max_range = 50.0f;

	dogm::DOGM grid_map(params, laser_params);

	float delta_time = 0.1f;
	for (int i = 0; i < mg_meas.size(); i++)
	{
		// Update measurement grid
		grid_map.updateMeasurementGridFromArray(mg_meas[i]);
#else
	dogm::GridParams params;
	params.size = 50.0f;
	params.resolution = 0.2f;
	params.particle_count = 3 * static_cast<int>(10e5);
	params.new_born_particle_count = 3 * static_cast<int>(10e4);
	params.persistence_prob = 0.99f;
	params.process_noise_position = 0.02f;
	params.process_noise_velocity = 0.8f;
	params.birth_prob = 0.02f;
	params.velocity_persistent = 30.0f;
	params.velocity_birth = 30.0f;

	dogm::LaserSensorParams laser_params;
	laser_params.fov = 120.0f;
	laser_params.max_range = 50.0f;

	// Just to init cuda
	cudaDeviceSynchronize();

	auto begin = chrono::high_resolution_clock::now();

	dogm::DOGM grid_map(params, laser_params);

	auto end = chrono::high_resolution_clock::now();
	auto dur = end - begin;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	std::cout << "### DOGM initialization took: " << ms << " ms" << " ###" << std::endl << std::endl;

	Simulator simulator(100);
	simulator.addVehicle(Vehicle(6, glm::vec2(20, 10), glm::vec2(0, 0)));
//	simulator.addVehicle(Vehicle(5, glm::vec2(46, 20), glm::vec2(0, 20)));
//	simulator.addVehicle(Vehicle(4, glm::vec2(80, 30), glm::vec2(0, -10)));

	simulator.addVehicle(Vehicle(6, glm::vec2(40, 30), glm::vec2(20, 5)));
	simulator.addVehicle(Vehicle(5, glm::vec2(80, 24), glm::vec2(-15, -5)));

	float delta_time = 0.1f;
	std::vector<std::vector<float>> sim_measurements = simulator.update(10, delta_time);

	for (int i = 0; i < sim_measurements.size(); i++)
	{
		grid_map.updateMeasurementGrid(sim_measurements[i].data(), sim_measurements[i].size());
#endif
		begin = chrono::high_resolution_clock::now();

		// Run Particle filter
		grid_map.updateParticleFilter(delta_time);

		end = chrono::high_resolution_clock::now();
		dur = end - begin;
		ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::cout << "### Iteration took: " << ms << " ms" << " ###" << std::endl;
		std::cout << "######  Saving result  #######" << std::endl;
		std::cout << "##############################" << std::endl;

		cv::Mat meas_grid_img = compute_measurement_grid_image(grid_map);
		cv::imwrite(cv::format("meas_grid_iter-%d.png", i + 1), meas_grid_img);

		cv::Mat raw_meas_grid_img = compute_raw_measurement_grid_image(grid_map);
		cv::imwrite(cv::format("raw_grid_iter-%d.png", i + 1), raw_meas_grid_img);

		cv::Mat grid_img = compute_dogm_image(grid_map, 0.7f, 4.0f);
		cv::imwrite(cv::format("dogm_iter-%d.png", i + 1), grid_img);

		cv::Mat particle_img = compute_particles_image(grid_map);
		cv::imwrite(cv::format("particles_iter-%d.png", i + 1), particle_img);
	}

#if	1
	cv::Mat particle_img = compute_particles_image(grid_map);
	cv::Mat grid_img = compute_dogm_image(grid_map, 0.7f, 4.0f);

	cv::namedWindow("particles", cv::WINDOW_NORMAL);
	cv::imshow("particles", particle_img);

	cv::namedWindow("dynamic_grid", cv::WINDOW_NORMAL);
	cv::imshow("dynamic_grid", grid_img);

	cv::waitKey(0);
#endif

	return 0;
// clang-format on
}