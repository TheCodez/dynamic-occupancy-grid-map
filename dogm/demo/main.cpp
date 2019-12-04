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
#include "occupancy_grid_map.h"

#include <glm/glm.hpp>

#include <array>
#include <iostream>
#include <stdio.h>

#include <chrono>
#include <cmath>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "measurements.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

using namespace std;

#define PI 3.14159265358979323846f

void hsv_to_rgb(int h, double s, double v, int output[3])
{
	double c = s * v;
	double x = c * (1 - abs(fmod(h / 60.0, 2) - 1));
	double m = v - c;
	double rs, gs, bs;

	if (h >= 0 && h < 60) 
	{
		rs = c;
		gs = x;
		bs = 0;
	}
	else if (h >= 60 && h < 120)
	{
		rs = x;
		gs = c;
		bs = 0;
	}
	else if (h >= 120 && h < 180)
	{
		rs = 0;
		gs = c;
		bs = x;
	}
	else if (h >= 180 && h < 240)
	{
		rs = 0;
		gs = x;
		bs = c;
	}
	else if (h >= 240 && h < 300)
	{
		rs = x;
		gs = 0;
		bs = c;
	}
	else 
	{
		rs = c;
		gs = 0;
		bs = x;
	}

	output[0] = (rs + m) * 255;
	output[1] = (gs + m) * 255;
	output[2] = (bs + m) * 255;
}

float pignistic_transformation(float free_mass, float occ_mass)
{
	return occ_mass + 0.5f * (1.0f - occ_mass - free_mass);
}

cv::Mat compute_measurement_grid_image(const OccupancyGridMap& grid_map)
{
	cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
	for (int y = 0; y < grid_map.getGridSize(); y++)
	{
		for (int x = 0; x < grid_map.getGridSize(); x++)
		{
			int index = y * grid_map.getGridSize() + x;

			const MeasurementCell& cell = grid_map.meas_cell_array[index];
			float occ = pignistic_transformation(cell.free_mass, cell.occ_mass);
			uchar temp = (uchar)floor(occ * 255);
			grid_img.at<cv::Vec3b>(y, x) = cv::Vec3b(255 - temp, 255 - temp, 255 - temp);
		}
	}

	return grid_img;
}

cv::Mat compute_dogm_image(const OccupancyGridMap& grid_map, float occ_tresh = 0.7f, float m_tresh = 4.0f)
{
	cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
	for (int y = 0; y < grid_map.getGridSize(); y++)
	{
		for (int x = 0; x < grid_map.getGridSize(); x++)
		{
			int index = y * grid_map.getGridSize() + x;

			const GridCell& cell = grid_map.grid_cell_array[index];
			float occ = pignistic_transformation(cell.free_mass, cell.occ_mass);
			uchar temp = (uchar)floor(occ * 255);

			cv::Mat vel_img(2, 1, CV_32FC1);
			vel_img.at<float>(0) = cell.mean_x_vel;
			vel_img.at<float>(1) = cell.mean_y_vel;

			cv::Mat covar_img(2, 2, CV_32FC1);
			covar_img.at<float>(0, 0) = cell.var_x_vel;
			covar_img.at<float>(1, 0) = cell.covar_xy_vel;
			covar_img.at<float>(0, 1) = cell.covar_xy_vel;
			covar_img.at<float>(1, 1) = cell.var_y_vel;

			cv::Mat mdist = vel_img.t() * covar_img.inv() * vel_img;

			if (occ >= occ_tresh && mdist.at<float>(0, 0) > m_tresh)
			{
				float angle = atan2(cell.mean_y_vel, cell.mean_x_vel) * (180.0f / PI);

				int color[3];
				hsv_to_rgb(static_cast<int>(ceil(angle)), 1.0, 1.0, color);

				grid_img.at<cv::Vec3b>(y, x) = cv::Vec3b(color[2], color[1], color[0]);

			}
			else
			{
				grid_img.at<cv::Vec3b>(y, x) = cv::Vec3b(255 - temp, 255 - temp, 255 - temp);
			}
		}
	}

	return grid_img;
}

cv::Mat compute_particles_image(const OccupancyGridMap& grid_map)
{
	cv::Mat particles_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < grid_map.particle_count; i++)
	{
		const Particle& part = grid_map.particle_array[i];
		float x = part.state[0];
		float y = part.state[1];

		if ((x >= 0 && x < grid_map.getGridSize()) && (y >= 0 && y < grid_map.getGridSize()))
		{
			particles_img.at<cv::Vec3b>(static_cast<int>(y), static_cast<int>(x)) = cv::Vec3b(0, 0, 255);
		}
	}

	return particles_img;
}

int main(int argc, const char** argv) 
{
	GridParams params;
	params.size = 50;
	params.resolution = 0.1f;
	params.particle_count = 2 * static_cast<int>(10e5);
	params.new_born_particle_count = 2 * static_cast<int>(10e4);
	params.p_S = 0.99f;
	params.process_noise_position = 0.02f;
	params.process_noise_velocity = 0.8f;
	params.p_B = 0.02f;

	LaserSensorParams laser_params;
	laser_params.fov = 120.0f;
	laser_params.max_range = 50.0f;

	OccupancyGridMap grid_map(params, laser_params);

	std::array<float*, 9> measurements = { ranges, ranges2, ranges3, ranges4, ranges4, ranges4, ranges4, ranges4, ranges4 };
//	std::array<float*, 9> measurements = { ranges, ranges2, ranges3, ranges4, ranges4, ranges4, ranges4, ranges4, ranges4 };
//	std::array<float*, 5> measurements = { ranges4, ranges3, ranges2, ranges, ranges };
	int size = sizeof(ranges) / sizeof(ranges[0]);

	for (int i = 0; i < measurements.size(); i++)
	{
		// Update measurement grid
		grid_map.updateMeasurementGrid(measurements[i], size);

		auto begin = chrono::high_resolution_clock::now();

		// Run Particle filter
		grid_map.updateDynamicGrid(0.1f);

		auto end = chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::cout << "Iteration took: " << ms << " ms" << std::endl;
		std::cout << "Saving result" << std::endl;
		std::cout << "#####################" << std::endl;

		cv::Mat meas_grid_img = compute_measurement_grid_image(grid_map);
		cv::imwrite(cv::format("meas_grid_iter-%d.png", i + 1), meas_grid_img);

		cv::Mat grid_img = compute_dogm_image(grid_map, 0.7f, 4.0f);
		cv::imwrite(cv::format("dogm_iter-%d.png", i + 1), grid_img);
	}

#if 1
	cv::Mat particle_img = compute_particles_image(grid_map);
	cv::Mat grid_img = compute_dogm_image(grid_map, 0.7f, 4.0f);

	cv::namedWindow("particles", cv::WINDOW_NORMAL);
	cv::imshow("particles", particle_img);

	cv::namedWindow("dynamic_grid", cv::WINDOW_NORMAL);
	cv::imshow("dynamic_grid", grid_img);

	cv::waitKey(0);
#endif

	return 0;
}