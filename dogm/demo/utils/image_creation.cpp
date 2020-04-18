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

#include "image_creation.h"
#include "color_wheel_adder.h"
#include "dbscan.h"
#include "dogm/dogm.h"
#include "dogm/dogm_types.h"

#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

static float pignistic_transformation(float free_mass, float occ_mass)
{
    return occ_mass + 0.5f * (1.0f - occ_mass - free_mass);
}

std::vector<Point<dogm::GridCell>> computeCellsWithVelocity(const dogm::DOGM& grid_map, float occ_tresh, float m_tresh)
{
    std::vector<Point<dogm::GridCell>> cells_with_velocity;
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        for (int x = 0; x < grid_map.getGridSize(); x++)
        {
            int index = y * grid_map.getGridSize() + x;

            const dogm::GridCell& cell = grid_map.grid_cell_array[index];
            float occ = pignistic_transformation(cell.free_mass, cell.occ_mass);
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
                Point<dogm::GridCell> point;
                point.x = x;
                point.y = y;
                point.data = cell;
                point.cluster_id = UNCLASSIFIED;

                cells_with_velocity.push_back(point);
            }
        }
    }

    return cells_with_velocity;
}

cv::Mat compute_measurement_grid_image(const dogm::DOGM& grid_map)
{
    cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        cv::Vec3b* row_ptr = grid_img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < grid_map.getGridSize(); x++)
        {
            int index = y * grid_map.getGridSize() + x;

            const dogm::MeasurementCell& cell = grid_map.meas_cell_array[index];
            float occ = pignistic_transformation(cell.free_mass, cell.occ_mass);
            uchar temp = static_cast<uchar>(occ * 255.0f);

            row_ptr[x] = cv::Vec3b(255 - temp, 255 - temp, 255 - temp);
        }
    }

    return grid_img;
}

cv::Mat compute_raw_measurement_grid_image(const dogm::DOGM& grid_map)
{
    cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        cv::Vec3b* row_ptr = grid_img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < grid_map.getGridSize(); x++)
        {
            int index = y * grid_map.getGridSize() + x;
            const dogm::MeasurementCell& cell = grid_map.meas_cell_array[index];
            int red = static_cast<int>(cell.occ_mass * 255.0f);
            int green = static_cast<int>(cell.free_mass * 255.0f);
            int blue = 255 - red - green;

            row_ptr[x] = cv::Vec3b(blue, green, red);
        }
    }

    return grid_img;
}

cv::Mat compute_raw_polar_measurement_grid_image(const dogm::DOGM& grid_map)
{
    cv::Mat grid_img(grid_map.getGridSize(), 100, CV_8UC3);
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        cv::Vec3b* row_ptr = grid_img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < 100; x++)
        {
            int index = y * 100 + x;
            const dogm::MeasurementCell& cell = grid_map.polar_meas_cell_array[index];
            int red = static_cast<int>(cell.occ_mass * 255.0f);
            int green = static_cast<int>(cell.free_mass * 255.0f);
            int blue = 255 - red - green;

            row_ptr[x] = cv::Vec3b(blue, green, red);
        }
    }

    return grid_img;
}

cv::Mat compute_dogm_image(const dogm::DOGM& grid_map, const std::vector<Point<dogm::GridCell>>& cells_with_velocity)
{
    cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        cv::Vec3b* row_ptr = grid_img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < grid_map.getGridSize(); x++)
        {
            int index = y * grid_map.getGridSize() + x;

            const dogm::GridCell& cell = grid_map.grid_cell_array[index];
            float occ = pignistic_transformation(cell.free_mass, cell.occ_mass);
            uchar grayscale_value = 255 - static_cast<uchar>(floor(occ * 255));

            row_ptr[x] = cv::Vec3b(grayscale_value, grayscale_value, grayscale_value);
        }
    }

    for (const auto& cell : cells_with_velocity)
    {
        float angle = fmodf((atan2(cell.data.mean_y_vel, cell.data.mean_x_vel) * (180.0f / M_PI)) + 360, 360);

        // printf("Angle: %f\n", angle);

        // OpenCV hue range is [0, 179], see https://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html
        const auto hue_opencv = static_cast<uint8_t>(angle * 0.5F);
        cv::Mat hsv{1, 1, CV_8UC3, cv::Scalar(hue_opencv, 255, 255)};
        cv::Mat rgb{1, 1, CV_8UC3};
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);

        grid_img.ptr<cv::Vec3b>(static_cast<int>(cell.y))[static_cast<int>(cell.x)] = rgb.at<cv::Vec3b>(0, 0);
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
