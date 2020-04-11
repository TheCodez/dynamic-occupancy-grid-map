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

#ifndef IMAGE_CREATION_H
#define IMAGE_CREATION_H

#include "dogm/dogm.h"
#include "dogm/dogm_types.h"

#include "dbscan.h"

#define _USE_MATH_DEFINES
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

inline float pignistic_transformation(float free_mass, float occ_mass)
{
    return occ_mass + 0.5f * (1.0f - occ_mass - free_mass);
}

inline cv::Mat compute_measurement_grid_image(const dogm::DOGM& grid_map)
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

inline cv::Mat compute_raw_measurement_grid_image(const dogm::DOGM& grid_map)
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

inline cv::Mat compute_raw_polar_measurement_grid_image(const dogm::DOGM& grid_map)
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

inline cv::Mat createCircularColorGradient(const int hue_offset)
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

inline cv::Mat createCircleMask(const cv::Size2d& size)
{
    cv::Mat mask{size, CV_8UC3, cv::Scalar(0, 0, 0)};
    const cv::Point image_center{mask.cols / 2, mask.rows / 2};
    const float outer_radius = mask.rows / 2;
    const float inner_radius = mask.rows / 4;
    cv::circle(mask, image_center, outer_radius, cv::Scalar(1, 1, 1), -1);
    cv::circle(mask, image_center, inner_radius, cv::Scalar(0, 0, 0), -1);
    return mask;
}

inline cv::Mat createColorWheel(const int hue_offset = 0)
{
    cv::Mat circular_color_gradient = createCircularColorGradient(hue_offset);
    cv::Mat circle_mask = createCircleMask(circular_color_gradient.size());

    cv::Mat color_wheel{};
    circular_color_gradient.copyTo(color_wheel, circle_mask);
    return color_wheel;
}

inline void resizeRelativeToShorterEdge(const cv::Mat& original_image, const float relative_size,
                                        cv::Mat& destination_image)
{
    const float target_edge_length = std::min(original_image.cols, original_image.rows) * relative_size;
    cv::resize(destination_image, destination_image, cv::Size(target_edge_length, target_edge_length));
}

inline cv::Mat createFullSaturationFullValueAlphaMaskOf(const cv::Mat& img)
{
    cv::Mat mask{};
    cv::Mat img_hsv{};
    cv::cvtColor(img, img_hsv, cv::COLOR_RGB2HSV);
    cv::inRange(img_hsv, cv::Scalar(0, 254, 254), cv::Scalar(180, 255, 255), mask);
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2RGB);
    return mask;
}

inline void blendIntoBottomRightCorner(const cv::Mat& img_to_blend_in, cv::Mat& img)
{
    const cv::Rect roi{img.cols - img_to_blend_in.cols, img.rows - img_to_blend_in.rows, img_to_blend_in.cols,
                       img_to_blend_in.rows};
    cv::Mat img_roi = img(roi);
    cv::Mat mask = createFullSaturationFullValueAlphaMaskOf(img_to_blend_in);
    cv::multiply(cv::Scalar::all(1.0) - mask, img_roi, img_roi);
    cv::addWeighted(img_roi, 1.0F, img_to_blend_in, 1.0F, 0.0, img_roi);
}

inline void addColorWheelToBottomRightCorner(cv::Mat& img, const float relative_size = 0.2, const int hue_offset = 0)
{
    cv::Mat color_wheel = createColorWheel(hue_offset);
    resizeRelativeToShorterEdge(img, relative_size, color_wheel);
    blendIntoBottomRightCorner(color_wheel, img);
}

inline std::vector<Point<dogm::GridCell>> computeCellsWithVelocity(const dogm::DOGM& grid_map, float occ_tresh = 0.7f,
                                                                   float m_tresh = 4.0f)
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

inline cv::Mat compute_dogm_image(const dogm::DOGM& grid_map,
                                  const std::vector<Point<dogm::GridCell>>& cells_with_velocity)
{

    cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        for (int x = 0; x < grid_map.getGridSize(); x++)
        {
            int index = y * grid_map.getGridSize() + x;

            const dogm::GridCell& cell = grid_map.grid_cell_array[index];
            float occ = pignistic_transformation(cell.free_mass, cell.occ_mass);
            uchar grayscale_value = 255 - static_cast<uchar>(floor(occ * 255));

            grid_img.at<cv::Vec3b>(y, x) = cv::Vec3b(grayscale_value, grayscale_value, grayscale_value);
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
        grid_img.at<cv::Vec3b>(cell.y, cell.x) = rgb.at<cv::Vec3b>(0, 0);
    }

    addColorWheelToBottomRightCorner(grid_img);

    return grid_img;
}

inline cv::Mat compute_particles_image(const dogm::DOGM& grid_map)
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

inline std::vector<float2> load_measurement_from_image(const std::string& file_name)
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

#endif  // IMAGE_CREATION_H
