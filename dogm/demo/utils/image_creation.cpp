// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "image_creation.h"
#include "color_wheel_adder.h"
#include "dbscan.h"
#include "dogm/dogm.h"
#include "dogm/dogm_types.h"

#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

static float pignistic_transformation(float free_mass, float occ_mass)
{
    return occ_mass + 0.5f * (1.0f - occ_mass - free_mass);
}

std::vector<Point<dogm::GridCell>> computeCellsWithVelocity(const dogm::DOGM& grid_map, float min_occupancy_threshold,
                                                            float min_velocity_threshold)
{
    const auto grid_cells = grid_map.getGridCells();
    std::vector<Point<dogm::GridCell>> cells_with_velocity;
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        for (int x = 0; x < grid_map.getGridSize(); x++)
        {
            int index = y * grid_map.getGridSize() + x;

            const dogm::GridCell& cell = grid_cells[index];
            float occ = pignistic_transformation(cell.free_mass, cell.occ_mass);
            cv::Mat velocity_mean(2, 1, CV_32FC1);
            velocity_mean.at<float>(0) = cell.mean_x_vel;
            velocity_mean.at<float>(1) = cell.mean_y_vel;

            cv::Mat velocity_covar(2, 2, CV_32FC1);
            velocity_covar.at<float>(0, 0) = cell.var_x_vel;
            velocity_covar.at<float>(1, 0) = cell.covar_xy_vel;
            velocity_covar.at<float>(0, 1) = cell.covar_xy_vel;
            velocity_covar.at<float>(1, 1) = cell.var_y_vel;

            cv::Mat velocity_normalized_by_variance = velocity_mean.t() * velocity_covar.inv() * velocity_mean;

            if (occ >= min_occupancy_threshold &&
                velocity_normalized_by_variance.at<float>(0, 0) >= min_velocity_threshold)
            {
                Point<dogm::GridCell> point;

                // Storing the point as grid index to be consistent with cell.mean_x_vel and cell.mean_y_vel
                point.x = static_cast<float>(x);
                point.y = static_cast<float>(y);
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
    const auto meas_cells = grid_map.getMeasurementCells();
    cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        auto* row_ptr = grid_img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < grid_map.getGridSize(); x++)
        {
            int index = y * grid_map.getGridSize() + x;

            const dogm::MeasurementCell& cell = meas_cells[index];
            float occ = pignistic_transformation(cell.free_mass, cell.occ_mass);
            auto temp = static_cast<uchar>(occ * 255.0f);

            row_ptr[x] = cv::Vec3b(255 - temp, 255 - temp, 255 - temp);
        }
    }

    return grid_img;
}

cv::Mat compute_raw_measurement_grid_image(const dogm::DOGM& grid_map)
{
    const auto meas_cells = grid_map.getMeasurementCells();
    cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        auto* row_ptr = grid_img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < grid_map.getGridSize(); x++)
        {
            int index = y * grid_map.getGridSize() + x;
            const dogm::MeasurementCell& cell = meas_cells[index];
            auto red = static_cast<int>(cell.occ_mass * 255.0f);
            auto green = static_cast<int>(cell.free_mass * 255.0f);
            int blue = 255 - red - green;

            row_ptr[x] = cv::Vec3b(blue, green, red);
        }
    }

    return grid_img;
}

cv::Mat compute_dogm_image(const dogm::DOGM& grid_map, const std::vector<Point<dogm::GridCell>>& cells_with_velocity)
{
    const auto grid_cells = grid_map.getGridCells();
    cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
    for (int y = 0; y < grid_map.getGridSize(); y++)
    {
        auto* row_ptr = grid_img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < grid_map.getGridSize(); x++)
        {
            int index = y * grid_map.getGridSize() + x;

            const dogm::GridCell& cell = grid_cells[index];
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
    dogm::ParticlesSoA particles = grid_map.getParticles();
    cv::Mat particles_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < grid_map.particle_count; i++)
    {
        float x = particles.state[i][0];
        float y = particles.state[i][1];

        // TODO normalize this to the maximum particle count found in a cell. Currently, does not depict if more than
        // 3*256 particles accumulate in one cell
        if ((x >= 0 && x < grid_map.getGridSize()) && (y >= 0 && y < grid_map.getGridSize()))
        {
            auto& cell = particles_img.at<cv::Vec3b>(static_cast<int>(y), static_cast<int>(x));
            if (cell[1] == 255 && cell[2] == 255)
            {
                cell += cv::Vec3b(1, 0, 0);
            }
            else
            {
                if (cell[2] == 255)
                {
                    cell += cv::Vec3b(0, 1, 0);
                }
                else
                {
                    cell += cv::Vec3b(0, 0, 1);
                }
            }
        }
    }

    return particles_img;
}

static void addTextToCenter(const std::string& text, cv::Mat& img)
{
    int fontFace = cv::FONT_HERSHEY_DUPLEX;
    double fontScale = 1;
    int thickness = 1;
    int baseline{0};
    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
    cv::Point textOrg((img.cols - textSize.width) / 2, (img.rows + textSize.height) / 2);
    cv::putText(img, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
}

static void addSubtitle(const std::string& subtitle, cv::Mat& img)
{
    cv::Mat subtitle_image(img.rows * 0.2f, img.cols, img.type(), cv::Scalar::all(30));
    addTextToCenter(subtitle, subtitle_image);
    cv::vconcat(img, subtitle_image, img);
}

void computeAndSaveResultImages(const dogm::DOGM& grid_map,
                                const std::vector<Point<dogm::GridCell>>& cells_with_velocity, const int step,
                                const bool concatenate_images, const bool show_during_execution)
{
    cv::Mat raw_meas_grid_img = compute_raw_measurement_grid_image(grid_map);
    cv::Mat particle_img = compute_particles_image(grid_map);
    cv::Mat dogm_img = compute_dogm_image(grid_map, cells_with_velocity);

    cv::Mat image_to_show{};
    if (concatenate_images)
    {
        addSubtitle("Grid", dogm_img);
        addSubtitle("Particles", particle_img);
        addSubtitle("Measurement", raw_meas_grid_img);

        cv::hconcat(dogm_img, particle_img, image_to_show);
        cv::hconcat(image_to_show, raw_meas_grid_img, image_to_show);

        cv::imwrite(cv::format("outputs_step_%d.png", step + 1), image_to_show);
    }
    else
    {
        cv::imwrite(cv::format("raw_grid_step_%d.png", step + 1), raw_meas_grid_img);
        cv::imwrite(cv::format("particles_step_%d.png", step + 1), particle_img);
        cv::imwrite(cv::format("dogm_step_%d.png", step + 1), dogm_img);
        image_to_show = dogm_img;
    }

    if (show_during_execution)
    {
        cv::namedWindow("DOGM", cv::WINDOW_NORMAL);
        cv::resizeWindow("DOGM", image_to_show.cols * 2, image_to_show.rows * 2);
        cv::imshow("DOGM", image_to_show);
        cv::waitKey(1);
    }
}
