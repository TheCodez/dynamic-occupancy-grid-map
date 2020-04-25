// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef IMAGE_CREATION_H
#define IMAGE_CREATION_H

#include "dbscan.h"
#include "dogm/dogm.h"

#include <opencv2/opencv.hpp>

#include <vector>

std::vector<Point<dogm::GridCell>> computeCellsWithVelocity(const dogm::DOGM& grid_map,
                                                            float min_occupancy_threshold = 0.7f,
                                                            float min_velocity_threshold = 4.0f);

cv::Mat compute_measurement_grid_image(const dogm::DOGM& grid_map);
cv::Mat compute_raw_measurement_grid_image(const dogm::DOGM& grid_map);
cv::Mat compute_raw_polar_measurement_grid_image(const dogm::DOGM& grid_map);
cv::Mat compute_dogm_image(const dogm::DOGM& grid_map, const std::vector<Point<dogm::GridCell>>& cells_with_velocity);
cv::Mat compute_particles_image(const dogm::DOGM& grid_map);

void computeAndSaveResultImages(const dogm::DOGM& grid_map,
                                const std::vector<Point<dogm::GridCell>>& cells_with_velocity, const int step,
                                const bool concatenate_images = true, const bool show_during_execution = true);
#endif  // IMAGE_CREATION_H
