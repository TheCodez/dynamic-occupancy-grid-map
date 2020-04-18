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

#include "dbscan.h"
#include "dogm/dogm.h"

#include <opencv2/opencv.hpp>

#include <vector>

std::vector<Point<dogm::GridCell>> computeCellsWithVelocity(const dogm::DOGM& grid_map, float occ_tresh = 0.7f,
                                                            float m_tresh = 4.0f);

cv::Mat compute_measurement_grid_image(const dogm::DOGM& grid_map);
cv::Mat compute_raw_measurement_grid_image(const dogm::DOGM& grid_map);
cv::Mat compute_raw_polar_measurement_grid_image(const dogm::DOGM& grid_map);
cv::Mat compute_dogm_image(const dogm::DOGM& grid_map, const std::vector<Point<dogm::GridCell>>& cells_with_velocity);
cv::Mat compute_particles_image(const dogm::DOGM& grid_map);

#endif  // IMAGE_CREATION_H
