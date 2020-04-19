// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef COLOR_WHEEL_ADDER_H
#define COLOR_WHEEL_ADDER_H

#include <opencv2/opencv.hpp>

void addColorWheelToBottomRightCorner(cv::Mat& img, const float relative_size = 0.2, const int hue_offset = 0);

#endif  // COLOR_WHEEL_ADDER_H
