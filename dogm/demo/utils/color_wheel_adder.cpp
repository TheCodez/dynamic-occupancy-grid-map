// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "color_wheel_adder.h"

#include <opencv2/opencv.hpp>

static cv::Mat createCircularColorGradient(const int hue_offset)
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

static cv::Mat createFullSaturationFullValueAlphaMaskOf(const cv::Mat& img)
{
    cv::Mat mask{};
    cv::Mat img_hsv{};
    cv::cvtColor(img, img_hsv, cv::COLOR_RGB2HSV);
    cv::inRange(img_hsv, cv::Scalar(0, 254, 254), cv::Scalar(180, 255, 255), mask);
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2RGB);
    return mask;
}

static cv::Mat createCircleMask(const cv::Size2d& size)
{
    cv::Mat mask{size, CV_8UC3, cv::Scalar(0, 0, 0)};
    const cv::Point image_center{mask.cols / 2, mask.rows / 2};
    const auto outer_radius = mask.rows / 2;
    const auto inner_radius = mask.rows / 4;
    cv::circle(mask, image_center, outer_radius, cv::Scalar(1, 1, 1), -1);
    cv::circle(mask, image_center, inner_radius, cv::Scalar(0, 0, 0), -1);
    return mask;
}

static void resizeRelativeToShorterEdge(const cv::Mat& original_image, const float relative_size,
                                        cv::Mat& destination_image)
{
    const float target_edge_length = std::min(original_image.cols, original_image.rows) * relative_size;
    cv::resize(destination_image, destination_image, cv::Size(target_edge_length, target_edge_length));
}

static cv::Mat createColorWheel(const int hue_offset)
{
    cv::Mat circular_color_gradient = createCircularColorGradient(hue_offset);
    cv::Mat circle_mask = createCircleMask(circular_color_gradient.size());

    cv::Mat color_wheel{};
    circular_color_gradient.copyTo(color_wheel, circle_mask);
    return color_wheel;
}

static void blendIntoBottomRightCorner(const cv::Mat& img_to_blend_in, cv::Mat& img)
{
    const cv::Rect roi{img.cols - img_to_blend_in.cols, img.rows - img_to_blend_in.rows, img_to_blend_in.cols,
                       img_to_blend_in.rows};
    cv::Mat img_roi = img(roi);
    cv::Mat circle_mask = createFullSaturationFullValueAlphaMaskOf(img_to_blend_in);
    cv::multiply(cv::Scalar::all(1.0) - circle_mask, img_roi, img_roi);
    cv::addWeighted(img_roi, 1.0F, img_to_blend_in, 1.0F, 0.0, img_roi);
}

void addColorWheelToBottomRightCorner(cv::Mat& img, const float relative_size, const int hue_offset)
{
    cv::Mat color_wheel = createColorWheel(hue_offset);
    resizeRelativeToShorterEdge(img, relative_size, color_wheel);
    blendIntoBottomRightCorner(color_wheel, img);
}
