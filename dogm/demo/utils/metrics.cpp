// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "metrics.h"
#include <cmath>

PointWithVelocity MAE::addObjectDetection(const PointWithVelocity& cluster_mean, const Vehicle& vehicle)
{
    const auto error = computeError(cluster_mean, vehicle);

    cumulative_error.x += std::fabs(error.x);
    cumulative_error.y += std::fabs(error.y);
    cumulative_error.v_x += std::fabs(error.v_x);
    cumulative_error.v_y += std::fabs(error.v_y);

    ++number_of_detections;

    return error;
}

PointWithVelocity MAE::computeErrorStatistic()
{
    PointWithVelocity error;
    error.x = cumulative_error.x / number_of_detections;
    error.y = cumulative_error.y / number_of_detections;
    error.v_x = cumulative_error.v_x / number_of_detections;
    error.v_y = cumulative_error.v_y / number_of_detections;

    return error;
}

PointWithVelocity RMSE::addObjectDetection(const PointWithVelocity& cluster_mean, const Vehicle& vehicle)
{
    const auto error = computeError(cluster_mean, vehicle);

    cumulative_error.x += powf(error.x, 2.0f);
    cumulative_error.y += powf(error.y, 2.0f);
    cumulative_error.v_x += powf(error.v_x, 2.0f);
    cumulative_error.v_y += powf(error.v_y, 2.0f);

    ++number_of_detections;

    return error;
}

PointWithVelocity RMSE::computeErrorStatistic()
{
    PointWithVelocity error;
    error.x = sqrtf(cumulative_error.x / number_of_detections);
    error.y = sqrtf(cumulative_error.y / number_of_detections);
    error.v_x = sqrtf(cumulative_error.v_x / number_of_detections);
    error.v_y = sqrtf(cumulative_error.v_y / number_of_detections);

    return error;
}
