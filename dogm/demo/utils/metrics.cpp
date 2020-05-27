// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "metrics.h"
#include <cmath>

PointWithVelocity MAE::update(const PointWithVelocity& cluster_mean, const Vehicle& vehicle)
{
    const auto error = computeError(cluster_mean, vehicle);

    cumulative_error.x += abs(error.x);
    cumulative_error.y += abs(error.y);
    cumulative_error.v_x += abs(error.v_x);
    cumulative_error.v_y += abs(error.v_y);

    ++number_of_detections;

    return error;
}

PointWithVelocity MAE::compute()
{
    PointWithVelocity error;
    error.x = cumulative_error.x / number_of_detections;
    error.y = cumulative_error.y / number_of_detections;
    error.v_x = cumulative_error.v_x / number_of_detections;
    error.v_y = cumulative_error.v_y / number_of_detections;

    return error;
}

PointWithVelocity RMSE::update(const PointWithVelocity& cluster_mean, const Vehicle& vehicle)
{
    const auto error = computeError(cluster_mean, vehicle);

    cumulative_error.x += powf(error.x, 2.0f);
    cumulative_error.y += powf(error.y, 2.0f);
    cumulative_error.v_x += powf(error.v_x, 2.0f);
    cumulative_error.v_y += powf(error.v_y, 2.0f);

    ++number_of_detections;

    return error;
}

PointWithVelocity RMSE::compute()
{
    PointWithVelocity error;
    error.x = sqrtf(cumulative_error.x / number_of_detections);
    error.y = sqrtf(cumulative_error.y / number_of_detections);
    error.v_x = sqrtf(cumulative_error.v_x / number_of_detections);
    error.v_y = sqrtf(cumulative_error.v_y / number_of_detections);

    return error;
}
