#include "metrics.h"

PointWithVelocity MAE::update(const PointWithVelocity& cluster_mean, const Vehicle& vehicle)
{
    const auto error = computeError(cluster_mean, vehicle);

    cumulative_error.x += std::abs(error.x);
    cumulative_error.y += std::abs(error.y);
    cumulative_error.v_x += std::abs(error.v_x);
    cumulative_error.v_y += std::abs(error.v_y);

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

    cumulative_error.x += std::powf(error.x, 2.0f);
    cumulative_error.y += std::powf(error.y, 2.0f);
    cumulative_error.v_x += std::powf(error.v_x, 2.0f);
    cumulative_error.v_y += std::powf(error.v_y, 2.0f);

    ++number_of_detections;

    return error;
}

PointWithVelocity RMSE::compute()
{
    PointWithVelocity error;
    error.x = std::sqrtf(cumulative_error.x / number_of_detections);
    error.y = std::sqrtf(cumulative_error.y / number_of_detections);
    error.v_x = std::sqrtf(cumulative_error.v_x / number_of_detections);
    error.v_y = std::sqrtf(cumulative_error.v_y / number_of_detections);

    return error;
}
