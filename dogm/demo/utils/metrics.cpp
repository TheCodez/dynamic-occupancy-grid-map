#include "metrics.h"

#include <iostream>

MeanSquaredError::MeanSquaredError() : cumulative_error{}, number_of_detections(0)
{
}

void MeanSquaredError::update(const PointWithVelocity& cluster_mean, const Vehicle& vehicle)
{
    const auto error = computeError(cluster_mean, vehicle);

    cumulative_error.x += std::abs(error.x);
    cumulative_error.y += std::abs(error.y);
    cumulative_error.v_x += std::abs(error.v_x);
    cumulative_error.v_y += std::abs(error.v_y);

    ++number_of_detections;
}

void MeanSquaredError::compute()
{
    std::cout << "Position: " << cumulative_error.x / number_of_detections << " "
              << cumulative_error.y / number_of_detections << "\n";
    std::cout << "Velocity: " << cumulative_error.v_x / number_of_detections << " "
              << cumulative_error.v_y / number_of_detections << "\n\n";
}
