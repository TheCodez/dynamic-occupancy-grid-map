// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "simulator.h"

#include <cmath>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>

static float regressAngleOffset(float angle_difference)
{
    // TODO find a final solution for this. The current regression is only an approximation to the true,
    // unkown function. Error is mostly <0.5 degrees, <0.05 for an fov of 120 degrees.
    // Expected results (found manually):
    // angle_difference==90: return 90
    // angle_difference==60: return 45
    // angle_difference==45: return 30
    // angle_difference==30: return 20
    // angle_difference==15: return 7.5
    // angle_difference== 0: return 0
    angle_difference /= 100.0f;
    return 76.7253f * powf(angle_difference, 3.0f) - 31.1917f * powf(angle_difference, 2.0f) +
           66.6564 * angle_difference - 0.3819;
}

SimulationData Simulator::update(int steps, float dt)
{
    SimulationData sim_data;

    for (int i = 0; i < steps; i++)
    {
        std::vector<float> measurement(num_horizontal_scan_points, INFINITY);

        for (auto& vehicle : vehicles)
        {
            vehicle.move(dt);
            addVehicleDetectionsToMeasurement(vehicle, measurement);
        }

        SimulationStep step;
        step.measurements = measurement;
        step.vehicles = vehicles;
        sim_data.push_back(step);
    }

    return sim_data;
}

void Simulator::addVehicleDetectionsToMeasurement(const Vehicle& vehicle, std::vector<float>& measurement) const
{
    const float max_field_of_view = 180.0f;
    const float sensor_position_x = num_horizontal_scan_points / 2;
    const float factor_angle_to_grid = (num_horizontal_scan_points / M_PI) * (max_field_of_view / field_of_view);
    const float angle_offset =
        num_horizontal_scan_points * (regressAngleOffset(max_field_of_view - field_of_view) / max_field_of_view);

    const float supersampling = 20.0f;
    const int num_sample_points = vehicle.width * static_cast<int>(supersampling);
    for (int point_on_vehicle = 0; point_on_vehicle < num_sample_points; ++point_on_vehicle)
    {
        const float x =
            mapper.mapAbsoluteGridPositionToRelativePosition(vehicle.pos.x) * float(num_horizontal_scan_points) +
            static_cast<float>(point_on_vehicle) / supersampling - sensor_position_x;
        const float radius = sqrtf(powf(x, 2) + powf(vehicle.pos.y, 2));

        const float angle = M_PI - atan2(vehicle.pos.y, x);
        const float angle_normalized_to_measurement_vector = (angle * factor_angle_to_grid) - angle_offset;

        int index = static_cast<int>(angle_normalized_to_measurement_vector);
        if (0 <= index && index <= measurement.size())
            measurement[index] = radius;
    }
}
