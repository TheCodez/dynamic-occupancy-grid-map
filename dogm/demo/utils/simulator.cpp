// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "simulator.h"

#include <cmath>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>

std::vector<glm::vec2> Vehicle::getPointsOnFacingSide(const float resolution) const
{
    assert(resolution > 0.0f);
    std::vector<glm::vec2> points_on_facing_side{};
    const auto leftmost_point_on_facing_side = pos.x - width * 0.5f;
    const auto rightmost_point_on_facing_side = pos.x + width * 0.5f;
    for (float point_on_facing_side = leftmost_point_on_facing_side;
         point_on_facing_side < rightmost_point_on_facing_side; point_on_facing_side += resolution)
    {
        points_on_facing_side.push_back(glm::vec2{point_on_facing_side, pos.y});
    }
    return points_on_facing_side;
}

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

Simulator::Simulator(int _num_horizontal_scan_points, const float _field_of_view, const float grid_size)
    : num_horizontal_scan_points(_num_horizontal_scan_points), field_of_view(_field_of_view), grid_size{grid_size}
{
    const float max_field_of_view = 180.0f;
    factor_angle_to_grid = (num_horizontal_scan_points / M_PI) * (max_field_of_view / field_of_view);
    angle_offset =
        num_horizontal_scan_points * (regressAngleOffset(max_field_of_view - field_of_view) / max_field_of_view);
    sensor_position = glm::vec2{grid_size * 0.5f, 0.0f};
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
    const auto points_on_facing_side = vehicle.getPointsOnFacingSide(grid_size / (20.0f * num_horizontal_scan_points));
    for (const auto& point_on_facing_side : points_on_facing_side)
    {
        const auto point_relative_to_sensor = point_on_facing_side - sensor_position;

        const float radius = sqrtf(powf(point_relative_to_sensor.x, 2) + powf(vehicle.pos.y, 2));
        const float angle = M_PI - atan2(point_relative_to_sensor.y, point_relative_to_sensor.x);
        const auto index = computeMeasurementVectorIndexFromAngle(angle);

        if (0 <= index && index <= measurement.size())
            measurement[index] = radius;
    }
}

int Simulator::computeMeasurementVectorIndexFromAngle(const float angle) const
{
    const float angle_normalized_to_measurement_vector = (angle * factor_angle_to_grid) - angle_offset;
    return static_cast<int>(angle_normalized_to_measurement_vector);
}