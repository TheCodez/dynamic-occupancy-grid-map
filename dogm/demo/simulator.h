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

#ifndef SIMULATOR_H
#define SIMULATOR_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <glm/glm.hpp>
#include <vector>

struct Vehicle
{
    Vehicle(const int width, const glm::vec2& pos, const glm::vec2& vel) : width(width), pos(pos), vel(vel) {}

    void move(float dt) { pos += vel * dt; }

    int width;
    glm::vec2 pos;
    glm::vec2 vel;
};

struct Simulator
{
    Simulator(int _num_horizontal_scan_points, const float _field_of_view)
        : num_horizontal_scan_points(_num_horizontal_scan_points), field_of_view(_field_of_view)
    {
    }

    void addVehicle(const Vehicle& vehicle) { vehicles.push_back(vehicle); }

    std::vector<std::vector<float>> update(int steps, float dt)
    {
        std::vector<std::vector<float>> measurements;

        for (int i = 0; i < steps; i++)
        {
            std::vector<float> measurement(num_horizontal_scan_points, INFINITY);

            for (auto& vehicle : vehicles)
            {
                vehicle.move(dt);
                addVehicleDetectionsToMeasurement(vehicle, measurement);
            }

            measurements.push_back(measurement);
        }

        return measurements;
    }

    void addVehicleDetectionsToMeasurement(const Vehicle& vehicle, std::vector<float>& measurement)
    {
        const float sensor_position_x = 50;
        const float factor_angle_to_grid = (num_horizontal_scan_points / M_PI) * (180.0f / field_of_view);
        const float angle_offset = num_horizontal_scan_points * (regressAngleOffset(180.0f - field_of_view) / 180.0f);

        const float supersampling = 20.0f;
        const int num_sample_points = vehicle.width * static_cast<int>(supersampling);
        for (int point_on_vehicle = 0; point_on_vehicle < num_sample_points; ++point_on_vehicle)
        {
            const float x = vehicle.pos.x + static_cast<float>(point_on_vehicle) / supersampling - sensor_position_x;
            const float radius = sqrtf(powf(x, 2) + powf(vehicle.pos.y, 2));

            const float angle = M_PI - atan2(vehicle.pos.y, x);
            const float angle_normalized_to_grid = (angle * factor_angle_to_grid) - angle_offset;

            int index = static_cast<int>(angle_normalized_to_grid);
            if (0 <= index && index <= measurement.size())
                measurement[index] = radius;
        }
    }

    float regressAngleOffset(float angle_difference)
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

    int num_horizontal_scan_points;
    float field_of_view;
    std::vector<Vehicle> vehicles;
};

#endif  // SIMULATOR_H
