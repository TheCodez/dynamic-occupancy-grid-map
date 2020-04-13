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

struct SimulationStep
{
    std::vector<Vehicle> vehicles;
    std::vector<float> measurements;
};

using SimulationData = std::vector<SimulationStep>;

struct Simulator
{
    Simulator(int _num_horizontal_scan_points, const float _field_of_view)
        : num_horizontal_scan_points(_num_horizontal_scan_points), field_of_view(_field_of_view)
    {
    }

    void addVehicle(const Vehicle& vehicle) { vehicles.push_back(vehicle); }

    float regressAngleOffset(float angle_difference)
    {
        angle_difference /= 100.0f;
        std::cout << angle_difference << " " << powf(angle_difference, 3.0f) << std::endl;
        return 76.7253f * powf(angle_difference, 3.0f) - 31.1917f * powf(angle_difference, 2.0f) + 66.6564 * angle_difference - 0.3819;
    }

    SimulationData update(int steps, float dt)
    {
        SimulationData sim_data;

        for (int i = 0; i < steps; i++)
        {
            std::vector<float> measurement(num_horizontal_scan_points, INFINITY);

            for (auto& vehicle : vehicles)
            {
                vehicle.move(dt);

                const float sensor_pos_x = 50;
                const float factor_angle_to_grid = (num_horizontal_scan_points / M_PI) * (180.0f / field_of_view);
                std::cout << regressAngleOffset(180.0f - field_of_view) << "\n";
                const float angle_offset =
                    num_horizontal_scan_points * ((regressAngleOffset(180.0f - field_of_view)) / 180.0f);

                const float supersampling = 20.0f;
                for (int i = 0; i < vehicle.width * static_cast<int>(supersampling); ++i)
                {
                    const float x = vehicle.pos.x + static_cast<float>(i) / supersampling - sensor_pos_x;
                    const float radius = sqrtf(powf(x, 2) + powf(vehicle.pos.y, 2));

                    const float angle = M_PI - atan2(vehicle.pos.y, x);
                    const float angle_normalized_to_grid = (angle * factor_angle_to_grid) - angle_offset;

                    int index = static_cast<int>(angle_normalized_to_grid);
                    // std::cout << "x y = " << vehicle.pos.x << " " << vehicle.pos.y << "\n";
                    // std::cout << "r t = " << radius << " " << index << "\n";
                    measurement[index] = radius;

                    // Previous implementation
                    // int m_index = static_cast<int>(vehicle.pos.x) + i;
                    // measurement[m_index] = vehicle.pos.y;
                }
            }

            SimulationStep step;
            step.measurements = measurement;
            step.vehicles = vehicles;
            sim_data.push_back(step);
        }

        return sim_data;
    }

    int num_horizontal_scan_points;
    float field_of_view;
    std::vector<Vehicle> vehicles;
};

#endif  // SIMULATOR_H
