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
    Simulator(int num_measurements) : num_measurements(num_measurements) {}

    void addVehicle(const Vehicle& vehicle) { vehicles.push_back(vehicle); }

    std::vector<std::vector<float>> update(int steps, float dt)
    {
        std::vector<std::vector<float>> measurements;

        for (int i = 0; i < steps; i++)
        {
            std::vector<float> measurement(num_measurements, INFINITY);

            for (auto& vehicle : vehicles)
            {
                vehicle.move(dt);

                for (int i = 0; i < vehicle.width; i++)
                {
                    int index = static_cast<int>(vehicle.pos.x) + i;
                    measurement[index] = vehicle.pos.y;
                }
            }

            measurements.push_back(measurement);
        }

        return measurements;
    }

    int num_measurements;
    std::vector<Vehicle> vehicles;
};

#endif  // SIMULATOR_H
