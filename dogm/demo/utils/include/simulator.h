// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <glm/glm.hpp>
#include <vector>

class CoordinateSystemMapper
{
public:
    CoordinateSystemMapper(const float grid_size, const float grid_resolution)
        : grid_size{grid_size}, grid_resolution{grid_resolution}
    {
    }

    float mapAbsoluteGridPositionToRelativePosition(const float position_in_meters)
    {
        return position_in_meters / grid_size;
    }

private:
    float grid_size;
    float grid_resolution;
};

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
    Simulator(int _num_horizontal_scan_points, const float _field_of_view, CoordinateSystemMapper& mapper)
        : num_horizontal_scan_points(_num_horizontal_scan_points), field_of_view(_field_of_view), mapper{mapper}
    {
    }

    void addVehicle(const Vehicle& vehicle) { vehicles.push_back(vehicle); }

    SimulationData update(int steps, float dt);
    void addVehicleDetectionsToMeasurement(const Vehicle& vehicle, std::vector<float>& measurement) const;

    int num_horizontal_scan_points;
    float field_of_view;
    std::vector<Vehicle> vehicles;
    CoordinateSystemMapper& mapper;
};

#endif  // SIMULATOR_H
