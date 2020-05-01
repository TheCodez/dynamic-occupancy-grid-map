// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <glm/glm.hpp>
#include <vector>

struct Vehicle
{
    // Construct a vehicle from the perspective of a sensor. Vehicles do not have bounding boxes, but are represented
    // only by one facing side. This facing side has a width [m] and a 2D position [m]. Position is in the center of the
    // facing side. 2D Velocity is given in [m/s].
    Vehicle(const int width, const glm::vec2& position, const glm::vec2& velocity)
        : width(width), pos(position), vel(velocity)
    {
    }

    void move(float dt) { pos += vel * dt; }

    /// brief Gets points on facing side
    /// param resolution [m] defines in which resolution points shall be sampled
    std::vector<glm::vec2> getPointsOnFacingSide(const float resolution) const;

    glm::vec2 pos;
    glm::vec2 vel;

private:
    float width;
};

struct SimulationStep
{
    std::vector<Vehicle> vehicles;
    std::vector<float> measurements;
};

using SimulationData = std::vector<SimulationStep>;

class Simulator
{
public:
    Simulator(int _num_horizontal_scan_points, const float _field_of_view, const float grid_size);

    void addVehicle(const Vehicle& vehicle) { vehicles.push_back(vehicle); }
    SimulationData update(int steps, float dt);
    void addVehicleDetectionsToMeasurement(const Vehicle& vehicle, std::vector<float>& measurement) const;

    int num_horizontal_scan_points;
    float field_of_view;
    std::vector<Vehicle> vehicles;

private:
    float grid_size;
    int computeMeasurementVectorIndexFromAngle(const float angle) const;
    float factor_angle_to_grid;
    float angle_offset;
    glm::vec2 sensor_position;
};

#endif  // SIMULATOR_H
