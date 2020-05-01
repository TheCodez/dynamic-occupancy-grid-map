// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/dogm.h"
#include "dogm/dogm_types.h"
#include "image_creation.h"
#include "precision_evaluator.h"
#include "simulator.h"
#include "timer.h"

#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

int main(int argc, const char** argv)
{
    dogm::GridParams grid_params;
    grid_params.size = 50.0f;
    grid_params.resolution = 0.2f;
    grid_params.particle_count = 3 * static_cast<int>(10e5);
    grid_params.new_born_particle_count = 3 * static_cast<int>(10e4);
    grid_params.persistence_prob = 0.99f;
    grid_params.process_noise_position = 0.1f;
    grid_params.process_noise_velocity = 1.0f;
    grid_params.birth_prob = 0.02f;
    grid_params.velocity_persistent = 30.0f;
    grid_params.velocity_birth = 30.0f;

    dogm::LaserSensorParams laser_params;
    laser_params.fov = 120.0f;
    laser_params.max_range = 50.0f;
    laser_params.resolution = grid_params.resolution;  // TODO make independent of grid_params.resolution
    const int sensor_horizontal_scan_points = 100;

    // Simulator parameters
    const int num_simulation_steps = 14;
    const float simulation_step_period = 0.1f;

    // Evaluator parameters
    const float minimum_occupancy_threshold = 0.7f;
    const float minimum_velocity_threshold = 4.0f;

    // Just to init cuda
    cudaDeviceSynchronize();

    Timer initialization_timer{"DOGM initialization"};
    dogm::DOGM grid_map(grid_params, laser_params);
    initialization_timer.toc(true);

    Simulator simulator(sensor_horizontal_scan_points, laser_params.fov, grid_params.size);
#if 1
    simulator.addVehicle(Vehicle(3.5, glm::vec2(10, 30), glm::vec2(15, 0)));
    simulator.addVehicle(Vehicle(3.0, glm::vec2(10, 20), glm::vec2(0, 5)));
    simulator.addVehicle(Vehicle(4.0, glm::vec2(35, 35), glm::vec2(0, -10)));
    simulator.addVehicle(Vehicle(1.8, glm::vec2(45, 15), glm::vec2(0, 0)));
#else
    simulator.addVehicle(Vehicle(4.0, glm::vec2(10, 25), glm::vec2(10, -8)));
    simulator.addVehicle(Vehicle(6.0, glm::vec2(40, 30), glm::vec2(-8, 6)));
    simulator.addVehicle(Vehicle(3.0, glm::vec2(48, 15), glm::vec2(-12, 0)));
#endif

    SimulationData sim_data = simulator.update(num_simulation_steps, simulation_step_period);
    PrecisionEvaluator precision_evaluator{sim_data, grid_params.resolution, grid_params.size};
    Timer cycle_timer{"DOGM cycle"};

    for (int step = 0; step < num_simulation_steps; ++step)
    {
        grid_map.updateMeasurementGrid(sim_data[step].measurements);

        cycle_timer.tic();
        grid_map.updateParticleFilter(simulation_step_period);
        cycle_timer.toc(true);

        const auto cells_with_velocity =
            computeCellsWithVelocity(grid_map, minimum_occupancy_threshold, minimum_velocity_threshold);
        precision_evaluator.evaluateAndStoreStep(step, cells_with_velocity);

        computeAndSaveResultImages(grid_map, cells_with_velocity, step);
    }

    cycle_timer.printStatsMs();
    precision_evaluator.printSummary();

    return 0;
}
