// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "clock.h"
#include "dogm/dogm.h"
#include "dogm/dogm_types.h"
#include "image_creation.h"
#include "mapping/laser_to_meas_grid.h"
#include "metrics.h"
#include "precision_evaluator.h"
#include "simulator.h"
#include "timer.h"

#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>

#include <memory>
#include <string>

int main(int argc, const char** argv)
{
    dogm::DOGM::Params grid_params;
    grid_params.size = 50.0f;
    grid_params.resolution = 0.2f;
    grid_params.particle_count = 3 * static_cast<int>(10e5);
    grid_params.new_born_particle_count = 3 * static_cast<int>(10e4);
    grid_params.persistence_prob = 0.99f;
    grid_params.stddev_process_noise_position = 0.1f;
    grid_params.stddev_process_noise_velocity = 1.0f;
    grid_params.birth_prob = 0.02f;
    grid_params.stddev_velocity = 30.0f;
    grid_params.init_max_velocity = 30.0f;

    LaserMeasurementGrid::Params laser_params;
    laser_params.fov = 120.0f;
    laser_params.max_range = 50.0f;
    laser_params.resolution = grid_params.resolution;  // TODO make independent of grid_params.resolution
    LaserMeasurementGrid grid_generator(laser_params, grid_params.size, grid_params.resolution);

    const int sensor_horizontal_scan_points = 100;

    // Simulator parameters
    const int num_simulation_steps = 14;
    const float simulation_step_period = 0.1f;
    const glm::vec2 ego_velocity{0.0f, 4.0f};

    // Evaluator parameters
    const float minimum_occupancy_threshold = 0.7f;
    const float minimum_velocity_threshold = 4.0f;

    // Just to init cuda
    cudaDeviceSynchronize();

    Timer initialization_timer{"DOGM initialization", std::make_unique<Clock>()};
    dogm::DOGM grid_map(grid_params);
    initialization_timer.toc(true);

    Simulator simulator(sensor_horizontal_scan_points, laser_params.fov, grid_params.size, ego_velocity);
    if (argc > 1 && argv[1] == std::string("-a"))
    {
        simulator.addVehicle(Vehicle(4.0, glm::vec2(10, 25), glm::vec2(10, -8)));
        simulator.addVehicle(Vehicle(6.0, glm::vec2(40, 30), glm::vec2(-8, 6)));
        simulator.addVehicle(Vehicle(3.0, glm::vec2(48, 15), glm::vec2(-12, 0)));
    }
    else
    {
        simulator.addVehicle(Vehicle(3.5, glm::vec2(10, 30), glm::vec2(15, 0)));
        simulator.addVehicle(Vehicle(3.0, glm::vec2(10, 20), glm::vec2(0, 5)));
        simulator.addVehicle(Vehicle(4.0, glm::vec2(35, 35), glm::vec2(0, -10)));
        simulator.addVehicle(Vehicle(1.8, glm::vec2(45, 15), glm::vec2(0, 0)));
    }

    SimulationData sim_data = simulator.update(num_simulation_steps, simulation_step_period);
    PrecisionEvaluator precision_evaluator{sim_data, grid_params.resolution, grid_params.size};
    precision_evaluator.registerMetric("Mean absolute error (MAE)", std::make_unique<MAE>());
    precision_evaluator.registerMetric("Root mean squared error (RMSE)", std::make_unique<RMSE>());

    Timer cycle_timer{"DOGM cycle", std::make_unique<Clock>()};

    for (int step = 0; step < num_simulation_steps; ++step)
    {
        grid_map.updatePose(sim_data[step].ego_pose.x, sim_data[step].ego_pose.y);

        dogm::MeasurementCell* meas_grid = grid_generator.generateGrid(sim_data[step].measurements);
        grid_map.addMeasurementGrid(meas_grid, true);

        const auto update_grid_caller = [&grid_map](const float dt) { grid_map.updateGrid(dt); };
        cycle_timer.timeFunctionCall(true, update_grid_caller, simulation_step_period);

        const auto cells_with_velocity =
            computeCellsWithVelocity(grid_map, minimum_occupancy_threshold, minimum_velocity_threshold);
        precision_evaluator.evaluateAndStoreStep(step, cells_with_velocity);

        computeAndSaveResultImages(grid_map, cells_with_velocity, step);
    }

    cycle_timer.printStatsMs();
    precision_evaluator.printSummary();

    return 0;
}
