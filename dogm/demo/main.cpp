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
    dogm::GridParams params;
    params.size = 50.0f;
    params.resolution = 0.2f;
    params.particle_count = 3 * static_cast<int>(10e5);
    params.new_born_particle_count = 3 * static_cast<int>(10e4);
    params.persistence_prob = 0.99f;
    params.process_noise_position = 0.1f;
    params.process_noise_velocity = 1.0f;
    params.birth_prob = 0.02f;
    params.velocity_persistent = 30.0f;
    params.velocity_birth = 30.0f;

    dogm::LaserSensorParams laser_params;
    laser_params.fov = 120.0f;
    laser_params.max_range = 50.0f;
    laser_params.resolution = 0.2f;

    // Just to init cuda
    cudaDeviceSynchronize();

    Timer initialization_timer{"DOGM initialization"};
    dogm::DOGM grid_map(params, laser_params);
    initialization_timer.toc(true);

    Simulator simulator(100, laser_params.fov);
#if 1
    simulator.addVehicle(Vehicle(3, glm::vec2(30, 20), glm::vec2(0, 6)));
    simulator.addVehicle(Vehicle(4, glm::vec2(30, 30), glm::vec2(15, 0)));
    simulator.addVehicle(Vehicle(4, glm::vec2(60, 30), glm::vec2(0, -8)));
    simulator.addVehicle(Vehicle(2, glm::vec2(68, 15), glm::vec2(0, 0)));
#else
    simulator.addVehicle(Vehicle(4, glm::vec2(30, 30), glm::vec2(10, -8)));
    simulator.addVehicle(Vehicle(6, glm::vec2(60, 30), glm::vec2(-8, 6)));
    simulator.addVehicle(Vehicle(3, glm::vec2(68, 15), glm::vec2(-12, 0)));
#endif

    float delta_time = 0.1f;
    SimulationData sim_data = simulator.update(14, delta_time);
    PrecisionEvaluator precision_evaluator{sim_data, params.resolution};
    Timer cycle_timer{"DOGM cycle"};

    for (int i = 0; i < sim_data.size(); i++)
    {
        grid_map.updateMeasurementGrid(sim_data[i].measurements);

        cycle_timer.tic();
        // Run Particle filter
        grid_map.updateParticleFilter(delta_time);

        cycle_timer.toc(true);
        std::cout << "######  Saving result  #######" << std::endl;
        std::cout << "##############################" << std::endl << std::endl;

        const auto cells_with_velocity = computeCellsWithVelocity(grid_map, 0.7f, 4.0f);
        precision_evaluator.evaluateAndStoreStep(i, cells_with_velocity);

        // cv::Mat meas_grid_img = compute_measurement_grid_image(grid_map);
        // cv::imwrite(cv::format("meas_grid_iter-%d.png", i + 1), meas_grid_img);

        cv::Mat raw_meas_grid_img = compute_raw_measurement_grid_image(grid_map);
        cv::imwrite(cv::format("raw_grid_iter-%d.png", i + 1), raw_meas_grid_img);

        cv::Mat dogm_img = compute_dogm_image(grid_map, cells_with_velocity);
        cv::imwrite(cv::format("dogm_iter-%d.png", i + 1), dogm_img);

        cv::Mat particle_img = compute_particles_image(grid_map);
        cv::imwrite(cv::format("particles_iter-%d.png", i + 1), particle_img);

        cv::imshow("dogm", dogm_img);
        cv::waitKey(1);
    }

    cycle_timer.printStatsMs();
    precision_evaluator.printSummary();

    return 0;
}