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
#include "dogm/dogm.h"
#include "dogm/dogm_types.h"
#include "image_creation.h"
#include "precision_evaluator.h"
#include "simulator.h"

#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

int main(int argc, const char** argv)
{
    std::vector<std::vector<float2>> mg_meas;

    for (int i = 0; i < 10; i++)
    {
        std::vector<float2> grid = load_measurement_from_image(cv::format("meas_grids/meas_grid%d.png", i));
        mg_meas.push_back(grid);
    }

// clang-format off
#if 0
	dogm::GridParams params;
	params.size = 128;
	params.resolution = 1.0f;
	params.particle_count = 2 * static_cast<int>(10e5);
	params.new_born_particle_count = 2 * static_cast<int>(10e4);
	params.persistence_prob = 0.99f;
	params.process_noise_position = 0.06f;
	params.process_noise_velocity = 2.4f;
	params.birth_prob = 0.02f;
	params.velocity_persistent = 12.0f;
	params.velocity_birth = 12.0f;

	dogm::LaserSensorParams laser_params;
	laser_params.fov = 120.0f;
	laser_params.max_range = 50.0f;

	dogm::DOGM grid_map(params, laser_params);

	float delta_time = 0.1f;
	for (int i = 0; i < mg_meas.size(); i++)
	{
		// Update measurement grid
		grid_map.updateMeasurementGridFromArray(mg_meas[i]);
#else
	dogm::GridParams params;
	params.size = 50.0f;
	params.resolution = 0.2f;
	params.particle_count = 3 * static_cast<int>(10e5);
	params.new_born_particle_count = 3 * static_cast<int>(10e4);
	params.persistence_prob = 0.99f;
	params.process_noise_position = 0.02f;
	params.process_noise_velocity = 0.8f;
	params.birth_prob = 0.02f;
	params.velocity_persistent = 30.0f;
	params.velocity_birth = 30.0f;

	dogm::LaserSensorParams laser_params;
	laser_params.fov = 120.0f;
	laser_params.max_range = 50.0f;

	// Just to init cuda
	cudaDeviceSynchronize();

	auto begin = std::chrono::high_resolution_clock::now();

	dogm::DOGM grid_map(params, laser_params);

	auto end = std::chrono::high_resolution_clock::now();
	auto dur = end - begin;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	std::cout << "### DOGM initialization took: " << ms << " ms" << " ###" << std::endl << std::endl;

	Simulator simulator(100);
	simulator.addVehicle(Vehicle(6, glm::vec2(20, 10), glm::vec2(0, 0)));
	simulator.addVehicle(Vehicle(5, glm::vec2(46, 20), glm::vec2(-5, 20)));
	simulator.addVehicle(Vehicle(4, glm::vec2(80, 30), glm::vec2(0, -10)));

//	simulator.addVehicle(Vehicle(6, glm::vec2(40, 30), glm::vec2(20, 5)));
//	simulator.addVehicle(Vehicle(5, glm::vec2(80, 24), glm::vec2(-15, -5)));

	float delta_time = 0.1f;
	SimulationData sim_data = simulator.update(10, delta_time);
	PrecisionEvaluator precision_evaluator{sim_data, params.resolution};

	for (int i = 0; i < sim_data.size(); i++)
	{
		grid_map.updateMeasurementGrid(sim_data[i].measurements.data(), sim_data[i].measurements.size());
#endif
		begin = std::chrono::high_resolution_clock::now();

		// Run Particle filter
		grid_map.updateParticleFilter(delta_time);

		end = std::chrono::high_resolution_clock::now();
		dur = end - begin;
		ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::cout << "### Iteration took: " << ms << " ms" << " ###" << std::endl;
		std::cout << "######  Saving result  #######" << std::endl;
		std::cout << "##############################" << std::endl;

		const auto cells_with_velocity{computeCellsWithVelocity(grid_map, 0.7f, 4.0f)};
		precision_evaluator.evaluateAndStoreStep(i, cells_with_velocity, true);

		cv::Mat meas_grid_img = compute_measurement_grid_image(grid_map);
		cv::imwrite(cv::format("meas_grid_iter-%d.png", i + 1), meas_grid_img);

		cv::Mat raw_meas_grid_img = compute_raw_measurement_grid_image(grid_map);
		cv::imwrite(cv::format("raw_grid_iter-%d.png", i + 1), raw_meas_grid_img);

		cv::Mat grid_img = compute_dogm_image(grid_map, cells_with_velocity);
		cv::imwrite(cv::format("dogm_iter-%d.png", i + 1), grid_img);

		cv::Mat particle_img = compute_particles_image(grid_map);
		cv::imwrite(cv::format("particles_iter-%d.png", i + 1), particle_img);
	}

	precision_evaluator.printSummary();

#if	0
	cv::Mat particle_img = compute_particles_image(grid_map);
	cv::Mat grid_img = compute_dogm_image(grid_map, 0.7f, 4.0f);

	cv::namedWindow("particles", cv::WINDOW_NORMAL);
	cv::imshow("particles", particle_img);

	cv::namedWindow("dynamic_grid", cv::WINDOW_NORMAL);
	cv::imshow("dynamic_grid", grid_img);

	cv::waitKey(0);
#endif

	return 0;
// clang-format on
}