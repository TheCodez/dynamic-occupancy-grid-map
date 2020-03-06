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
#pragma once

#include <memory>

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <cuda_runtime.h>

class Renderer;

struct GridCell
{
	int start_idx;
	int end_idx;
	double new_born_occ_mass;
	double pers_occ_mass;
	double free_mass;
	double occ_mass;
	double mu_A;
	double mu_UA;

	double w_A;
	double w_UA;

	double mean_x_vel;
	double mean_y_vel;
	double var_x_vel;
	double var_y_vel;
	double covar_xy_vel;

	int2 pos;
};

struct MeasurementCell
{
	double free_mass;
	double occ_mass;
	double likelihood;
	double p_A;
};

struct Particle
{
	int grid_cell_idx;
	double weight;
	bool associated;
	glm::vec4 state;
};

struct GridParams
{
	double size;
	double resolution;
	int particle_count;
	int new_born_particle_count;
	double persistence_prob;
	double process_noise_position;
	double process_noise_velocity;
	double birth_prob;
};

struct LaserSensorParams
{
	double max_range;
	double fov;
};

class DOGM
{
public:
	DOGM(const GridParams& params, const LaserSensorParams& laser_params);
	~DOGM();

	void updateMeasurementGrid(double* measurements, int num_measurements);
	void updateParticleFilter(double dt);

	int getGridSize() const { return grid_size; }

private:
	void initialize();

public:
	void particlePrediction(double dt);
	void particleAssignment();
	void gridCellOccupancyUpdate();
	void updatePersistentParticles();
	void initializeNewParticles();
	void statisticalMoments();
	void resampling();

public:

	GridParams params;
	LaserSensorParams laser_params;

	std::unique_ptr<Renderer> renderer;

	GridCell* grid_cell_array;
	Particle* particle_array;
	Particle* particle_array_next;
	Particle* birth_particle_array;

	MeasurementCell* polar_meas_cell_array;
	MeasurementCell* meas_cell_array;

	double* weight_array;
	double* birth_weight_array;
	double* born_masses_array;

	int grid_size;

	int grid_cell_count;
	int particle_count;
	int new_born_particle_count;

	static int BLOCK_SIZE;
};
