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

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <cuda_runtime.h>

class Renderer;

struct GridCell
{
	int start_idx;
	int end_idx;
	float new_born_occ_mass;
	float pers_occ_mass;
	float free_mass;
	float occ_mass;
	float mu_A;
	float mu_UA;

	float w_A;
	float w_UA;

	float mean_x_vel;
	float mean_y_vel;
	float var_x_vel;
	float var_y_vel;
	float covar_xy_vel;

	int2 pos;
};

struct MeasurementCell
{
	float free_mass;
	float occ_mass;
	float likelihood;
	float p_A;
};

struct Particle
{
	int grid_cell_idx;
	float weight;
	bool associated;
	glm::vec4 state;
};

struct GridParams
{
	float size;
	float resolution;
	int particle_count;
	int new_born_particle_count;
	float p_S;
	float process_noise_position;
	float process_noise_velocity;
	float p_B;
};

struct LaserSensorParams
{
	float max_range;
	float fov;
};

class OccupancyGridMap
{
public:
	OccupancyGridMap(const GridParams& params, const LaserSensorParams& laser_params);
	~OccupancyGridMap();

	void updateMeasurementGrid(float* measurements, int num_measurements);
	void updateDynamicGrid(float dt);

	int getGridSize() const { return grid_size; }

private:
	void initialize();

public:
	void particlePrediction(float dt);
	void particleAssignment();
	void gridCellOccupancyUpdate();
	void updatePersistentParticles();
	void initializeNewParticles();
	void statisticalMoments();
	void resampling();

public:

	GridParams params;
	LaserSensorParams laser_params;

	Renderer* renderer;

	GridCell* grid_cell_array;
	Particle* particle_array;
	Particle* particle_array_next;
	Particle* birth_particle_array;

	MeasurementCell* meas_cell_array;

	float* weight_array;
	float* birth_weight_array;
	float* born_masses_array;

	int grid_size;

	int grid_cell_count;
	int particle_count;
	int new_born_particle_count;

	static int BLOCK_SIZE;
};
