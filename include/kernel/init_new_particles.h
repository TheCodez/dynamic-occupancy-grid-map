#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__device__ __host__ void normalize_particle_orders(float* particle_orders_array_accum, int particle_orders_count, int v_B);

__global__ void initNewParticlesKernel1(Particle* particle_array, GridCell* grid_cell_array, MeasurementCell* meas_cell_array,
	float* weight_array, float* born_masses_array, Particle* birth_particle_array, float* particle_orders_array_accum, int cell_count);

__global__ void initNewParticlesKernel2(Particle* birth_particle_array, GridCell* grid_cell_array, float* birth_weight_array, int width,
	int particle_count);
