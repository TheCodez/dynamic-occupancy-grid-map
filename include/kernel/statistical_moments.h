#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void statisticalMomentsKernel1(Particle* particle_array, float* weight_array, float* vel_x_array, float* vel_y_array,
	float* vel_x_squared_array, float* vel_y_squared_array, float* vel_xy_array, int particle_count);

__global__ void statisticalMomentsKernel2(GridCell* grid_cell_array, float* vel_x_array_accum, float* vel_y_array_accum,
	float* vel_x_squared_array_accum, float* vel_y_squared_array_accum, float* vel_xy_array_accum, int cell_count);

