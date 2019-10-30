#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void gridCellPredictionUpdateKernel(GridCell* grid_cell_array, Particle* particle_array, float* weight_array,
	float* weight_array_accum, MeasurementCell* meas_cell_array, float* born_masses_array, float p_B, float p_S, int cell_count);
