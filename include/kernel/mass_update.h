#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void gridCellPredictionUpdateKernel(GridCell* grid_cell_array, float* weight_array_accum, MeasurementCell* meas_cell_array,
	float* born_masses_array, float pb);
