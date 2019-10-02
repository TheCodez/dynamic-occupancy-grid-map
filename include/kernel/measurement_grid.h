#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void createPolarGridMapKernel(cudaSurfaceObject_t polar, float* measurements, int width, int height, float resolution,
	float max_range);

__global__ void cartesianGridToMeasurementGridKernel(MeasurementCell* meas_grid, cudaSurfaceObject_t cart, int width, int height);
