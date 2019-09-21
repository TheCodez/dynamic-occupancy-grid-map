#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void createPolarGridMapKernel(float2* polar, float* measurements, int width, int height, float resolution,
	float min_range, float max_range);

__global__ void polarToCartesianGridMapKernel(MeasurementCell* cartesian, float2* polar, int width, int height,
	int polar_w, int polar_h);
