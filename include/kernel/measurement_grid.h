#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void createPolarGridTextureKernel(cudaSurfaceObject_t polar, float* measurements, int width, int height, float resolution);

__global__ void fusePolarGridsKernel(cudaSurfaceObject_t result, cudaSurfaceObject_t polar, cudaSurfaceObject_t prev_polar,
	int width, int height);

__global__ void cartesianGridToMeasurementGridKernel(MeasurementCell* meas_grid, cudaSurfaceObject_t cart, int width, int height);
