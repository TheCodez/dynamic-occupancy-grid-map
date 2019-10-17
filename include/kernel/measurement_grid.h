#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void createPolarGridTextureKernel(cudaSurfaceObject_t polar, float* measurements, int width, int height, float resolution);

__global__ void fusePolarGridTextureKernel(cudaSurfaceObject_t polar, float* measurements, int width, int height, float resolution);

__global__ void cartesianGridToMeasurementGridKernel(MeasurementCell* meas_grid, cudaSurfaceObject_t cart, int grid_size);