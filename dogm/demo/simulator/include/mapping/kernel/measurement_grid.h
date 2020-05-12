// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <device_launch_parameters.h>

namespace dogm
{
struct MeasurementCell;
}

__global__ void createPolarGridTextureKernel(cudaSurfaceObject_t polar, const float* __restrict__ measurements,
                                             int width, int height, float resolution);

__global__ void fusePolarGridTextureKernel(cudaSurfaceObject_t polar, const float* __restrict__ measurements, int width,
                                           int height, float resolution);

__global__ void cartesianGridToMeasurementGridKernel(dogm::MeasurementCell* __restrict__ meas_grid,
                                                     cudaSurfaceObject_t cart, int grid_size);

__global__ void gridArrayToMeasurementGridKernel(dogm::MeasurementCell* __restrict__ meas_grid,
                                                 const float2* __restrict__ grid, int grid_size);
