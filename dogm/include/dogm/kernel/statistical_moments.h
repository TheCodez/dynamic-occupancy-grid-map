// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <device_launch_parameters.h>

namespace dogm
{

struct GridCell;
struct Particle;

__global__ void statisticalMomentsKernel1(const ParticlesSoA particle_array, const float* __restrict__ weight_array,
                                          float* __restrict__ vel_x_array, float* __restrict__ vel_y_array,
                                          float* __restrict__ vel_x_squared_array,
                                          float* __restrict__ vel_y_squared_array, float* __restrict__ vel_xy_array,
                                          int particle_count);

__global__ void statisticalMomentsKernel2(GridCell* __restrict__ grid_cell_array,
                                          const float* __restrict__ vel_x_array_accum,
                                          const float* __restrict__ vel_y_array_accum,
                                          const float* __restrict__ vel_x_squared_array_accum,
                                          const float* __restrict__ vel_y_squared_array_accum,
                                          const float* __restrict__ vel_xy_array_accum, int cell_count);

} /* namespace dogm */
