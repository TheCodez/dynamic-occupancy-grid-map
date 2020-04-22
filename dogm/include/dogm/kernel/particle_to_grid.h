// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <device_launch_parameters.h>

namespace dogm
{

struct GridCell;
struct Particle;

__global__ void particleToGridKernel(const ParticlesSoA particle_array, GridCell* __restrict__ grid_cell_array,
                                     float* __restrict__ weight_array, int particle_count);

} /* namespace dogm */
