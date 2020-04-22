// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <curand_kernel.h>
#include <device_launch_parameters.h>

namespace dogm
{

struct GridCell;
struct MeasurementCell;
struct Particle;

__global__ void setupRandomStatesKernel(curandState* __restrict__ states, unsigned long long seed, int count);

__global__ void initParticlesKernel(ParticlesSoA particle_array, curandState* __restrict__ global_state, float velocity,
                                    int grid_size, int particle_count);

__global__ void initBirthParticlesKernel(ParticlesSoA birth_particle_array, curandState* __restrict__ global_state,
                                         float velocity, int grid_size, int particle_count);

__global__ void initGridCellsKernel(GridCell* __restrict__ grid_cell_array,
                                    MeasurementCell* __restrict__ meas_cell_array, int grid_size, int cell_count);

__global__ void reinitGridParticleIndices(GridCell* __restrict__ grid_cell_array, int cell_count);

} /* namespace dogm */
