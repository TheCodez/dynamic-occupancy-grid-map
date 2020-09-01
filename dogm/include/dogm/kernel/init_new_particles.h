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

void normalize_particle_orders(float* particle_orders_array_accum, int particle_orders_count, int v_B);

__global__ void copyMassesKernel(const MeasurementCell* __restrict__ meas_cell_array, float* __restrict__ masses,
                                 int cell_count);

__global__ void initParticlesKernel1(GridCell* __restrict__ grid_cell_array,
                                     const MeasurementCell* __restrict__ meas_cell_array, ParticlesSoA particle_array,
                                     const float* __restrict__ particle_orders_array_accum, int cell_count);

__global__ void initParticlesKernel2(ParticlesSoA particle_array, const GridCell* __restrict__ grid_cell_array,
                                     curandState* __restrict__ global_state, float velocity, int grid_size,
                                     float new_weight, int particle_count);

__global__ void initNewParticlesKernel1(GridCell* __restrict__ grid_cell_array,
                                        const MeasurementCell* __restrict__ meas_cell_array,
                                        const float* __restrict__ weight_array,
                                        const float* __restrict__ born_masses_array, ParticlesSoA birth_particle_array,
                                        const float* __restrict__ particle_orders_array_accum, int cell_count);

__global__ void initNewParticlesKernel2(ParticlesSoA birth_particle_array, const GridCell* __restrict__ grid_cell_array,
                                        curandState* __restrict__ global_state, float stddev_velocity,
                                        float max_velocity, int grid_size, int particle_count);

__global__ void copyBirthWeightKernel(const ParticlesSoA birth_particle_array, float* __restrict__ birth_weight_array,
                                      int particle_count);

} /* namespace dogm */
