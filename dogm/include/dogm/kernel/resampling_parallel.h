// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <curand_kernel.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr auto kTRI {256};
constexpr auto kWarpSize {32};
constexpr bool systematic {true};

namespace dogm
{

__global__ void resampleIndexKernel(const ParticlesSoA particle_array, ParticlesSoA particle_array_next,
    const ParticlesSoA birth_particle_array, const int* __restrict__ idx_array_up,
    const int* __restrict__ idx_array_down, float new_weight, int particle_count);

// Systematic / Stratified max optimized

__global__ void resampleSystematicIndexUp(int const num_particles,
    unsigned long long int const seed, int* __restrict__ resampling_index_up, float* __restrict__ prefix_sum);

__device__ void ResamplingUpPerWarp(cg::thread_block_tile<kWarpSize> const &tile_32,
    unsigned int const &tid, int const &num_particles, float const &distro,
    float *shared, float *__restrict__ prefix_sum, int *__restrict__ resampling_index_up);

__global__ void resampleSystematicIndexDown(int const num_particles,
    unsigned long long int const seed, int *__restrict__ resampling_index_down, float *__restrict__ prefix_sum);

__device__ void ResamplingDownPerWarp(cg::thread_block_tile<kWarpSize> const &tile_32,
    unsigned int const &tid, int const &num_particles, float const &distro,
    float *shared, float *__restrict__ prefix_sum, int *__restrict__ resampling_index_down );

} /* namespace dogm */
