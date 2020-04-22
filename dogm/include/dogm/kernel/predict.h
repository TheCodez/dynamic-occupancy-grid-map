// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

namespace dogm
{

struct Particle;

__global__ void predictKernel(ParticlesSoA particle_array, curandState* __restrict__ global_state, float velocity,
                              int grid_size, float p_S, const glm::mat4x4 transition_matrix,
                              float process_noise_position, float process_noise_velocity, int particle_count);

} /* namespace dogm */
