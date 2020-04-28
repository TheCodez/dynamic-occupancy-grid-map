// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/predict.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dogm
{

__global__ void predictKernel(ParticlesSoA particle_array, curandState* __restrict__ global_state, float velocity,
                              int grid_size, float p_S, const glm::mat4x4 transition_matrix,
                              float process_noise_position, float process_noise_velocity, int particle_count)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState local_state = global_state[thread_id];

    for (int i = thread_id; i < particle_count; i += stride)
    {
        float noise_pos_x = curand_normal(&local_state, 0.0f, process_noise_position);
        float noise_pos_y = curand_normal(&local_state, 0.0f, process_noise_position);
        float noise_vel_x = curand_normal(&local_state, 0.0f, process_noise_velocity);
        float noise_vel_y = curand_normal(&local_state, 0.0f, process_noise_velocity);
        glm::vec4 process_noise(noise_pos_x, noise_pos_y, noise_vel_x, noise_vel_y);

        particle_array.state[i] = transition_matrix * particle_array.state[i] + process_noise;
        particle_array.weight[i] = p_S * particle_array.weight[i];

        glm::vec4 state = particle_array.state[i];
        float x = state[0];
        float y = state[1];

        // Particle out of grid so decrease its chance of being resampled
        if ((x > grid_size - 1 || x < 0) || (y > grid_size - 1 || y < 0))
        {
            particle_array.weight[i] = 0.0f;
        }

        int pos_x = clamp(static_cast<int>(x), 0, grid_size - 1);
        int pos_y = clamp(static_cast<int>(y), 0, grid_size - 1);
        particle_array.grid_cell_idx[i] = pos_x + grid_size * pos_y;

        // printf("X: %d, Y: %d, Cell index: %d\n", pos_x, pos_y, (pos_x + grid_size * pos_y));
    }

    global_state[thread_id] = local_state;
}

} /* namespace dogm */
