/*
MIT License

Copyright (c) 2019 Michael Kösel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/predict.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dogm
{

__global__ void predictKernel(Particle* __restrict__ particle_array, curandState* __restrict__ global_state,
                              float velocity, int grid_size, float p_S, const glm::mat4x4 transition_matrix,
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

        particle_array[i].state = transition_matrix * particle_array[i].state + process_noise;
        particle_array[i].weight = p_S * particle_array[i].weight;

        float x = particle_array[i].state[0];
        float y = particle_array[i].state[1];

        if ((x > grid_size - 1 || x < 0) || (y > grid_size - 1 || y < 0))
        {
            x = curand_uniform(&local_state, 0.0f, grid_size - 1);
            y = curand_uniform(&local_state, 0.0f, grid_size - 1);
            float vel_x = curand_uniform(&local_state, -velocity, velocity);
            float vel_y = curand_uniform(&local_state, -velocity, velocity);

            particle_array[i].state = glm::vec4(x, y, vel_x, vel_y);
        }

        int pos_x = clamp(static_cast<int>(x), 0, grid_size - 1);
        int pos_y = clamp(static_cast<int>(y), 0, grid_size - 1);
        particle_array[i].grid_cell_idx = pos_x + grid_size * pos_y;

        // printf("X: %d, Y: %d, Cell index: %d\n", pos_x, pos_y, (pos_x + grid_size * pos_y));
    }

    global_state[thread_id] = local_state;
}

} /* namespace dogm */
