// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/init.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dogm
{

__global__ void setupRandomStatesKernel(curandState* __restrict__ states, unsigned long long seed, int count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
    {
        curand_init(seed, i, 0, &states[i]);
    }
}

__global__ void initParticlesKernel(ParticlesSoA particle_array, curandState* __restrict__ global_state, float velocity,
                                    int grid_size, int particle_count)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState local_state = global_state[thread_id];

    for (int i = thread_id; i < particle_count; i += stride)
    {
        float x = curand_uniform(&local_state, 0.0f, grid_size - 1);
        float y = curand_uniform(&local_state, 0.0f, grid_size - 1);
        float vel_x = curand_uniform(&local_state, -velocity, velocity);
        float vel_y = curand_uniform(&local_state, -velocity, velocity);

        particle_array.weight[i] = 1.0f / particle_count;
        particle_array.state[i] = glm::vec4(x, y, vel_x, vel_y);

        // printf("w: %f, x: %f, y: %f, vx: %f, vy: %f\n", particle_array[i].weight, particle_array[i].state[0],
        // particle_array[i].state[1], 	particle_array[i].state[2], particle_array[i].state[3]);
    }

    global_state[thread_id] = local_state;
}

__global__ void initBirthParticlesKernel(ParticlesSoA birth_particle_array, curandState* __restrict__ global_state,
                                         float velocity, int grid_size, int particle_count)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // curandState local_state = global_state[thread_id];

    for (int i = thread_id; i < particle_count; i += stride)
    {
        // float x = curand_uniform(&local_state, 0.0f, grid_size - 1);
        // float y = curand_uniform(&local_state, 0.0f, grid_size - 1);
        // float vel_x = curand_normal(&local_state, 0.0f, velocity);
        // float vel_y = curand_normal(&local_state, 0.0f, velocity);

        birth_particle_array.weight[i] = 0.0f;
        birth_particle_array.associated[i] = false;
        birth_particle_array.state[i] = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // global_state[thread_id] = local_state;
}

__global__ void initGridCellsKernel(GridCell* __restrict__ grid_cell_array,
                                    MeasurementCell* __restrict__ meas_cell_array, int grid_size, int cell_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
    {
        grid_cell_array[i].free_mass = 0.0f;
        grid_cell_array[i].occ_mass = 0.0f;
        grid_cell_array[i].start_idx = -1;
        grid_cell_array[i].end_idx = -1;

        meas_cell_array[i].occ_mass = 0.0f;
        meas_cell_array[i].free_mass = 0.0f;
        meas_cell_array[i].likelihood = 1.0f;
        meas_cell_array[i].p_A = 1.0f;
    }
}

__global__ void reinitGridParticleIndices(GridCell* __restrict__ grid_cell_array, int cell_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
    {
        grid_cell_array[i].start_idx = -1;
        grid_cell_array[i].end_idx = -1;
    }
}

} /* namespace dogm */
