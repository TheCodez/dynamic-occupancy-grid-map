// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/init_new_particles.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace dogm
{

__device__ void set_cell_idx_A(const ParticlesSoA& birth_particle_array, int i, int grid_cell_idx)
{
    birth_particle_array.grid_cell_idx[i] = grid_cell_idx;
    birth_particle_array.associated[i] = true;
}

__device__ void set_cell_idx_UA(const ParticlesSoA& birth_particle_array, int i, int grid_cell_idx)
{
    birth_particle_array.grid_cell_idx[i] = grid_cell_idx;
    birth_particle_array.associated[i] = false;
}

__device__ int calc_start_idx(const float* __restrict__ particle_orders_array_accum, int index)
{
    if (index == 0)
    {
        return 0;
    }

    return static_cast<int>(particle_orders_array_accum[index - 1]);
}

__device__ int calc_end_idx(const float* __restrict__ particle_orders_array_accum, int index)
{
    return static_cast<int>(particle_orders_array_accum[index]) - 1;
}

__device__ int calc_num_assoc(int num_new_particles, float p_A)
{
    return static_cast<int>(roundf(num_new_particles * p_A));
}

__device__ float calc_weight_assoc(int nu_A, float p_A, float born_mass)
{
    return nu_A > 0 ? (p_A * born_mass) / nu_A : 0.0;
}

__device__ float calc_weight_unassoc(int nu_UA, float p_A, float born_mass)
{
    return nu_UA > 0 ? ((1.0 - p_A) * born_mass) / nu_UA : 0.0;
}

__device__ void store_weights(float w_A, float w_UA, GridCell* __restrict__ grid_cell_array, int j)
{
    grid_cell_array[j].w_A = w_A;
    grid_cell_array[j].w_UA = w_UA;
}

void normalize_particle_orders(float* particle_orders_array_accum, int particle_orders_count, int v_B)
{
    thrust::device_ptr<float> particle_orders_accum(particle_orders_array_accum);

    float max = 1.0f;
    cudaMemcpy(&max, &particle_orders_array_accum[particle_orders_count - 1], sizeof(float), cudaMemcpyDeviceToHost);
    thrust::transform(particle_orders_accum, particle_orders_accum + particle_orders_count, particle_orders_accum,
                      GPU_LAMBDA(float x) { return x * (v_B / max); });
}

__global__ void copyMassesKernel(const MeasurementCell* __restrict__ meas_cell_array, float* __restrict__ masses,
                                 int cell_count)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < cell_count; j += blockDim.x * gridDim.x)
    {
        masses[j] = meas_cell_array[j].occ_mass;
    }
}

__global__ void initParticlesKernel1(GridCell* __restrict__ grid_cell_array,
                                     const MeasurementCell* __restrict__ meas_cell_array, ParticlesSoA particle_array,
                                     const float* __restrict__ particle_orders_array_accum, int cell_count)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < cell_count; j += blockDim.x * gridDim.x)
    {
        int start_idx = calc_start_idx(particle_orders_array_accum, j);
        int end_idx = calc_end_idx(particle_orders_array_accum, j);

        for (int i = start_idx; i < end_idx + 1; i++)
        {
            particle_array.grid_cell_idx[i] = j;
        }
    }
}

__global__ void initParticlesKernel2(ParticlesSoA particle_array, const GridCell* __restrict__ grid_cell_array,
                                     curandState* __restrict__ global_state, float velocity, int grid_size,
                                     float new_weight, int particle_count)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState local_state = global_state[thread_id];

    for (int i = thread_id; i < particle_count; i += stride)
    {
        int cell_idx = particle_array.grid_cell_idx[i];

        float x = cell_idx % grid_size + 0.5f;
        float y = cell_idx / grid_size + 0.5f;
        float vel_x = curand_uniform(&local_state, -velocity, velocity);
        float vel_y = curand_uniform(&local_state, -velocity, velocity);

        particle_array.weight[i] = new_weight;
        particle_array.state[i] = glm::vec4(x, y, vel_x, vel_y);
    }

    global_state[thread_id] = local_state;
}

__global__ void initNewParticlesKernel1(GridCell* __restrict__ grid_cell_array,
                                        const MeasurementCell* __restrict__ meas_cell_array,
                                        const float* __restrict__ weight_array,
                                        const float* __restrict__ born_masses_array, ParticlesSoA birth_particle_array,
                                        const float* __restrict__ particle_orders_array_accum, int cell_count)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < cell_count; j += blockDim.x * gridDim.x)
    {
        int start_idx = calc_start_idx(particle_orders_array_accum, j);
        int end_idx = calc_end_idx(particle_orders_array_accum, j);

        // printf("Start idx: %d, End idx: %d\n", start_idx, end_idx);

        int num_new_particles = start_idx <= end_idx ? end_idx - start_idx + 1 : 0;
        float p_A = meas_cell_array[j].p_A;
        int nu_A = calc_num_assoc(num_new_particles, p_A);
        int nu_UA = num_new_particles - nu_A;
        float w_A = calc_weight_assoc(nu_A, p_A, born_masses_array[j]);
        float w_UA = calc_weight_unassoc(nu_UA, p_A, born_masses_array[j]);
        store_weights(w_A, w_UA, grid_cell_array, j);

        // printf("w_A: %f, w_UA: %f\n", w_A, w_UA);

        for (int i = start_idx; i < start_idx + nu_A + 1; i++)
        {
            set_cell_idx_A(birth_particle_array, i, j);
        }

        for (int i = start_idx + nu_A + 1; i < end_idx + 1; i++)
        {
            set_cell_idx_UA(birth_particle_array, i, j);
        }
    }
}

__global__ void initNewParticlesKernel2(ParticlesSoA birth_particle_array, const GridCell* __restrict__ grid_cell_array,
                                        curandState* __restrict__ global_state, float stddev_velocity,
                                        float max_velocity, int grid_size, int particle_count)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState local_state = global_state[thread_id];

    for (int i = thread_id; i < particle_count; i += stride)
    {
        int cell_idx = birth_particle_array.grid_cell_idx[i];
        const GridCell& grid_cell = grid_cell_array[cell_idx];
        bool associated = birth_particle_array.associated[i];

        float x = cell_idx % grid_size + 0.5f;
        float y = cell_idx / static_cast<float>(grid_size) + 0.5f;

        if (associated)
        {
            // TODO: associated particles would use the doppler velocity in their velocity distribution
            float vel_x = curand_normal(&local_state, 0.0f, stddev_velocity);
            float vel_y = curand_normal(&local_state, 0.0f, stddev_velocity);

            birth_particle_array.weight[i] = grid_cell.w_A;
            birth_particle_array.state[i] = glm::vec4(x, y, vel_x, vel_y);
        }
        else
        {
            float vel_x = curand_normal(&local_state, 0.0f, stddev_velocity);
            float vel_y = curand_normal(&local_state, 0.0f, stddev_velocity);

            birth_particle_array.weight[i] = grid_cell.w_UA;
            birth_particle_array.state[i] = glm::vec4(x, y, vel_x, vel_y);
        }
    }

    global_state[thread_id] = local_state;
}

__global__ void copyBirthWeightKernel(const ParticlesSoA birth_particle_array, float* __restrict__ birth_weight_array,
                                      int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        birth_weight_array[i] = birth_particle_array.weight[i];
    }
}

} /* namespace dogm */
