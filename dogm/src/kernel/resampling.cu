// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/resampling.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/binary_search.h>

namespace dogm
{

__global__ void resamplingGenerateRandomNumbersKernel(float* __restrict__ rand_array,
                                                      curandState* __restrict__ global_state, float max,
                                                      int particle_count)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState local_state = global_state[thread_id];

    for (int i = thread_id; i < particle_count; i += stride)
    {
        rand_array[i] = curand_uniform(&local_state, 0.0f, max);
    }

    global_state[thread_id] = local_state;
}

void calc_resampled_indices(thrust::device_vector<float>& joint_weight_accum, thrust::device_vector<float>& rand_array,
                            thrust::device_vector<int>& indices, float accum_max)
{
    float rand_max = rand_array.back();

    if (accum_max != rand_max)
    {
        joint_weight_accum.back() = rand_max;
    }

    // multinomial sampling
    thrust::lower_bound(joint_weight_accum.begin(), joint_weight_accum.end(), rand_array.begin(), rand_array.end(),
                        indices.begin());
}

__device__ Particle copy_particle(const Particle* __restrict__ particle_array, int particle_count,
                                  const Particle* __restrict__ birth_particle_array, int idx)
{
    if (idx < particle_count)
    {
        return particle_array[idx];
    }
    else
    {
        return birth_particle_array[idx - particle_count];
    }
}

__global__ void resamplingKernel(const ParticleSoA particle_array, ParticleSoA particle_array_next,
                                 const ParticleSoA birth_particle_array, const int* __restrict__ idx_array_resampled,
                                 float new_weight, int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        int idx = idx_array_resampled[i];

        if (idx < particle_count)
        {
            particle_array_next.grid_cell_idx[i] = particle_array.grid_cell_idx[idx];
            particle_array_next.weight[i] = particle_array.weight[idx];
            particle_array_next.associated[i] = particle_array.associated[idx];
            particle_array_next.state[i] = particle_array.state[idx];
        }
        else
        {
            particle_array_next.grid_cell_idx[i] = birth_particle_array.grid_cell_idx[idx - particle_count];
            particle_array_next.weight[i] = birth_particle_array.weight[idx - particle_count];
            particle_array_next.associated[i] = birth_particle_array.associated[idx - particle_count];
            particle_array_next.state[i] = birth_particle_array.state[idx - particle_count];
        }

        particle_array_next.weight[i] = new_weight;
    }
}

} /* namespace dogm */
