// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/particle_to_grid.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>

namespace dogm
{

__device__ bool is_first_particle(const ParticlesSoA& particle_array, int i)
{
    return i == 0 || particle_array.grid_cell_idx[i] != particle_array.grid_cell_idx[i - 1];
}

__device__ bool is_last_particle(const ParticlesSoA& particle_array, int particle_count, int i)
{
    return i == particle_count - 1 || particle_array.grid_cell_idx[i] != particle_array.grid_cell_idx[i + 1];
}

__global__ void particleToGridKernel(const ParticlesSoA particle_array, GridCell* __restrict__ grid_cell_array,
                                     float* __restrict__ weight_array, int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        int cell_idx = particle_array.grid_cell_idx[i];

        if (is_first_particle(particle_array, i))
        {
            grid_cell_array[cell_idx].start_idx = i;
        }
        if (is_last_particle(particle_array, particle_count, i))
        {
            grid_cell_array[cell_idx].end_idx = i;
        }

        weight_array[i] = particle_array.weight[i];
    }
}

} /* namespace dogm */
