// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/update_persistent_particles.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dogm
{

__device__ float calc_norm_assoc(float occ_accum, float rho_p)
{
    return occ_accum > 0.0f ? rho_p / occ_accum : 0.0f;
}

__device__ float calc_norm_unassoc(const GridCell& grid_cell)
{
    float pred_occ_mass = grid_cell.pred_occ_mass;
    return pred_occ_mass > 0.0f ? grid_cell.pers_occ_mass / pred_occ_mass : 0.0f;
}

__device__ void set_normalization_components(GridCell* __restrict__ grid_cell_array, int i, float mu_A, float mu_UA)
{
    grid_cell_array[i].mu_A = mu_A;
    grid_cell_array[i].mu_UA = mu_UA;
}

__device__ float update_unnorm(const ParticlesSoA& particle_array, int i,
                               const MeasurementCell* __restrict__ meas_cell_array)
{
    return meas_cell_array[particle_array.grid_cell_idx[i]].likelihood * particle_array.weight[i];
}

__device__ float normalize(const ParticlesSoA& particle, int i, const GridCell* __restrict__ grid_cell_array,
                           const MeasurementCell* __restrict__ meas_cell_array, float weight)
{
    const int cell_idx = particle.grid_cell_idx[i];
    const GridCell& cell = grid_cell_array[cell_idx];
    const MeasurementCell& meas_cell = meas_cell_array[cell_idx];

    return meas_cell.p_A * cell.mu_A * weight + (1.0f - meas_cell.p_A) * cell.mu_UA * particle.weight[i];
}

__global__ void updatePersistentParticlesKernel1(const ParticlesSoA particle_array,
                                                 const MeasurementCell* __restrict__ meas_cell_array,
                                                 float* __restrict__ weight_array, int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        weight_array[i] = update_unnorm(particle_array, i, meas_cell_array);
    }
}

__global__ void updatePersistentParticlesKernel2(GridCell* __restrict__ grid_cell_array,
                                                 const float* __restrict__ weight_array_accum, int cell_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
    {
        int start_idx = grid_cell_array[i].start_idx;
        int end_idx = grid_cell_array[i].end_idx;

        if (start_idx != -1)
        {
            float m_occ_accum = subtract(weight_array_accum, start_idx, end_idx);
            float rho_p = grid_cell_array[i].pers_occ_mass;
            float mu_A = calc_norm_assoc(m_occ_accum, rho_p);
            float mu_UA = calc_norm_unassoc(grid_cell_array[i]);
            set_normalization_components(grid_cell_array, i, mu_A, mu_UA);
            // printf("mu_A: %f, mu_UA: %f\n", mu_A, mu_UA);
        }
    }
}

__global__ void updatePersistentParticlesKernel3(const ParticlesSoA particle_array,
                                                 const MeasurementCell* __restrict__ meas_cell_array,
                                                 const GridCell* __restrict__ grid_cell_array,
                                                 float* __restrict__ weight_array, int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        weight_array[i] = normalize(particle_array, i, grid_cell_array, meas_cell_array, weight_array[i]);
    }
}

} /* namespace dogm */
