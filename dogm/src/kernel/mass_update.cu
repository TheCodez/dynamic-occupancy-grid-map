// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/mass_update.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dogm
{

__device__ float predict_free_mass(const GridCell& grid_cell, float m_occ_pred, float alpha = 0.9)
{
    return min(alpha * grid_cell.free_mass, 1.0f - m_occ_pred);
}

__device__ float2 update_masses(float m_occ_pred, float m_free_pred, const MeasurementCell& meas_cell)
{
    float unknown_pred = 1.0 - m_occ_pred - m_free_pred;
    float meas_unknown = 1.0 - meas_cell.free_mass - meas_cell.occ_mass;
    float K = m_free_pred * meas_cell.occ_mass + m_occ_pred * meas_cell.free_mass;

    float occ_mass =
        (m_occ_pred * meas_unknown + unknown_pred * meas_cell.occ_mass + m_occ_pred * meas_cell.occ_mass) / (1.0 - K);
    float free_mass =
        (m_free_pred * meas_unknown + unknown_pred * meas_cell.free_mass + m_free_pred * meas_cell.free_mass) /
        (1.0 - K);

    return make_float2(occ_mass, free_mass);
}

__device__ float separate_newborn_part(float m_occ_pred, float m_occ_up, float p_B)
{
    return (m_occ_up * p_B * (1.0 - m_occ_pred)) / (m_occ_pred + p_B * (1.0 - m_occ_pred));
}

__device__ void store_values(float rho_b, float rho_p, float m_free_up, float m_occ_up, float m_occ_pred,
                             GridCell* __restrict__ grid_cell_array, int i)
{
    grid_cell_array[i].pers_occ_mass = rho_p;
    grid_cell_array[i].new_born_occ_mass = rho_b;
    grid_cell_array[i].free_mass = m_free_up;
    grid_cell_array[i].occ_mass = m_occ_up;
    grid_cell_array[i].pred_occ_mass = m_occ_pred;
}

__device__ void normalize_weights(const ParticlesSoA& particle_array, float* __restrict__ weight_array, int start_idx,
                                  int end_idx, float occ_pred)
{
    for (int i = start_idx; i < end_idx + 1; i++)
    {
        weight_array[i] = weight_array[i] / occ_pred;
        particle_array.weight[i] = weight_array[i];
    }
}

__global__ void gridCellPredictionUpdateKernel(GridCell* __restrict__ grid_cell_array, ParticlesSoA particle_array,
                                               float* __restrict__ weight_array,
                                               const float* __restrict__ weight_array_accum,
                                               const MeasurementCell* __restrict__ meas_cell_array,
                                               float* __restrict__ born_masses_array, float p_B, int cell_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
    {
        int start_idx = grid_cell_array[i].start_idx;
        int end_idx = grid_cell_array[i].end_idx;

        if (start_idx != -1)
        {
            float m_occ_pred = subtract(weight_array_accum, start_idx, end_idx);

            if (m_occ_pred > 1.0f)
            {
                // printf("Predicted mass greater 1. Mass is: %f\n", m_occ_pred);
                normalize_weights(particle_array, weight_array, start_idx, end_idx, m_occ_pred);
                m_occ_pred = 1.0f;
            }

            float m_free_pred = predict_free_mass(grid_cell_array[i], m_occ_pred);
            float2 masses_up = update_masses(m_occ_pred, m_free_pred, meas_cell_array[i]);
            float rho_b = separate_newborn_part(m_occ_pred, masses_up.x, p_B);
            float rho_p = masses_up.x - rho_b;
            born_masses_array[i] = rho_b;

            // printf("Rho B: %f\n", rho_b);

            store_values(rho_b, rho_p, masses_up.y, masses_up.x, m_occ_pred, grid_cell_array, i);
        }
        else
        {
            float m_occ = grid_cell_array[i].occ_mass;
            float m_free = predict_free_mass(grid_cell_array[i], m_occ);
            float2 masses_up = update_masses(m_occ, m_free, meas_cell_array[i]);
            born_masses_array[i] = 0.0f;
            store_values(0.0f, masses_up.x, masses_up.y, masses_up.x, 0.0f, grid_cell_array, i);
        }
    }
}

} /* namespace dogm */
