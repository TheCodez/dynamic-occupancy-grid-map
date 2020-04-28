// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/statistical_moments.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dogm
{

__device__ float calc_mean(const float* __restrict__ vel_array_accum, int start_idx, int end_idx, float rho_p)
{
    if (rho_p > 0.0f)
    {
        float vel_accum = subtract(vel_array_accum, start_idx, end_idx);
        return (1.0f / rho_p) * vel_accum;
    }
    return 0.0f;
}

__device__ float calc_variance(const float* __restrict__ vel_squared_array_accum, int start_idx, int end_idx,
                               float rho_p, float mean_vel)
{
    if (rho_p > 0.0f)
    {
        float vel_accum = subtract(vel_squared_array_accum, start_idx, end_idx);
        return (1.0f / rho_p) * vel_accum - mean_vel * mean_vel;
    }
    return 0.0f;
}

__device__ float calc_covariance(const float* __restrict__ vel_xy_array_accum, int start_idx, int end_idx, float rho_p,
                                 float mean_x_vel, float mean_y_vel)
{
    if (rho_p > 0.0f)
    {
        float vel_accum = subtract(vel_xy_array_accum, start_idx, end_idx);
        return (1.0f / rho_p) * vel_accum - mean_x_vel * mean_y_vel;
    }
    return 0.0f;
}

__device__ void store(GridCell* __restrict__ grid_cell_array, int j, float mean_x_vel, float mean_y_vel,
                      float var_x_vel, float var_y_vel, float covar_xy_vel)
{
    grid_cell_array[j].mean_x_vel = mean_x_vel;
    grid_cell_array[j].mean_y_vel = mean_y_vel;
    grid_cell_array[j].var_x_vel = var_x_vel;
    grid_cell_array[j].var_y_vel = var_y_vel;
    grid_cell_array[j].covar_xy_vel = covar_xy_vel;
}

__global__ void statisticalMomentsKernel1(const ParticlesSoA particle_array, const float* __restrict__ weight_array,
                                          float* __restrict__ vel_x_array, float* __restrict__ vel_y_array,
                                          float* __restrict__ vel_x_squared_array,
                                          float* __restrict__ vel_y_squared_array, float* __restrict__ vel_xy_array,
                                          int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        float weight = weight_array[i];
        float vel_x = particle_array.state[i][2];
        float vel_y = particle_array.state[i][3];
        vel_x_array[i] = weight * vel_x;
        vel_y_array[i] = weight * vel_y;
        vel_x_squared_array[i] = weight * vel_x * vel_x;
        vel_y_squared_array[i] = weight * vel_y * vel_y;
        vel_xy_array[i] = weight * vel_x * vel_y;

        // printf("vx: %f, vy: %f\n", vel_x_array[i], vel_y_array[i]);
    }
}

__global__ void statisticalMomentsKernel2(GridCell* __restrict__ grid_cell_array,
                                          const float* __restrict__ vel_x_array_accum,
                                          const float* __restrict__ vel_y_array_accum,
                                          const float* __restrict__ vel_x_squared_array_accum,
                                          const float* __restrict__ vel_y_squared_array_accum,
                                          const float* __restrict__ vel_xy_array_accum, int cell_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
    {
        int start_idx = grid_cell_array[i].start_idx;
        int end_idx = grid_cell_array[i].end_idx;
        float rho_p = grid_cell_array[i].pers_occ_mass;
        // printf("rho p: %f\n", rho_p);

        if (start_idx != -1)
        {
            float mean_x_vel = calc_mean(vel_x_array_accum, start_idx, end_idx, rho_p);
            float mean_y_vel = calc_mean(vel_y_array_accum, start_idx, end_idx, rho_p);
            float var_x_vel = calc_variance(vel_x_squared_array_accum, start_idx, end_idx, rho_p, mean_x_vel);
            float var_y_vel = calc_variance(vel_y_squared_array_accum, start_idx, end_idx, rho_p, mean_y_vel);
            float covar_xy_vel = calc_covariance(vel_xy_array_accum, start_idx, end_idx, rho_p, mean_x_vel, mean_y_vel);
            // printf("x: %f, y: %f\n", mean_x_vel, mean_y_vel);

            store(grid_cell_array, i, mean_x_vel, mean_y_vel, var_x_vel, var_y_vel, covar_xy_vel);
        }
        else
        {
            store(grid_cell_array, i, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

} /* namespace dogm */
