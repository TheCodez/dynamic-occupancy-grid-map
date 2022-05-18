// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/ego_motion_compensation.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dogm
{

__global__ void moveParticlesKernel(ParticlesSoA particle_array, int x_move, int y_move, int particle_count,
                                    float resolution, int grid_size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        particle_array.state[i][0] -= (x_move * resolution);
        particle_array.state[i][1] -= (y_move * resolution);

        particle_array.grid_cell_idx[i] = static_cast<int>(particle_array.state[i][1] / resolution) * grid_size
            + static_cast<int>(particle_array.state[i][0] / resolution);
    }
}

__global__ void moveMapKernel(GridCellsSoA grid_cell_array, GridCellsSoA old_grid_cell_array,
                              MeasurementCellsSoA meas_cell_array, ParticlesSoA particle_array,
                              int x_move, int y_move, int grid_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    float eps = 0.001f;

    if (x < grid_size && y < grid_size)
    {
        int index = x + grid_size * y;
        int old_y = y + y_move;
        int old_x = x + x_move;
        int old_index = old_x + grid_size * old_y;

        if (old_x >= 0 && old_x < grid_size && old_y >= 0 && old_y < grid_size && meas_cell_array.occ_mass[index] > eps)
        {
            grid_cell_array.copy(old_grid_cell_array, index, old_index);
        }
        else
        {
            // delete particles on old cells? looks like it break something
            // for (int i = old_grid_cell_array.start_idx[old_index]; i < old_grid_cell_array.end_idx[old_index]; ++i)
            //     particle_array.weight[i] = 0;
            grid_cell_array.start_idx[index] = -1;
            grid_cell_array.end_idx[index] = -1;
            grid_cell_array.new_born_occ_mass[index] = 0.0f;
            grid_cell_array.pers_occ_mass[index] = 0.0f;
            grid_cell_array.free_mass[index] = 0.0f;
            grid_cell_array.occ_mass[index] = 0.0f;            
            grid_cell_array.pred_occ_mass[index] = 0.0f;

            grid_cell_array.mu_A[index] = 0.0f;
            grid_cell_array.mu_UA[index] = 0.0f;

            grid_cell_array.w_A[index] = 0.0f;
            grid_cell_array.w_UA[index] = 0.0f;

            grid_cell_array.mean_x_vel[index] = 0.0f;
            grid_cell_array.mean_y_vel[index] = 0.0f;
            grid_cell_array.var_x_vel[index] = 0.0f;
            grid_cell_array.var_y_vel[index] = 0.0f;
            grid_cell_array.covar_xy_vel[index] = 0.0f;
            
        }
    }
}

} /* namespace dogm */
