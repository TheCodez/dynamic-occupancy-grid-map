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

__global__ void moveParticlesKernel(ParticlesSoA particle_array, int x_move, int y_move, int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        particle_array.state[i][0] -= x_move;
        particle_array.state[i][1] -= y_move;
    }
}

__global__ void moveMapKernel(GridCellsSoA grid_cell_array, GridCellsSoA old_grid_cell_array,
                              MeasurementCellsSoA meas_cell_array, ParticlesSoA particle_array,
                              int x_move, int y_move, int grid_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grid_size && y < grid_size)
    {
        int index = x + grid_size * y;
        int new_y = y + y_move;
        int new_x = x + x_move;
        int new_index = new_x + grid_size * new_y;

        if (new_x > 0 && new_x < grid_size && new_y > 0 && new_y < grid_size)
        {
            grid_cell_array.copy(old_grid_cell_array, index, new_index);
        }
    }
}

} /* namespace dogm */
