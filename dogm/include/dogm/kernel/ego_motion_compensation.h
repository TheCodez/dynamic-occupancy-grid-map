// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <curand_kernel.h>
#include <device_launch_parameters.h>

namespace dogm
{

__global__ void moveParticlesKernel(ParticlesSoA particle_array, int x_move, int y_move, int particle_count);

__global__ void moveMapKernel(GridCellsSoA grid_cell_array, GridCellsSoA old_grid_cell_array,
                              MeasurementCellsSoA meas_cell_array, ParticlesSoA particle_array,
                              int x_move, int y_move, int grid_size);

} /* namespace dogm */
