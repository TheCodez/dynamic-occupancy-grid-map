// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <device_launch_parameters.h>

namespace dogm
{

__global__ void gridCellPredictionUpdateKernel(GridCellsSoA grid_cell_array, ParticlesSoA particle_array,
                                               float* __restrict__ weight_array,
                                               const float* __restrict__ weight_array_accum,
                                               const MeasurementCellsSoA meas_cell_array,
                                               float* __restrict__ born_masses_array, float p_B, int cell_count);

} /* namespace dogm */
