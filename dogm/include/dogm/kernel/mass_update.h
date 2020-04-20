// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <device_launch_parameters.h>

namespace dogm
{

struct GridCell;
struct MeasurementCell;
struct Particle;

__global__ void gridCellPredictionUpdateKernel(GridCellSoA grid_cell_array, ParticleSoA particle_array,
                                               float* __restrict__ weight_array,
                                               const float* __restrict__ weight_array_accum,
                                               const MeasurementCell* __restrict__ meas_cell_array,
                                               float* __restrict__ born_masses_array, float p_B, int cell_count);

} /* namespace dogm */
