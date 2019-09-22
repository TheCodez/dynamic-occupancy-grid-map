#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void particleToGridKernel(Particle* particle_array, GridCell* grid_cell_array, float* weight_array, int particle_count);
