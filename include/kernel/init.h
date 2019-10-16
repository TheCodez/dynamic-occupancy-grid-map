#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void initParticlesKernel(Particle* particle_array, int grid_size, int particle_count);

__global__ void initGridCellsKernel(GridCell* grid_cell_array, int grid_size, int cell_count);
