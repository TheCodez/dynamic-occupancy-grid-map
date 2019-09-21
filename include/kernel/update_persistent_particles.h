#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void updatePersistentParticlesKernel1(Particle* particle_array, MeasurementCell* meas_cell_array, float* weight_array);

__global__ void updatePersistentParticlesKernel2(GridCell* grid_cell_array, float* weight_array_accum);

__global__ void updatePersistentParticlesKernel3(Particle* particle_array, MeasurementCell* meas_cell_array, 
	GridCell* grid_cell_array, float* weight_array);
