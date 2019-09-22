#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void resamplingKernel(Particle* particle_array, Particle* particle_array_next, Particle* birth_particle_array,
	float* rand_array, int* idx_array_resampled, int particle_count);
