#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>

__global__ void predictKernel(Particle* particle_array, int width, int height, float p_S, const glm::mat4x4 transition_matrix,
	const glm::vec4 process_noise, int particle_count);
