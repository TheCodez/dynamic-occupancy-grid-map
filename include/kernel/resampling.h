#pragma once

#include "occupancy_grid_map.h"
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void calc_resampled_indeces(thrust::device_vector<float>& weight_accum, thrust::device_vector<float>& rand_array,
	thrust::device_vector<int>& indices);

__global__ void resamplingKernel(Particle* particle_array, Particle* particle_array_next, Particle* birth_particle_array,
	int* idx_array_resampled, float joint_max, int particle_count);
