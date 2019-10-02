#include "kernel/particle_to_grid.h"
#include "cuda_utils.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ bool is_first_particle(Particle* particle_array, int i)
{
	return i == 0 || particle_array[i].grid_cell_idx != particle_array[i - 1].grid_cell_idx;
}

__device__ bool is_last_particle(Particle* particle_array, int particle_count, int i)
{
	return i == particle_count - 1 || particle_array[i].grid_cell_idx != particle_array[i + 1].grid_cell_idx;
}

__global__ void particleToGridKernel(Particle* particle_array, GridCell* grid_cell_array, float* weight_array, int particle_count)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < particle_count)
	{
		int j = particle_array[i].grid_cell_idx;

		if (is_first_particle(particle_array, i))
		{
			grid_cell_array[j].start_idx = i;
		}
		if (is_last_particle(particle_array, particle_count, i))
		{
			grid_cell_array[j].end_idx = i;
		}

		weight_array[i] = particle_array[i].weight;
	}
}
