#include "occupancy_grid_map.h"
#include "cuda_utils.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ bool is_first_particle(Particle* particle_array, int i)
{
	return i == 0 || particle_array[i].grid_cell_idx != particle_array[i - 1].grid_cell_idx;
}

__device__ bool is_last_particle(Particle* particle_array, int i)
{
	return i == ARRAY_SIZE(particle_array) - 1 || particle_array[i].grid_cell_idx != particle_array[i + 1].grid_cell_idx;
}

__global__ void particleToGridKernel(Particle* particle_array, GridCell* grid_cell_array, float* weight_array)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(particle_array); i += blockDim.x * gridDim.x)
	{
		int j = particle_array[i].grid_cell_idx;

		if (is_first_particle(particle_array, i))
		{
			grid_cell_array[j].start_idx = i;
		}
		if (is_last_particle(particle_array, i))
		{
			grid_cell_array[j].end_idx = i;
		}

		weight_array[i] = particle_array[i].weight;
	}
}

void OccupancyGridMap::particleAssignment()
{
	struct sort_particles
	{
		__host__ __device__ bool operator()(Particle x, Particle y)
		{
			return x.grid_cell_idx < y.grid_cell_idx;
		}
	};

	CHECK_ERROR(cudaDeviceSynchronize());
	thrust::device_ptr<Particle> particles(particle_array);
	thrust::sort(particles, particles + ARRAY_SIZE(particle_array), sort_particles());

	particleToGridKernel<<<divUp(ARRAY_SIZE(particle_array), 256), 256>>>(particle_array, grid_cell_array, weight_array);

	CHECK_ERROR(cudaGetLastError());
}