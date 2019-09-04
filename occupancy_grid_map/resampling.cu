#include "occupancy_grid_map.h"
#include "common.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ Particle copy_particle(Particle* particle_array, Particle* birth_particle_array, int idx)
{
	if (idx < ARRAY_SIZE(particle_array))
	{
		return particle_array[idx];
	}
	else
	{
		return birth_particle_array[idx - ARRAY_SIZE(particle_array)];
	}
}

__global__ void resamplingKernel(Particle* particle_array, Particle* particle_array_next, Particle* birth_particle_array,
	float* rand_array, int* idx_array_resampled)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(particle_array); i += blockDim.x * gridDim.x)
	{
		particle_array_next[i] = copy_particle(particle_array, birth_particle_array, idx_array_resampled[i]);
	}
}

void OccupancyGridMap::resampling()
{
	resamplingKernel<<<divUp(ARRAY_SIZE(particle_array), 256), 256>>>(particle_array, particle_array/*_next*/, birth_particle_array,
		rand_array, nullptr/*idx_array_resampled*/);

	CHECK_ERROR(cudaGetLastError());
}
