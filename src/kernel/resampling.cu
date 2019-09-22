#include "kernel/resampling.h"
#include "common.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ Particle copy_particle(Particle* particle_array, int particle_count, Particle* birth_particle_array, int idx)
{
	if (idx < particle_count)
	{
		return particle_array[idx];
	}
	else
	{
		return birth_particle_array[idx - particle_count];
	}
}

__global__ void resamplingKernel(Particle* particle_array, Particle* particle_array_next, Particle* birth_particle_array,
	float* rand_array, int* idx_array_resampled, int particle_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
	{
		//particle_array_next[i] = copy_particle(particle_array, particle_count, birth_particle_array, idx_array_resampled[i]);
	}
}
