#include "kernel/resampling.h"
#include "common.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__host__ __device__ void calc_resampled_indeces(float* joint_weight_array_accum, float* rand_array, int* idx_array_resampled)
{

}

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
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < particle_count)
	{
		//particle_array_next[i] = copy_particle(particle_array, particle_count, birth_particle_array, idx_array_resampled[i]);
	}
}
