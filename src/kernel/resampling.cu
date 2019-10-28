#include "kernel/resampling.h"
#include "common.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/binary_search.h>


void calc_resampled_indeces(thrust::device_vector<float>& joint_weight_accum, thrust::device_vector<float>& rand_array,
	thrust::device_vector<int>& indices)
{
	thrust::device_vector<float> norm_weight_accum(joint_weight_accum.size());
	float max = joint_weight_accum.back();
	size_t size = joint_weight_accum.size();
	thrust::transform(joint_weight_accum.begin(), joint_weight_accum.end(), norm_weight_accum.begin(), GPU_LAMBDA(float x)
	{
		return  x * (size / max);
	});

	float norm_max = norm_weight_accum.back();
	float rand_max = rand_array.back();

	if (norm_max != rand_max)
	{
		norm_weight_accum.back() = rand_max;
	}

	// multinomial sampling
	thrust::lower_bound(norm_weight_accum.begin(), norm_weight_accum.end(), rand_array.begin(), rand_array.end(), indices.begin());
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
	int* idx_array_resampled, float joint_max, int particle_count)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < particle_count)
	{
		particle_array_next[i] = copy_particle(particle_array, particle_count, birth_particle_array, idx_array_resampled[i]);
		particle_array_next[i].weight = joint_max / particle_count;
	}
}
