/*
MIT License

Copyright (c) 2019 Michael Kösel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "kernel/resampling.h"
#include "common.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/binary_search.h>

void calc_resampled_indeces(thrust::device_vector<double>& joint_weight_accum, thrust::device_vector<int>& rand_array,
	thrust::device_vector<int>& indices)
{
	thrust::device_vector<double> norm_weight_accum(joint_weight_accum.size());
	double max = joint_weight_accum.back();
	size_t size = joint_weight_accum.size();
	thrust::transform(joint_weight_accum.begin(), joint_weight_accum.end(), norm_weight_accum.begin(), GPU_LAMBDA(double x)
	{
		return  x * (size / max);
	});

	double norm_max = norm_weight_accum.back();
	int rand_max = rand_array.back();

	if (norm_max != rand_max)
	{
		norm_weight_accum.back() = static_cast<double>(rand_max);
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
	int* idx_array_resampled, double joint_max, int particle_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
	{
		particle_array_next[i] = copy_particle(particle_array, particle_count, birth_particle_array, idx_array_resampled[i]);
		particle_array_next[i].weight = joint_max / particle_count;
	}
}
