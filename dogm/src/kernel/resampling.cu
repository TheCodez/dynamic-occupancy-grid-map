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
#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/resampling.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/binary_search.h>

namespace dogm
{

__global__ void resamplingGenerateRandomNumbersKernel(float* __restrict__ rand_array,
                                                      curandState* __restrict__ global_state, float max,
                                                      int particle_count)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState local_state = global_state[thread_id];

    for (int i = thread_id; i < particle_count; i += stride)
    {
        rand_array[i] = curand_uniform(&local_state, 0.0f, max);
    }

    global_state[thread_id] = local_state;
}

void calc_resampled_indices(thrust::device_vector<float>& joint_weight_accum, thrust::device_vector<float>& rand_array,
                            thrust::device_vector<int>& indices)
{
    thrust::device_vector<float> norm_weight_accum(joint_weight_accum.size());
    float max = joint_weight_accum.back();
    size_t size = joint_weight_accum.size();
    thrust::transform(joint_weight_accum.begin(), joint_weight_accum.end(), norm_weight_accum.begin(),
                      GPU_LAMBDA(float x) { return x * (size / max); });

    float norm_max = norm_weight_accum.back();
    float rand_max = rand_array.back();

    printf("Norm: %f, Rand: %f\n", norm_max, rand_max);

    if (norm_max != rand_max)
    {
        norm_weight_accum.back() = rand_max;
    }

    // multinomial sampling
    thrust::lower_bound(norm_weight_accum.begin(), norm_weight_accum.end(), rand_array.begin(), rand_array.end(),
                        indices.begin());
}

__device__ Particle copy_particle(const Particle* __restrict__ particle_array, int particle_count,
                                  const Particle* __restrict__ birth_particle_array, int idx)
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

__global__ void resamplingKernel(const Particle* __restrict__ particle_array,
                                 Particle* __restrict__ particle_array_next,
                                 const Particle* __restrict__ birth_particle_array,
                                 const int* __restrict__ idx_array_resampled, float new_weight, int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        particle_array_next[i] =
            copy_particle(particle_array, particle_count, birth_particle_array, idx_array_resampled[i]);
        particle_array_next[i].weight = new_weight;
    }
}

} /* namespace dogm */
