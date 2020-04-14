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
#pragma once

#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <curand_kernel.h>

namespace dogm
{

struct Particle;

__global__ void resamplingGenerateRandomNumbersKernel(float* __restrict__ rand_array,
                                                      curandState* __restrict__ global_state, float max,
                                                      int particle_count);

void calc_resampled_indices(thrust::device_vector<float>& weight_accum, thrust::device_vector<float>& rand_array,
                            thrust::device_vector<int>& indices);

__global__ void resamplingKernel(const ParticleSoA particle_array,
                                 ParticleSoA particle_array_next,
                                 const ParticleSoA birth_particle_array,
                                 const int* __restrict__ idx_array_resampled, float new_weight, int particle_count);

} /* namespace dogm */
