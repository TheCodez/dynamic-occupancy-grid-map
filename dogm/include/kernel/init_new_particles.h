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

#include <curand_kernel.h>
#include <device_launch_parameters.h>

namespace dogm
{

struct GridCell;
struct MeasurementCell;
struct Particle;

void normalize_particle_orders(float* particle_orders_array_accum, int particle_orders_count, int v_B);

__global__ void initNewParticlesKernel1(Particle* __restrict__ particle_array, GridCell* __restrict__ grid_cell_array,
	const MeasurementCell *__restrict__ meas_cell_array, const float *__restrict__ weight_array, const float *__restrict__ born_masses_array, 
	Particle* __restrict__ birth_particle_array, const float *__restrict__ particle_orders_array_accum, int cell_count);

__global__ void initNewParticlesKernel2(Particle* __restrict__ birth_particle_array, const GridCell* __restrict__ grid_cell_array, 
	curandState* __restrict__ global_state, float velocity, int grid_size, int particle_count);

__global__ void copyBirthWeightKernel(const Particle* __restrict__ birth_particle_array, float* __restrict__ birth_weight_array,
	int particle_count);

} /* namespace dogm */
