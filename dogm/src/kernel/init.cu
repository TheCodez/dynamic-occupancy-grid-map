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
#include "kernel/init.h"
#include "common.h"
#include "cuda_utils.h"
#include "dogm_types.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dogm
{__global__ void setupRandomStatesKernel(curandState* __restrict__ states, unsigned long long seed, int count){
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
	{
		curand_init(seed, i, 0, &states[i]);
	}
}

__global__ void initParticlesKernel(Particle* __restrict__ particle_array, curandState* __restrict__ global_state, 
	float velocity, int grid_size, int particle_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
	{
		curandState local_state = global_state[i];

		float x = curand_uniform(&local_state, 0.0f, grid_size - 1);
		float y = curand_uniform(&local_state, 0.0f, grid_size - 1);
		float vel_x = curand_normal(&local_state, 0.0f, velocity);
		float vel_y = curand_normal(&local_state, 0.0f, velocity);

		particle_array[i].weight = 1.0f / particle_count;
		particle_array[i].state = glm::vec4(x, y, vel_x, vel_y);

		global_state[i] = local_state;

		//printf("w: %f, x: %f, y: %f, vx: %f, vy: %f\n", particle_array[i].weight, particle_array[i].state[0], particle_array[i].state[1],
		//	particle_array[i].state[2], particle_array[i].state[3]);
	}
}

__global__ void initBirthParticlesKernel(Particle* __restrict__ birth_particle_array, curandState* __restrict__ global_state,
	float velocity, int grid_size, int particle_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
	{
		curandState local_state = global_state[i];

		float x = curand_uniform(&local_state, 0.0f, grid_size - 1);
		float y = curand_uniform(&local_state, 0.0f, grid_size - 1);
		float vel_x = curand_normal(&local_state, 0.0f, velocity);
		float vel_y = curand_normal(&local_state, 0.0f, velocity);

		birth_particle_array[i].weight = 0.0;
		birth_particle_array[i].associated = false;
		birth_particle_array[i].state = glm::vec4(x, y, vel_x, vel_y);

		global_state[i] = local_state;
	}
}

__global__ void initGridCellsKernel(GridCell* __restrict__ grid_cell_array, MeasurementCell* __restrict__ meas_cell_array, 
	int grid_size, int cell_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
	{
		grid_cell_array[i].free_mass = 0.0f;
		grid_cell_array[i].occ_mass = 0.0f;
		grid_cell_array[i].start_idx = -1;
		grid_cell_array[i].end_idx = -1;

		meas_cell_array[i].occ_mass = 0.0f;
		meas_cell_array[i].free_mass = 0.0f;
		meas_cell_array[i].likelihood = 1.0f;
		meas_cell_array[i].p_A = 1.0f;
	}
}

__global__ void reinitGridParticleIndices(GridCell* __restrict__ grid_cell_array, int cell_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
	{
		grid_cell_array[i].start_idx = -1;
		grid_cell_array[i].end_idx = -1;
	}
}

} /* namespace dogm */
