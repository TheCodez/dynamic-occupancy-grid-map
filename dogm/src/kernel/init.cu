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

#include <thrust/random.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void initParticlesKernel(Particle* particle_array, int grid_size, int particle_count)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < particle_count)
	{
		unsigned int seed = hash(i);
		thrust::default_random_engine rng(seed);
		thrust::uniform_int_distribution<int> dist_idx(0, grid_size * grid_size);
		thrust::normal_distribution<float> dist_vel(0.0f, 12.0f);

		int index = dist_idx(rng);

		float x = index % grid_size;
		float y = index / grid_size;

		particle_array[i].weight = 1.0f / particle_count;
		particle_array[i].state = glm::vec4(x, y, dist_vel(rng), dist_vel(rng));

		//printf("w: %f, x: %f, y: %f, vx: %f, vy: %f\n", particle_array[i].weight, particle_array[i].state[0], particle_array[i].state[1],
		//	particle_array[i].state[2], particle_array[i].state[3]);
	}
}

__global__ void initGridCellsKernel(GridCell* grid_cell_array, MeasurementCell* meas_cell_array, int grid_size, int cell_count)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < cell_count)
	{
		const int x = i % grid_size;
		const int y = i / grid_size;
		grid_cell_array[i].pos = make_int2(x, y);
		grid_cell_array[i].free_mass = 0.0f;
		grid_cell_array[i].occ_mass = 0.0f;

		meas_cell_array[i].occ_mass = 0.0f;
		meas_cell_array[i].free_mass = 0.0f;
		meas_cell_array[i].likelihood = 1.0f;
		meas_cell_array[i].p_A = 1.0f;
	}
}
