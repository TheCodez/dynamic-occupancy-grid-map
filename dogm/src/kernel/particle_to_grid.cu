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
#include "kernel/particle_to_grid.h"
#include "cuda_utils.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ bool is_first_particle(Particle* particle_array, int i)
{
	return i == 0 || particle_array[i].grid_cell_idx != particle_array[i - 1].grid_cell_idx;
}

__device__ bool is_last_particle(Particle* particle_array, int particle_count, int i)
{
	return i == particle_count - 1 || particle_array[i].grid_cell_idx != particle_array[i + 1].grid_cell_idx;
}

__global__ void particleToGridKernel(Particle* particle_array, GridCell* grid_cell_array, double* weight_array, int particle_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
	{
		int j = particle_array[i].grid_cell_idx;

		if (is_first_particle(particle_array, i))
		{
			grid_cell_array[j].start_idx = i;
		}
		if (is_last_particle(particle_array, particle_count, i))
		{
			grid_cell_array[j].end_idx = i;
		}

		//printf("Cell: %d, Start idx: %d, End idx: %d\n", j, grid_cell_array[j].start_idx, grid_cell_array[j].end_idx);
		weight_array[i] = particle_array[i].weight;
	}
}
