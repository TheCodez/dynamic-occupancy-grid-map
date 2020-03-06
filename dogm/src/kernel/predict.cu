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
#include "kernel/predict.h"
#include "cuda_utils.h"
#include "common.h"

#include <thrust/random.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void predictKernel(Particle* particle_array, int grid_size, double p_S, const glm::mat4x4 transition_matrix,
	const glm::vec4 process_noise, int particle_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
	{
		particle_array[i].state = transition_matrix * particle_array[i].state + process_noise;
		particle_array[i].weight = p_S * particle_array[i].weight;

		double x = particle_array[i].state[0];
		double y = particle_array[i].state[1];

		if ((x > grid_size - 1 || x < 0) || (y > grid_size - 1 || y < 0))
		{
			unsigned int seed = hash(i);
			thrust::default_random_engine rng(seed);
			thrust::uniform_int_distribution<int> dist_idx(0, grid_size * grid_size);
			thrust::normal_distribution<double> dist_vel(0.0f, 12.0);

			const int index = dist_idx(rng);

			x = index % grid_size;
			y = index / grid_size;

			particle_array[i].state = glm::vec4(x, y, dist_vel(rng), dist_vel(rng));
		}

		int pos_x = clamp(static_cast<int>(x), 0, grid_size - 1);
		int pos_y = clamp(static_cast<int>(y), 0, grid_size - 1);
		particle_array[i].grid_cell_idx = pos_x + grid_size * pos_y;

		//printf("X: %d, Y: %d, Cell index: %d\n", pos_x, pos_y, (pos_x + grid_size * pos_y));
	}
}
