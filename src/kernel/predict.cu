#include "kernel/predict.h"
#include "cuda_utils.h"
#include "common.h"

#include <thrust/random.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void predictKernel(Particle* particle_array, int width, int height, float p_S, const glm::mat4x4 transition_matrix,
	const glm::vec4 process_noise, int particle_count)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < particle_count)
	{
		particle_array[i].state = transition_matrix * particle_array[i].state + process_noise;
		particle_array[i].weight = p_S * particle_array[i].weight;

		float x = particle_array[i].state[0];
		float y = particle_array[i].state[1];

		//printf("X: %f, Y: %f\n", x, y);

		if ((x > width - 1 || x < 0)
			|| (y > height - 1 || y < 0))
		{
			unsigned int seed = hash(i);
			thrust::default_random_engine rng(seed);
			thrust::uniform_int_distribution<int> dist_idx(0, width * height);

			const int index = dist_idx(rng);

			x = index % width + 0.5f;
			y = index / width + 0.5f;
		}

		int pos_x = clamp(static_cast<int>(x), 0, width - 1);
		int pos_y = clamp(static_cast<int>(y), 0, height - 1);
		particle_array[i].grid_cell_idx = pos_x + width * pos_y;

		//printf("X: %d, Y: %d, Cell index: %d\n", pos_x, pos_y, (pos_x + width * pos_y));
	}
}
