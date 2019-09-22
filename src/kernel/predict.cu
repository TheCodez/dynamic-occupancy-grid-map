#include "kernel/predict.h"
#include "cuda_utils.h"
#include "common.h"

#include <thrust/random.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void predictKernel(Particle* particle_array, int width, int height, float ps, const glm::mat4x4 transition_matrix,
	const glm::vec4 process_noise, int particle_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
	{
		particle_array[i].state = transition_matrix * particle_array[i].state + process_noise;
		particle_array[i].weight = ps * particle_array[i].weight;

		float x = particle_array[i].state[0];
		float y = particle_array[i].state[1];

		if ((x > width - 1 || x < 0)
			|| (y > height - 1 || y < 0))
		{
			thrust::default_random_engine rng;
			thrust::uniform_int_distribution<int> dist_idx(0, width * height);

			int index = dist_idx(rng);

			x = index % width + 0.5f;
			y = index / width + 0.5f;
		}

		int pos_x = clamp(static_cast<int>(x), 0, width - 1);
		int pos_y = clamp(static_cast<int>(y), 0, height - 1);
		particle_array[i].grid_cell_idx = pos_x + width * pos_y;
	}
}
