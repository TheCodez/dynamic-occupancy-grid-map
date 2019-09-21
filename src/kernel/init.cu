#include "kernel/init.h"
#include "common.h"
#include "cuda_utils.h"

#include <thrust/random.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void initParticlesKernel(Particle* particle_array, int width, int height, int particles_size)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(particle_array); i += blockDim.x * gridDim.x)
	{
		thrust::default_random_engine rng;
		thrust::uniform_int_distribution<int> dist_idx(0, width * height);
		thrust::normal_distribution<float> dist_vel(0.0f, 4.0f);

		int index = dist_idx(rng);

		float x = index % width + 0.5f;
		float y = index / width + 0.5f;

		particle_array[i].weight = 1.0f / particles_size;
		particle_array[i].state = glm::vec4(x, y, dist_vel(rng), dist_vel(rng));
	}
}
