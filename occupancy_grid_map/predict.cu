#include "occupancy_grid_map.h"
#include "cuda_utils.h"

#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void predictKernel(Particle* particle_array, int width, int height, float ps, const Eigen::Matrix4f transition_matrix,
	const Eigen::Vector4f process_noise)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(particle_array); i += blockDim.x * gridDim.x)
	{
		particle_array[i].state = transition_matrix * particle_array[i].state + process_noise;
		particle_array[i].weight = ps * particle_array[i].weight;

		float x = particle_array[i].state[0];
		float y = particle_array[i].state[1];

		if ((x > width - 1 || x < 0)
			|| (y > height - 1 || y < 0))
		{
			// TODO: resolution?
			int size = width * height;

			thrust::default_random_engine rng;
			thrust::uniform_int_distribution<int> dist_idx(0, width * height);

			int index = dist_idx(rng);

			int x = index % width + 0.5f;
			int y = index / width + 0.5f;
		}

		x = std::max(std::min((int)x, width - 1), 0);
		y = std::max(std::min((int)y, height - 1), 0);
		particle_array[i].grid_cell_idx = x + width * y;
	}
}

void OccupancyGridMap::particlePrediction(float dt)
{
	Eigen::Matrix4f transition_matrix;
	transition_matrix << 1, 0, dt, 0,
						 0, 1, 0, dt,
						 0, 0, 1, 0,
						 0, 0, 0, 1;

	thrust::default_random_engine rng;
	thrust::normal_distribution<float> dist_pos(0.0f, params.process_noise_position);
	thrust::normal_distribution<float> dist_vel(0.0f, params.process_noise_velocity);

	Eigen::Vector4f process_noise;
	process_noise << dist_pos(rng), dist_pos(rng), dist_vel(rng), dist_vel(rng);

	predictKernel<<<divUp(ARRAY_SIZE(particle_array), 256), 256>>>(particle_array, params.ps, params.width, params.height,
		transitionMatrix, process_noise);

	CHECK_ERROR(cudaGetLastError());
}