#include "kernel/init_new_particles.h"
#include "common.h"
#include "cuda_utils.h"

#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ void set_cell_idx_A(Particle* birth_particle_array, int i, int grid_cell_idx)
{
	birth_particle_array[i].grid_cell_idx = grid_cell_idx;
	birth_particle_array[i].associated = true;
}

__device__ void set_cell_idx_UA(Particle* birth_particle_array, int i, int grid_cell_idx)
{
	birth_particle_array[i].grid_cell_idx = grid_cell_idx;
	birth_particle_array[i].associated = false;
}

__device__ int calc_start_idx(float* particle_orders_array_accum, int j)
{
	return j > 0 ? static_cast<int>(particle_orders_array_accum[j - 1]) : 0;
}

__device__ int calc_end_idx(float* particle_orders_array_accum, int j)
{
	return static_cast<int>(particle_orders_array_accum[j]) + 1;
}

__device__ int calc_num_assoc(int num_new_particles, float p_A)
{
	return static_cast<int>(num_new_particles * p_A);
}

__device__ float calc_weight_assoc(int nuA, float p_A, float born_mass)
{
	return (p_A * born_mass) / nuA;
}

__device__ float calc_weight_unassoc(int nuUA, float p_A, float born_mass)
{
	return ((1.0f - p_A) * born_mass) / nuUA;
}

__device__ void store_weights(float w_A, float w_UA, GridCell* grid_cell_array, int j)
{
	grid_cell_array[j].w_A = w_A;
	grid_cell_array[j].w_UA = w_UA;
}

__device__ void initialize_new_particle(Particle* birth_particle_array, int i, GridCell* grid_cell_array, int width)
{
	int cell_idx = birth_particle_array[i].grid_cell_idx;
	GridCell& grid_cell = grid_cell_array[cell_idx];

	thrust::default_random_engine rng;
	thrust::normal_distribution<float> dist_vel(0.0f, 4.0f);

	float x = cell_idx % width + 0.5f;
	float y = cell_idx / width + 0.5f;

	bool associated = birth_particle_array[i].associated;
	birth_particle_array[i].weight = associated ? grid_cell.w_A : grid_cell.w_UA;
	birth_particle_array[i].state = glm::vec4(x, y, dist_vel(rng), dist_vel(rng));
}

__host__ void normalize_particle_orders(float* particle_orders_array_accum, int particle_orders_count, int v_B)
{
	thrust::device_ptr<float> particle_orders(particle_orders_array_accum);

	thrust::transform(particle_orders, particle_orders + particle_orders_count, particle_orders, GPU_LAMBDA(float x)
	{
		float max = particle_orders_array_accum[particle_orders_count - 1];
		return x * (v_B / (1.0f * max));
	});
}

__global__ void initNewParticlesKernel1(Particle* particle_array, GridCell* grid_cell_array, MeasurementCell* meas_cell_array,
	float* weight_array, float* born_masses_array, Particle* birth_particle_array, float* particle_orders_array_accum, int cell_count)
{
	const int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < cell_count)
	{
		int start_idx = calc_start_idx(particle_orders_array_accum, j);
		int end_idx = calc_end_idx(particle_orders_array_accum, j);
		int num_new_particles = end_idx - start_idx + 1;
		float p_A = meas_cell_array[j].p_A;
		int nu_A = calc_num_assoc(num_new_particles, p_A);
		int nu_UA = num_new_particles - nu_A;
		float w_A = calc_weight_assoc(nu_A, p_A, born_masses_array[j]);
		float w_UA = calc_weight_unassoc(nu_UA, p_A, born_masses_array[j]);
		store_weights(w_A, w_UA, grid_cell_array, j);

		for (int i = start_idx; i < start_idx + nu_A + 1; i++)
		{
			set_cell_idx_A(birth_particle_array, i, j);
		}

		for (int i = start_idx + nu_A + 1; i < end_idx; i++)
		{
			set_cell_idx_UA(birth_particle_array, i, j);
		}
	}
}

__global__ void initNewParticlesKernel2(Particle* birth_particle_array, GridCell* grid_cell_array, float* birth_weight_array, int width,
	int particle_count)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < particle_count)
	{
		initialize_new_particle(birth_particle_array, i, grid_cell_array, width);
		birth_weight_array[i] = birth_particle_array[i].weight;
	}
}
