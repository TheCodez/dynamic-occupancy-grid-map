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

__device__ int calc_start_idx(double* particle_orders_array_accum, int j)
{
	return j > 0 ? static_cast<int>(particle_orders_array_accum[j - 1]) : 0;
}

__device__ int calc_end_idx(double* particle_orders_array_accum, int j)
{
	return static_cast<int>(particle_orders_array_accum[j]) - 1;
}

__device__ int calc_num_assoc(int num_new_particles, double p_A)
{
	return static_cast<int>(num_new_particles * p_A);
}

__device__ double calc_weight_assoc(int nu_A, double p_A, double born_mass)
{
	return nu_A > 0 ? (p_A * born_mass) / nu_A : 0.0;
}

__device__ double calc_weight_unassoc(int nu_UA, double p_A, double born_mass)
{
	return nu_UA > 0 ? ((1.0 - p_A) * born_mass) / nu_UA : 0.0;
}

__device__ void store_weights(double w_A, double w_UA, GridCell* grid_cell_array, int j)
{
	grid_cell_array[j].w_A = w_A;
	grid_cell_array[j].w_UA = w_UA;
}

__device__ void initialize_new_particle(Particle* birth_particle_array, int i, GridCell* grid_cell_array, int grid_size)
{
	int cell_idx = birth_particle_array[i].grid_cell_idx;
	GridCell& grid_cell = grid_cell_array[cell_idx];

	unsigned int seed = hash(i);
	thrust::default_random_engine rng(seed);
	thrust::uniform_int_distribution<int> dist_idx(0, grid_size * grid_size);
	thrust::normal_distribution<double> dist_vel(0.0f, 4.0);

	bool associated = birth_particle_array[i].associated;
	if (associated)
	{
		double x = cell_idx % grid_size;
		double y = cell_idx / grid_size;

		birth_particle_array[i].weight = grid_cell.w_A;
		birth_particle_array[i].state = glm::vec4(x, y, dist_vel(rng), dist_vel(rng));
	}
	else
	{
		int index = dist_idx(rng);

		double x = index % grid_size;
		double y = index / grid_size;

		birth_particle_array[i].weight = grid_cell.w_UA;
		birth_particle_array[i].state = glm::vec4(x, y, dist_vel(rng), dist_vel(rng));
	}
}

void normalize_particle_orders(double* particle_orders_array_accum, int particle_orders_count, int v_B)
{
	thrust::device_ptr<double> particle_orders_accum(particle_orders_array_accum);

	float max = 1.0f;
	cudaMemcpy(&max, &particle_orders_array_accum[particle_orders_count - 1], sizeof(float), cudaMemcpyDeviceToHost);
	thrust::transform(particle_orders_accum, particle_orders_accum + particle_orders_count, particle_orders_accum, GPU_LAMBDA(double x)
	{
		return x * (v_B / max);
	});
}

__global__ void initNewParticlesKernel1(Particle* particle_array, GridCell* grid_cell_array, MeasurementCell* meas_cell_array,
	double* weight_array, double* born_masses_array, Particle* birth_particle_array, double* particle_orders_array_accum, int cell_count)
{
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < cell_count; j += blockDim.x * gridDim.x)
	{
		int start_idx = calc_start_idx(particle_orders_array_accum, j);
		int end_idx = calc_end_idx(particle_orders_array_accum, j);

		//printf("Start idx: %d, End idx: %d\n", start_idx, end_idx);

		int num_new_particles = start_idx <= end_idx ? end_idx - start_idx + 1 : 0;
		double p_A = meas_cell_array[j].p_A;
		int nu_A = calc_num_assoc(num_new_particles, p_A);
		int nu_UA = num_new_particles - nu_A;
		double w_A = calc_weight_assoc(nu_A, p_A, born_masses_array[j]);
		double w_UA = calc_weight_unassoc(nu_UA, p_A, born_masses_array[j]);
		store_weights(w_A, w_UA, grid_cell_array, j);

		//printf("w_A: %f, w_UA: %f\n", w_A, w_UA);

		for (int i = start_idx; i < start_idx + nu_A + 1; i++)
		{
			set_cell_idx_A(birth_particle_array, i, j);
		}

		for (int i = start_idx + nu_A + 1; i < end_idx + 1; i++)
		{
			set_cell_idx_UA(birth_particle_array, i, j);
		}
	}
}

__global__ void initNewParticlesKernel2(Particle* birth_particle_array, GridCell* grid_cell_array, int grid_size, int particle_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
	{
		initialize_new_particle(birth_particle_array, i, grid_cell_array, grid_size);
	}
}

__global__ void copyBirthWeightKernel(Particle* birth_particle_array, double* birth_weight_array, int particle_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
	{
		birth_weight_array[i] = birth_particle_array[i].weight;
	}
}
