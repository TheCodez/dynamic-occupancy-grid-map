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
#include "kernel/update_persistent_particles.h"
#include "common.h"
#include "cuda_utils.h"
#include "dogm_types.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dogm
{

__device__ float calc_norm_assoc(float occ_accum, float rho_p)
{
	return occ_accum > 0.0 ? rho_p / occ_accum : 0.0;
}

__device__ float calc_norm_unassoc(const GridCell& grid_cell)
{
	return grid_cell.occ_mass > 0.0 ? grid_cell.pers_occ_mass / grid_cell.occ_mass : 0.0;
}

__device__ void set_normalization_components(GridCell* grid_cell_array, int i, float mu_A, float mu_UA)
{
	grid_cell_array[i].mu_A = mu_A;
	grid_cell_array[i].mu_UA = mu_UA;
}

__device__ float update_unnorm(Particle* particle_array, int i, MeasurementCell* meas_cell_array)
{
	Particle& particle = particle_array[i];
	return meas_cell_array[particle.grid_cell_idx].likelihood * particle.weight;
}

__device__ float normalize(Particle& particle, GridCell* grid_cell_array, MeasurementCell* meas_cell_array, float weight)
{
	GridCell& cell = grid_cell_array[particle.grid_cell_idx];
	MeasurementCell& meas_cell = meas_cell_array[particle.grid_cell_idx];

	return meas_cell.p_A * cell.mu_A * weight + (1.0 - meas_cell.p_A) * cell.mu_UA * particle.weight;
}

__global__ void updatePersistentParticlesKernel1(Particle* particle_array, MeasurementCell* meas_cell_array, float* weight_array,
	int particle_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
	{
		weight_array[i] = update_unnorm(particle_array, i, meas_cell_array);
	}
}

__global__ void updatePersistentParticlesKernel2(GridCell* grid_cell_array, float* weight_array_accum, int cell_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
	{
		int start_idx = grid_cell_array[i].start_idx;
		int end_idx = grid_cell_array[i].end_idx;

		if (start_idx != -1)
		{
			float m_occ_accum = subtract(weight_array_accum, start_idx, end_idx);
			float rho_p = grid_cell_array[i].pers_occ_mass;
			float mu_A = calc_norm_assoc(m_occ_accum, rho_p);
			float mu_UA = calc_norm_unassoc(grid_cell_array[i]);
			set_normalization_components(grid_cell_array, i, mu_A, mu_UA);
			//printf("mu_A: %f, mu_UA: %f\n", mu_A, mu_UA);
		}
	}
}

__global__ void updatePersistentParticlesKernel3(Particle* particle_array, MeasurementCell* meas_cell_array, GridCell* grid_cell_array,
	float* weight_array, int particle_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
	{
		weight_array[i] = normalize(particle_array[i], grid_cell_array, meas_cell_array, weight_array[i]);
	}
}

} /* namespace dogm */
