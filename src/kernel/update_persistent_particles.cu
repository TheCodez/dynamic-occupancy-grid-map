#include "kernel/update_persistent_particles.h"
#include "common.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float update_unnorm(Particle* particle_array, int i, MeasurementCell* meas_cell_array)
{
	Particle& particle = particle_array[i];
	return meas_cell_array[particle.grid_cell_idx].likelihood * particle.weight;
}

__device__ float calc_norm_assoc(float occ_accum, float rho_p)
{
	return occ_accum > 0.0f ? rho_p / occ_accum : 0.0f;
}

__device__ float calc_norm_unassoc(const GridCell& grid_cell)
{
	return grid_cell.occ_mass > 0.0f ? grid_cell.pers_occ_mass / grid_cell.occ_mass : 0.0f;
}

__device__ void set_normalization_components(GridCell* grid_cell_array, int i, float mu_A, float mu_UA)
{
	grid_cell_array[i].mu_A = mu_A;
	grid_cell_array[i].mu_UA = mu_UA;
}

__device__ float normalize(Particle& particle, GridCell* grid_cell_array, MeasurementCell* meas_cell_array, float weight)
{
	GridCell& cell = grid_cell_array[particle.grid_cell_idx];
	MeasurementCell& measCell = meas_cell_array[particle.grid_cell_idx];

	return measCell.p_A * cell.mu_A * weight + (1.0f - measCell.p_A) * cell.mu_UA * particle.weight;
}

__global__ void updatePersistentParticlesKernel1(Particle* particle_array, MeasurementCell* meas_cell_array, float* weight_array,
	int particle_count)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < particle_count)
	{
		weight_array[i] = update_unnorm(particle_array, i, meas_cell_array);
	}
}

__global__ void updatePersistentParticlesKernel2(GridCell* grid_cell_array, float* weight_array_accum, int cell_count)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < cell_count)
	{
		int start_idx = grid_cell_array[i].start_idx;
		int end_idx = grid_cell_array[i].end_idx;
		float m_occ_accum = subtract(weight_array_accum, start_idx, end_idx);
		float rho_p = grid_cell_array[i].pers_occ_mass;
		float mu_A = calc_norm_assoc(m_occ_accum, rho_p);
		float mu_UA = calc_norm_unassoc(grid_cell_array[i]);
		set_normalization_components(grid_cell_array, i, mu_A, mu_UA);
	}
}

__global__ void updatePersistentParticlesKernel3(Particle* particle_array, MeasurementCell* meas_cell_array, GridCell* grid_cell_array,
	float* weight_array, int particle_count)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < particle_count)
	{
		weight_array[i] = normalize(particle_array[i], grid_cell_array, meas_cell_array, weight_array[i]);
	}
}
