#include "kernel/statistical_moments.h"
#include "common.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float calc_mean(float* vel_array_accum, int start_idx, int end_idx, float rhoP)
{
	float velAccum = subtract(vel_array_accum, start_idx, end_idx);
	return (1.0f / rhoP) * velAccum;
}

__device__ float calc_variance(float* vel_squared_array_accum, int start_idx, int end_idx, float rhoP, float mean_vel)
{
	float velAccum = subtract(vel_squared_array_accum, start_idx, end_idx);
	return (1.0f / rhoP) * velAccum - mean_vel * mean_vel;
}

__device__ float calc_covariance(float* vel_xy_array_accum, int start_idx, int end_idx, float rhoP, float mean_x_vel, float mean_y_vel)
{
	float velAccum = subtract(vel_xy_array_accum, start_idx, end_idx);
	return (1.0f / rhoP) * velAccum - mean_x_vel * mean_y_vel;
}

__device__ void store(GridCell* grid_cell_array, int j, float mean_x_vel, float mean_y_vel, float var_x_vel, float var_y_vel,
	float covar_xy_vel)
{
	grid_cell_array[j].mean_x_vel = mean_x_vel;
	grid_cell_array[j].mean_y_vel = mean_y_vel;
	grid_cell_array[j].var_x_vel = var_x_vel;
	grid_cell_array[j].var_y_vel = var_y_vel;
	grid_cell_array[j].covar_xy_vel = covar_xy_vel;
}

__global__ void statisticalMomentsKernel1(Particle* particle_array, float* weight_array, float* vel_x_array, float* vel_y_array,
	float* vel_x_squared_array, float* vel_y_squared_array, float* vel_xy_array)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(particle_array); i += blockDim.x * gridDim.x)
	{
		float weight = weight_array[i];
		float vel_x = 0.0f;//particle_array[i].state(2);
		float vel_y = 0.0f;//particle_array[i].state(3);
		vel_x_array[i] = weight * vel_x;
		vel_y_array[i] = weight * vel_y;
		vel_x_squared_array[i] = weight * vel_x * vel_x;
		vel_y_squared_array[i] = weight * vel_y * vel_y;
		vel_xy_array[i] = weight * vel_x * vel_y;
	}
}

__global__ void statisticalMomentsKernel2(GridCell* grid_cell_array, float* vel_x_array_accum, float* vel_y_array_accum,
	float* vel_x_squared_array_accum, float* vel_y_squared_array_accum, float* vel_xy_array_accum)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(grid_cell_array); i += blockDim.x * gridDim.x)
	{
		float rho_p = grid_cell_array[i].pers_occ_mass;
		int start_idx = grid_cell_array[i].start_idx;
		int end_idx = grid_cell_array[i].end_idx;
		float mean_x_vel = calc_mean(vel_x_array_accum, start_idx, end_idx, rho_p);
		float mean_y_vel = calc_mean(vel_y_array_accum, start_idx, end_idx, rho_p);
		float var_x_vel = calc_variance(vel_x_squared_array_accum, start_idx, end_idx, rho_p, mean_x_vel);
		float var_y_vel = calc_variance(vel_y_squared_array_accum, start_idx, end_idx, rho_p, mean_y_vel);
		float covar_xy_vel = calc_covariance(vel_xy_array_accum, start_idx, end_idx, rho_p, mean_x_vel, mean_y_vel);
		store(grid_cell_array, i, mean_x_vel, mean_y_vel, var_x_vel, var_y_vel, covar_xy_vel);
	}
}
