#pragma once

#include "cuda_runtime.h"

struct Particle;
struct GridCell;

template <typename T>
__device__ __host__ thrust::device_vector<T> accumulate(T* arr);

__device__ __host__ void normalize_particle_orders(float* particle_orders_array_accum, int νB);

__global__ void predictKernel(Particle* particle_array, float pS, const Eigen::Matrix4f transitionMatrix, const Eigen::Vector4f zeta);

__global__ void particleToGridKernel(Particle* particle_array, GridCell* grid_cell_array, float* weight_array);

__global__ void gridCellPredictionUpdateKernel(GridCell* grid_cell_array, float* weight_array_accum, MeasurementCell* meas_cell_array,
	float* born_masses_array, float pB);

__global__ void updatePersistentParticlesKernel1(Particle* particle_array, MeasurementCell* meas_cell_array, float* weight_array);
__global__ void updatePersistentParticlesKernel2(GridCell* grid_cell_array, float* weight_array_accum);
__global__ void updatePersistentParticlesKernel3(Particle* particle_array, GridCell* grid_cell_array, float* weight_array);

__global__ void initNewParticlesKernel1(Particle* particle_array, GridCell* grid_cell_array, MeasurementCell* meas_cell_array,
	float* weight_array, float* born_masses_array, Particle* birth_particle_array, float* particle_orders_array_accum);
__global__ void initNewParticlesKernel2(Particle* birth_particle_array, GridCell* grid_cell_array);

__global__ void statisticalMomentsKernel1(Particle* particle_array, float* weight_array, float* vel_x_array, float* vel_y_array,
	float* vel_x_squared_array, float* vel_y_squared_array, float* vel_xy_array);
__global__ void statisticalMomentsKernel2(GridCell* grid_cell_array, float* vel_x_array_accum, float* vel_y_array_accum,
	float* vel_x_squared_array_accum, float* vel_y_squared_array_accum, float* vel_xy_array_accum);


__global__ void resamplingKernel(Particle* particle_array, Particle* particle_array_next, Particle* birth_particle_array, float* rand_array,
	int* idx_array_resampled);
