#include "OccupancyGridKernel.h"
#include "OccupancyGridMap.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include <thrust/transform.h>
#include <thrust/functional.h>

#define ARRAY_SIZE(x) sizeof(x) / sizeof(x[0])

__device__ float separate_newborn_part(float occPred, float occUp, float pB) 
{
	return (occUp * pB * (1.0f - occPred)) / (occPred + pB * (1.0f - occPred));
}

__device__ bool is_first_particle(Particle* particle_array, int i) 
{
	return i == 0 || particle_array[i].gridCellIdx != particle_array[i - 1].gridCellIdx;
}

__device__ bool is_last_particle(Particle* particle_array, int i) 
{
	return i == ARRAY_SIZE(particle_array) - 1 || particle_array[i].gridCellIdx != particle_array[i + 1].gridCellIdx;
}

__device__ float subtract(float* accum_array, int startIdx, int endIdx)
{
	return accum_array[endIdx] - accum_array[startIdx];
}

__device__ float predict_free_mass(GridCell& grid_cell, float occPred, float alpha = 0.9f)
{
	return std::min(alpha * grid_cell.freeMass, 1.0f - occPred);
}

__device__ float update_o(float occPred, float freePred, const MeasurementCell& meas)
{
	return (occPred * meas.occMass) / (2 * occPred * meas.occMass - occPred - meas.occMass + 1);
}

__device__ float update_f(float occPred, float freePred, const MeasurementCell& meas)
{
	return (freePred * meas.freeMass) / (2 * freePred * meas.freeMass - freePred - meas.freeMass + 1);
}

__device__ void store_values(float rhoB, float rhoP, float freeUp, float occUp, GridCell* grid_cell_array, int i)
{
	grid_cell_array[i].persOccMass = rhoP;
	grid_cell_array[i].newBornOccMass = rhoB;
	grid_cell_array[i].freeMass = freeUp;
}

__device__ float update_unnorm(Particle* particle_array, int i, MeasurementCell* meas_cell_array)
{
	Particle& particle = particle_array[i];
	return meas_cell_array[particle.gridCellIdx].likelihood * particle.weight;
}

__device__ float calc_norm_assoc(float occAccum, float rhoP)
{
	return rhoP / occAccum;
}

__device__ float calc_norm_unassoc(const GridCell& gridCell)
{
	return gridCell.persOccMass / gridCell.occMass;
}

__device__ void set_normalization_components(GridCell* grid_cell_array, int i, float muA, float muUA)
{
	grid_cell_array[i].muA = muA;
	grid_cell_array[i].muUA = muUA;
}

__device__ void set_cell_idx_A(Particle* birth_particle_array, int i, int grid_cell_idx)
{
	birth_particle_array[i].gridCellIdx = grid_cell_idx;
	birth_particle_array[i].associated = true;
}

__device__ void set_cell_idx_UA(Particle* birth_particle_array, int i, int grid_cell_idx)
{
	birth_particle_array[i].gridCellIdx = grid_cell_idx;
	birth_particle_array[i].associated = false;
}

__device__ float normalize(Particle& particle, GridCell* grid_cell_array, MeasurementCell* meas_cell_array, float weight)
{
	GridCell& cell = grid_cell_array[particle.gridCellIdx];
	MeasurementCell& measCell = meas_cell_array[particle.gridCellIdx];

	return measCell.pA * cell.muA * weight + (1.0f - measCell.pA) * cell.muUA * particle.weight;
}

__device__ void predict(Particle* particle_array, int i, float pS, const Eigen::Matrix4f transitionMatrix, const Eigen::Vector4f zeta,
	int width, int height)
{
	particle_array[i].state = transitionMatrix * particle_array[i].state + zeta;
	particle_array[i].weight = pS * particle_array[i].weight;

	float x = particle_array[i].state[0];
	float y = particle_array[i].state[1];

	if ((x > width - 1 || x < 0)
		|| (y > height - 1 || y < 0))
	{
		// TODO: resolution?
		int size = width * height;

		thrust::default_random_engine rng;
		thrust::uniform_int_distribution<int> distIdx(0, width * height);

		int index = distIdx(rng);

		int x = index % width + 0.5f;
		int y = index / width + 0.5f;
	}

	x = std::max(std::min((int)x, width - 1), 0);
	y = std::max(std::min((int)y, height - 1), 0);
	particle_array[i].gridCellIdx = x + width * y;
}

template <typename T>
__device__ __host__ thrust::device_vector<T> accumulate(T* arr)
{
	thrust::device_ptr<T> ptr = thrust::device_pointer_cast(arr);
	thrust::device_vector<T> result;
	thrust::inclusive_scan(ptr, ptr + ARRAY_SIZE(array), result.begin());

	return result;
}

__device__ int calc_start_idx(float* particle_orders_array_accum, int j)
{
	return j > 0 ? (int)particle_orders_array_accum[j - 1] : 0;
}

__device__ int calc_end_idx(float* particle_orders_array_accum, int j)
{
	return (int)particle_orders_array_accum[j] + 1;
}

__device__ int calc_num_assoc(int num_new_particles, float pA)
{
	return (int)num_new_particles * pA;
}

__device__ float calc_weight_assoc(int nuA, float pA, float born_mass)
{
	return (pA * born_mass) / nuA;
}

__device__ float calc_weight_unassoc(int nuUA, float pA, float born_mass)
{
	return ((1.0f - pA) * born_mass) / nuUA;
}

__device__ void store_weights(float wA, float wUA, GridCell* grid_cell_array, int j)
{
	grid_cell_array[j].wA = wA;
	grid_cell_array[j].wUA = wUA;
}

__device__ void initialize_new_particle(Particle* birth_particle_array, int i, GridCell* grid_cell_array, int width)
{
	int cellIdx = birth_particle_array[i].gridCellIdx;
	GridCell& gridCell = grid_cell_array[cellIdx];

	thrust::default_random_engine rng;
	thrust::normal_distribution<float> distVel(0.0f, 4.0f);

	float x = cellIdx % width + 0.5f;
	float y = cellIdx / width + 0.5f;

	bool associated = birth_particle_array[i].associated;
	birth_particle_array[i].weight = associated ? gridCell.wA : gridCell.wUA;
	birth_particle_array[i].state << x, y, distVel(rng), distVel(rng);
}

__device__ float calc_mean(float* vel_array_accum, int startIdx, int endIdx, float rhoP)
{
	float velAccum = subtract(vel_array_accum, startIdx, endIdx);
	return (1.0f / rhoP) * velAccum;
}

__device__ float calc_variance(float* vel_squared_array_accum, int startIdx, int endIdx, float rhoP, float mean_vel)
{
	float velAccum = subtract(vel_squared_array_accum, startIdx, endIdx);
	return (1.0f / rhoP) * velAccum - mean_vel * mean_vel;
}

__device__ float calc_covariance(float* vel_xy_array_accum, int startIdx, int endIdx, float rhoP, float mean_x_vel, float mean_y_vel)
{
	float velAccum = subtract(vel_xy_array_accum, startIdx, endIdx);
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

__global__ void initializationKernel(Particle* particle_array, int width, int height, int size)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(particle_array); i += blockDim.x * gridDim.x)
	{
		int size = width * height;

		thrust::default_random_engine rng;
		thrust::uniform_int_distribution<int> distIdx(0, size);
		thrust::normal_distribution<float> distVel(0.0f, 4.0f);

		int index = distIdx(rng);

		float x = index % width + 0.5f;
		float y = index / width + 0.5f;

		particle_array[i].weight = 1.0f / size;
		particle_array[i].state << x, y, distVel(rng), distVel(rng);
	}
}

__global__ void predictKernel(Particle* particle_array, int width, int height, float pS, const Eigen::Matrix4f transitionMatrix,
	const Eigen::Vector4f zeta)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(particle_array); i += blockDim.x * gridDim.x)
	{
		predict(particle_array, i, pS, transitionMatrix, zeta, width, height);
	}
}

__device__ __host__ void normalize_particle_orders(float* particle_orders_array_accum, int size)
{
	struct Normalize : public thrust::unary_function<float, float>
	{
		float max;
		int vB;

		Normalize(float max, int vB) : max(max), vB(vB) {}

		__host__ __device__ float operator()(float x)
		{
			return x * (vB / (1.0f * max));
		}
	};

	int arraySize = ARRAY_SIZE(particle_orders_array_accum);
	float max = particle_orders_array_accum[arraySize - 1];
	thrust::device_ptr<Particle> particleOrders = thrust::device_pointer_cast(particle_orders_array_accum);
	thrust::transform(particleOrders, particleOrders + ARRAY_SIZE(particle_orders_array_accum), particleOrders, Normalize(max, size));
}

__device__ Particle copy_particle(Particle* particle_array, Particle* birth_particle_array, int idx)
{
	if (idx < ARRAY_SIZE(particle_array))
	{
		return particle_array[idx];
	}
	else
	{
		return birth_particle_array[idx - ARRAY_SIZE(particle_array)];
	}
}

__global__ void particleToGridKernel(Particle* particle_array, GridCell* grid_cell_array, float* weight_array)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(particle_array); i += blockDim.x * gridDim.x)
	{
		int j = particle_array[i].gridCellIdx;

		if (is_first_particle(particle_array, i)) 
		{
			grid_cell_array[j].startIdx = i;
		}
		if (is_last_particle(particle_array, i)) 
		{
			grid_cell_array[j].endIdx = i;
		}

		weight_array[i] = particle_array[i].weight;
	}
}

__global__ void gridCellPredictionUpdateKernel(GridCell* grid_cell_array, float* weight_array_accum, MeasurementCell* meas_cell_array,
	float* born_masses_array, float pB)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(grid_cell_array); i += blockDim.x * gridDim.x)
	{
		int startIdx = grid_cell_array[i].startIdx;
		int endIdx = grid_cell_array[i].endIdx;
		float occPred = subtract(weight_array_accum, startIdx, endIdx);
		float freePred = predict_free_mass(grid_cell_array[i], occPred);
		float occUp = update_o(occPred, freePred, meas_cell_array[i]);
		float freeUp = update_f(occPred, freePred, meas_cell_array[i]);
		float rhoB = separate_newborn_part(occPred, occUp, pB);
		float rhoP = occUp - rhoB;
		born_masses_array[i] = rhoB;
		store_values(rhoB, rhoP, freeUp, occUp, grid_cell_array, i);
	}
}

__global__ void updatePersistentParticlesKernel1(Particle* particle_array, MeasurementCell* meas_cell_array, float* weight_array)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(particle_array); i += blockDim.x * gridDim.x)
	{
		weight_array[i] = update_unnorm(particle_array, i, meas_cell_array);
	}
}

__global__ void updatePersistentParticlesKernel2(GridCell* grid_cell_array, float* weight_array_accum)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(grid_cell_array); i += blockDim.x * gridDim.x)
	{
		int startIdx = grid_cell_array[i].startIdx;
		int endIdx = grid_cell_array[i].endIdx;
		float occAccum = subtract(weight_array_accum, startIdx, endIdx);
		float rhoP = grid_cell_array[i].persOccMass;
		float muA = calc_norm_assoc(occAccum, rhoP);
		float muUA = calc_norm_unassoc(grid_cell_array[i]);
		set_normalization_components(grid_cell_array, i, muA, muUA);
	}
}

__global__ void updatePersistentParticlesKernel3(Particle* particle_array, MeasurementCell* meas_cell_array, GridCell* grid_cell_array,
	float* weight_array)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(particle_array); i += blockDim.x * gridDim.x)
	{
		weight_array[i] = normalize(particle_array[i], grid_cell_array, meas_cell_array, weight_array[i]);
	}
}

__global__ void initNewParticlesKernel1(Particle* particle_array, GridCell* grid_cell_array, MeasurementCell* meas_cell_array,
	float* weight_array, float* born_masses_array, Particle* birth_particle_array, float* particle_orders_array_accum)
{
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < ARRAY_SIZE(grid_cell_array); j += blockDim.x * gridDim.x)
	{
		int startIdx = calc_start_idx(particle_orders_array_accum, j);
		int endIdx = calc_end_idx(particle_orders_array_accum, j);
		int num_new_particles = endIdx - startIdx + 1;
		float pA = meas_cell_array[j].pA;
		int nuA = calc_num_assoc(num_new_particles, pA);
		int nuUA = num_new_particles - nuA;
		float wA = calc_weight_assoc(nuA, pA, born_masses_array[j]);
		float wUA = calc_weight_unassoc(nuUA, pA, born_masses_array[j]);
		store_weights(wA, wUA, grid_cell_array, j);

		for (int i = startIdx; i < startIdx + nuA + 1; i++)
		{
			set_cell_idx_A(birth_particle_array, i, j);
		}

		for (int i = startIdx + nuA + 1; i < endIdx; i++)
		{
			set_cell_idx_UA(birth_particle_array, i, j);
		}
	}
}

__global__ void initNewParticlesKernel2(Particle* birth_particle_array, GridCell* grid_cell_array, float* birth_weight_array, int width)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(birth_particle_array); i += blockDim.x * gridDim.x)
	{
		initialize_new_particle(birth_particle_array, i, grid_cell_array, width);
		birth_weight_array[i] = birth_particle_array[i].weight;
	}
}

__global__ void statisticalMomentsKernel1(Particle* particle_array, float* weight_array, float* vel_x_array, float* vel_y_array,
	float* vel_x_squared_array, float* vel_y_squared_array, float* vel_xy_array)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(particle_array); i += blockDim.x * gridDim.x)
	{
		float weight = weight_array[i];
		float vel_x = particle_array[i].state(2);
		float vel_y = particle_array[i].state(3);
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
		float rhoP = grid_cell_array[i].persOccMass;
		int startIdx = grid_cell_array[i].startIdx;
		int endIdx = grid_cell_array[i].endIdx;
		float mean_x_vel = calc_mean(vel_x_array_accum, startIdx, endIdx, rhoP);
		float mean_y_vel = calc_mean(vel_y_array_accum, startIdx, endIdx, rhoP);
		float var_x_vel = calc_variance(vel_x_squared_array_accum, startIdx, endIdx, rhoP, mean_x_vel);
		float var_y_vel = calc_variance(vel_y_squared_array_accum, startIdx, endIdx, rhoP, mean_y_vel);
		float covar_xy_vel = calc_covariance(vel_xy_array_accum, startIdx, endIdx, rhoP, mean_x_vel, mean_y_vel);
		store(grid_cell_array, i, mean_x_vel, mean_y_vel, var_x_vel, var_y_vel, covar_xy_vel);
	}
}

__global__ void resamplingKernel(Particle* particle_array, Particle* particle_array_next, Particle* birth_particle_array,
	float* rand_array, int* idx_array_resampled)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ARRAY_SIZE(particle_array); i += blockDim.x * gridDim.x)
	{
		particle_array_next[i] = copy_particle(particle_array, birth_particle_array, idx_array_resampled[i]);
	}
}
