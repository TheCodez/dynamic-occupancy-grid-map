#include "OccupancyGridMap.h"
#include "OccupancyGridKernel.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <Eigen/Dense>

#include "cuda_runtime.h"

#define ARRAY_SIZE(x) sizeof(x) / sizeof(x[0])

OccupancyGridMap::OccupancyGridMap(const GridParams& params)
	: params(params)
{
	int c = (params.width / params.resolution) * (params.height / params.resolution);

	cudaMallocManaged((void**)& grid_cell_array, c * sizeof(GridCell));
	cudaMallocManaged((void**)& particle_array, params.v * sizeof(Particle));
	cudaMallocManaged(&weight_array, params.v * sizeof(float));
	cudaMallocManaged(&meas_cell_array, c * sizeof(MeasurementCell));

	cudaMallocManaged(&born_masses_array, c * sizeof(float));
	cudaMallocManaged(&vel_x_array, params.v * sizeof(float));
	cudaMallocManaged(&vel_y_array, params.v * sizeof(float));
	cudaMallocManaged(&vel_x_squared_array, params.v * sizeof(float));
	cudaMallocManaged(&vel_y_squared_array, params.v * sizeof(float));
	cudaMallocManaged(&vel_xy_array, params.v * sizeof(float));
	cudaMallocManaged(&rand_array, params.v * sizeof(float));
}

OccupancyGridMap::~OccupancyGridMap()
{
	cudaFree(grid_cell_array);
	cudaFree(particle_array);
	cudaFree(weight_array);
	cudaFree(meas_cell_array);

	cudaFree(born_masses_array);
	cudaFree(vel_x_array);
	cudaFree(vel_y_array);
	cudaFree(vel_x_squared_array);
	cudaFree(vel_y_squared_array);
	cudaFree(vel_xy_array);
	cudaFree(rand_array);
}

void OccupancyGridMap::update(float t)
{
	particlePrediction(t);

	particleAssignment();

	gridCellOccupancyUpdate();

	updatePersistentParticles();

	initializeNewParticles();

	statisticalMoments();

	resampling();
}

void OccupancyGridMap::particlePrediction(float t)
{
	Eigen::Matrix4f transitionMatrix;
	transitionMatrix << 1, 0, t, 0,
						0, 1, 0, t,
						0, 0, 1, 0,
						0, 0, 0, 1;

	Eigen::Vector4f zeta;
	
	predictKernel/*<<<(particles.size() + 256 - 1) / 256, 256>>>*/(particle_array, pS, transitionMatrix, zeta);
}

void OccupancyGridMap::particleAssignment()
{
	struct SortParticles
	{
		__host__ __device__ bool operator()(Particle x, Particle y)
		{
			return x.gridCellIdx < y.gridCellIdx;
		}
	};
	
	thrust::device_ptr<Particle> particles = thrust::device_pointer_cast(particle_array);
	thrust::sort(particles, particles + ARRAY_SIZE(particle_array), SortParticles());

	particleToGridKernel/*<<<(particlesSize + 256 - 1) / 256, 256>>>*/(particle_array, grid_cell_array, weight_array);
}

void OccupancyGridMap::gridCellOccupancyUpdate()
{
	thrust::device_vector<float> weightsAccum = accumulate(weight_array);
	float* weight_array_accum = thrust::raw_pointer_cast(&weightsAccum[0]);
	gridCellPredictionUpdateKernel/*<<<(gridSize + 256 - 1) / 256, 256>>>*/(grid_cell_array, weight_array_accum, meas_cell_array, born_masses_array, pB);
}

void OccupancyGridMap::updatePersistentParticles()
{
	updatePersistentParticlesKernel1/*<<<(particlesSize + 256 - 1) / 256, 256>>>*/(particle_array, meas_cell_array, weight_array);
	thrust::device_vector<float> weightsAccum = accumulate(weight_array);
	float* weight_array_accum = thrust::raw_pointer_cast(&weightsAccum[0]);

	updatePersistentParticlesKernel2/*<<<(gridSize + 256 - 1) / 256, 256>>>*/(grid_cell_array, weight_array_accum);

	updatePersistentParticlesKernel3/*<<<(particlesSize + 256 - 1) / 256, 256>>>*/(particle_array, grid_cell_array, weight_array);
}

void OccupancyGridMap::initializeNewParticles()
{
	thrust::device_vector<float> particleOrdersAccum = accumulate(born_masses_array);
	float* particle_orders_array_accum = thrust::raw_pointer_cast(&particleOrdersAccum[0]);
	normalize_particle_orders(particle_orders_array_accum, vB);
	initNewParticlesKernel1/*<<<(particlesSize + 256 - 1) / 256, 256>>>*/(particle_array, grid_cell_array, meas_cell_array, weight_array, born_masses_array, birth_particle_array,
		particle_orders_array_accum);
	initNewParticlesKernel2/*<<<(birtParticlesSize + 256 - 1) / 256, 256>>>*/(birth_particle_array, grid_cell_array);
}

void OccupancyGridMap::statisticalMoments()
{
	statisticalMomentsKernel1/*<<<(particlesSize + 256 - 1) / 256, 256>>>*/(particle_array, weight_array, vel_x_array, vel_y_array,
		vel_x_squared_array, vel_y_squared_array, vel_xy_array);

	thrust::device_vector<float> velXAccum = accumulate(vel_x_array);
	thrust::device_vector<float> velYAccum = accumulate(vel_y_array);
	thrust::device_vector<float> velXSquaredAccum = accumulate(vel_x_squared_array);
	thrust::device_vector<float> velYSquaredAccum = accumulate(vel_y_squared_array);
	thrust::device_vector<float> velXYAccum = accumulate(vel_xy_array);

	float* vel_x_array_accum = thrust::raw_pointer_cast(&velXAccum[0]);
	float* vel_y_array_accum = thrust::raw_pointer_cast(&velYAccum[0]);
	float* vel_x_squared_array_accum = thrust::raw_pointer_cast(&velXSquaredAccum[0]);
	float* vel_y_squared_array_accum = thrust::raw_pointer_cast(&velYSquaredAccum[0]);
	float* vel_xy_array_accum = thrust::raw_pointer_cast(&velXYAccum[0]);

	statisticalMomentsKernel2/*<<<(gridSize + 256 - 1) / 256, 256>>>*/(grid_cell_array, vel_x_array_accum, vel_y_array_accum, 
		vel_x_squared_array_accum, vel_y_squared_array_accum, vel_xy_array_accum);
}

void OccupancyGridMap::resampling()
{
	resamplingKernel/*<<<(particlesSize + 256 - 1) / 256, 256>>>*/(particle_array, particle_array/*_next*/, birth_particle_array,
		rand_array, nullptr/*idx_array_resampled*/);
}
