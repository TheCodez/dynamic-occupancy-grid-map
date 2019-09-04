#include "occupancy_grid_map.h"
#include "cuda_utils.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/random.h>
#include <Eigen/Dense>

#include "cuda_runtime.h"

OccupancyGridMap::OccupancyGridMap(const GridParams& params)
	: params(params)
{
	int c = (params.width / params.resolution) * (params.height / params.resolution);

	cudaMallocManaged((void**)&grid_cell_array, c * sizeof(GridCell));
	cudaMallocManaged((void**)&particle_array, params.v * sizeof(Particle));
	cudaMallocManaged(&weight_array, params.v * sizeof(float));
	cudaMallocManaged(&birth_weight_array, params.v * sizeof(float));
	cudaMallocManaged(&meas_cell_array, c * sizeof(MeasurementCell));

	cudaMalloc(&born_masses_array, c * sizeof(float));
	cudaMalloc(&vel_x_array, params.v * sizeof(float));
	cudaMalloc(&vel_y_array, params.v * sizeof(float));
	cudaMalloc(&vel_x_squared_array, params.v * sizeof(float));
	cudaMalloc(&vel_y_squared_array, params.v * sizeof(float));
	cudaMalloc(&vel_xy_array, params.v * sizeof(float));
	cudaMalloc(&rand_array, params.v * sizeof(float));

	initialize();
}

OccupancyGridMap::~OccupancyGridMap()
{
	cudaFree(grid_cell_array);
	cudaFree(particle_array);
	cudaFree(weight_array);
	cudaFree(birth_weight_array);
	cudaFree(meas_cell_array);

	cudaFree(born_masses_array);
	cudaFree(vel_x_array);
	cudaFree(vel_y_array);
	cudaFree(vel_x_squared_array);
	cudaFree(vel_y_squared_array);
	cudaFree(vel_xy_array);
	cudaFree(rand_array);
}

void OccupancyGridMap::update(float dt)
{
	particlePrediction(dt);
	particleAssignment();
	gridCellOccupancyUpdate();
	updatePersistentParticles();
	initializeNewParticles();
	statisticalMoments();
	resampling();
}
