#include "occupancy_grid_map.h"
#include "cuda_utils.h"
#include "common.h"

#include "kernel/measurement_grid.h"
#include "kernel/init.h"
#include "kernel/predict.h"
#include "kernel/particle_to_grid.h"
#include "kernel/mass_update.h"
#include "kernel/init_new_particles.h"
#include "kernel/update_persistent_particles.h"
#include "kernel/statistical_moments.h"
#include "kernel/resampling.h"

#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>

OccupancyGridMap::OccupancyGridMap(const GridParams& params)
	: params(params)
{
	const int grid_size = static_cast<int>(params.width / params.resolution) * static_cast<int>(params.height / params.resolution);

	CHECK_ERROR(cudaMallocManaged((void**)&grid_cell_array, grid_size * sizeof(GridCell)));
	CHECK_ERROR(cudaMallocManaged((void**)&particle_array, params.particle_count * sizeof(Particle)));
	CHECK_ERROR(cudaMalloc((void**)&meas_cell_array, grid_size * sizeof(MeasurementCell)));
	
	CHECK_ERROR(cudaMalloc(&birth_weight_array, params.particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&weight_array, params.particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&born_masses_array, grid_size * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&vel_x_array, params.particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&vel_y_array, params.particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&vel_x_squared_array, params.particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&vel_y_squared_array, params.particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&vel_xy_array, params.particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&rand_array, params.particle_count * sizeof(float)));

	initialize();
}

OccupancyGridMap::~OccupancyGridMap()
{
	CHECK_ERROR(cudaFree(grid_cell_array));
	CHECK_ERROR(cudaFree(particle_array));
	CHECK_ERROR(cudaFree(weight_array));
	CHECK_ERROR(cudaFree(birth_weight_array));
	CHECK_ERROR(cudaFree(meas_cell_array));

	CHECK_ERROR(cudaFree(born_masses_array));
	CHECK_ERROR(cudaFree(vel_x_array));
	CHECK_ERROR(cudaFree(vel_y_array));
	CHECK_ERROR(cudaFree(vel_x_squared_array));
	CHECK_ERROR(cudaFree(vel_y_squared_array));
	CHECK_ERROR(cudaFree(vel_xy_array));
	CHECK_ERROR(cudaFree(rand_array));
}

void OccupancyGridMap::initialize()
{
	initParticlesKernel<<<divUp(ARRAY_SIZE(particle_array), 256), 256>>>(particle_array, params.width, params.height, 
		ARRAY_SIZE(particle_array));

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::update(float dt, float* measurements)
{
	updateMeasurementGrid(measurements);

	particlePrediction(dt);
	particleAssignment();
	gridCellOccupancyUpdate();
	updatePersistentParticles();
	initializeNewParticles();
	statisticalMoments();
	resampling();
}

void OccupancyGridMap::updateMeasurementGrid(float* measurements)
{
	float* d_measurements;
	CHECK_ERROR(cudaMalloc(&d_measurements, sizeof(measurements)));
	CHECK_ERROR(cudaMemcpy(meas_array, measurements, sizeof(measurements), cudaMemcpyHostToDevice));

	const float resolution = 0.2f;
	const float min_range = 0.1f;
	const float max_range = 50.0f;
	const int polar_width = ARRAY_SIZE(measurements);
	const int polar_height = static_cast<int>(max_range / resolution);

	float2* polar_img;
	CHECK_ERROR(cudaMalloc(&polar_img, polar_width * polar_height * sizeof(float2)));
	
	dim3 block_dim(32, 32);
	dim3 grid_dim(divUp(polar_width, block_dim.x), divUp(polar_height, block_dim.y));
	
	createPolarGridMapKernel<<<dimGrid, dimBlock>>>(polar_img, d_measurements, polar_width, polar_height, resolution,
		min_range, max_range);
	
	CHECK_ERROR(cudaGetLastError());
	
	int width = static_cast<int>(params.width / params.resolution);
	int height = static_cast<int>(params.height / params.resolution);
	dim3 cart_grid_dim(divUp(width, block_dim.x), divUp(500, block_dim.y));
	polarToCartesianGridMapKernel<<<cart_dimGrid, dimBlock>>>(meas_cell_array, polar_img, width, height, polar_width, polar_height);
	
	CHECK_ERROR(cudaGetLastError());

	CHECK_ERROR(cudaFree(d_measurements));
	CHECK_ERROR(cudaFree(polar_img));
}

void OccupancyGridMap::particlePrediction(float dt)
{
	glm::mat4x4 transition_matrix(1, 0, dt, 0, 
								  0, 1, 0, dt, 
								  0, 0, 1, 0, 
								  0, 0, 0, 1);

	thrust::default_random_engine rng;
	thrust::normal_distribution<float> dist_pos(0.0f, params.process_noise_position);
	thrust::normal_distribution<float> dist_vel(0.0f, params.process_noise_velocity);

	glm::vec4 process_noise(dist_pos(rng), dist_pos(rng), dist_vel(rng), dist_vel(rng));

	int width = static_cast<int>(params.width / params.resolution);
	int height = static_cast<int>(params.height / params.resolution);
	predictKernel<<<divUp(ARRAY_SIZE(particle_array), 256), 256>>>(particle_array, width, height, params.ps,
		transition_matrix, process_noise);

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::particleAssignment()
{
	CHECK_ERROR(cudaDeviceSynchronize());
	thrust::device_ptr<Particle> particles(particle_array);
	thrust::sort(particles, particles + ARRAY_SIZE(particle_array), GPU_LAMBDA(Particle x, Particle y)
	{
		return x.grid_cell_idx < y.grid_cell_idx;
	});

	particleToGridKernel<<<divUp(ARRAY_SIZE(particle_array), 256), 256>>>(particle_array, grid_cell_array, weight_array);

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::gridCellOccupancyUpdate()
{
	thrust::device_vector<float> weightsAccum = accumulate(weight_array);
	float* weight_array_accum = thrust::raw_pointer_cast(weightsAccum.data());

	gridCellPredictionUpdateKernel<<<divUp(ARRAY_SIZE(grid_cell_array), 256), 256>>>(grid_cell_array, weight_array_accum,
		meas_cell_array, born_masses_array, params.pb);

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::updatePersistentParticles()
{
	updatePersistentParticlesKernel1<<<divUp(ARRAY_SIZE(particle_array), 256), 256>>>(particle_array, meas_cell_array, weight_array);

	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());

	thrust::device_vector<float> weightsAccum = accumulate(weight_array);
	float* weight_array_accum = thrust::raw_pointer_cast(weightsAccum.data());

	updatePersistentParticlesKernel2<<<divUp(ARRAY_SIZE(grid_cell_array), 256), 256>>>(grid_cell_array, weight_array_accum);

	CHECK_ERROR(cudaGetLastError());

	updatePersistentParticlesKernel3<<<divUp(ARRAY_SIZE(particle_array), 256), 256>>>(particle_array, meas_cell_array,
		grid_cell_array, weight_array);

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::initializeNewParticles()
{
	thrust::device_vector<float> particleOrdersAccum = accumulate(born_masses_array);
	float* particle_orders_array_accum = thrust::raw_pointer_cast(particleOrdersAccum.data());
	normalize_particle_orders(particle_orders_array_accum, params.new_born_particle_count);

	initNewParticlesKernel1<<<divUp(ARRAY_SIZE(particle_array), 256), 256>>>(particle_array, grid_cell_array, meas_cell_array,
		weight_array, born_masses_array, birth_particle_array, particle_orders_array_accum);

	CHECK_ERROR(cudaGetLastError());

	initNewParticlesKernel2<<<divUp(ARRAY_SIZE(birth_particle_array), 256), 256>>>(birth_particle_array, grid_cell_array,
		birth_weight_array, static_cast<int>(params.width / params.resolution));

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::statisticalMoments()
{
	statisticalMomentsKernel1<<<divUp(ARRAY_SIZE(particle_array), 256), 256>>>(particle_array, weight_array, vel_x_array,
		vel_y_array, vel_x_squared_array, vel_y_squared_array, vel_xy_array);

	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());

	thrust::device_vector<float> velXAccum = accumulate(vel_x_array);
	thrust::device_vector<float> velYAccum = accumulate(vel_y_array);
	thrust::device_vector<float> velXSquaredAccum = accumulate(vel_x_squared_array);
	thrust::device_vector<float> velYSquaredAccum = accumulate(vel_y_squared_array);
	thrust::device_vector<float> velXYAccum = accumulate(vel_xy_array);

	float* vel_x_array_accum = thrust::raw_pointer_cast(velXAccum.data());
	float* vel_y_array_accum = thrust::raw_pointer_cast(velYAccum.data());
	float* vel_x_squared_array_accum = thrust::raw_pointer_cast(velXSquaredAccum.data());
	float* vel_y_squared_array_accum = thrust::raw_pointer_cast(velYSquaredAccum.data());
	float* vel_xy_array_accum = thrust::raw_pointer_cast(velXYAccum.data());

	statisticalMomentsKernel2<<<divUp(ARRAY_SIZE(grid_cell_array), 256), 256>>>(grid_cell_array, vel_x_array_accum, vel_y_array_accum,
		vel_x_squared_array_accum, vel_y_squared_array_accum, vel_xy_array_accum);

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::resampling()
{
	resamplingKernel<<<divUp(ARRAY_SIZE(particle_array), 256), 256>>>(particle_array, particle_array/*_next*/, birth_particle_array,
		rand_array, nullptr/*idx_array_resampled*/);

	CHECK_ERROR(cudaGetLastError());
}
