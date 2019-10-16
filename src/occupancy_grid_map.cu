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

#include "opengl/renderer.h"
#include "opengl/texture.h"
#include "opengl/framebuffer.h"

#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <cuda_runtime.h>

int OccupancyGridMap::BLOCK_SIZE = 256;

OccupancyGridMap::OccupancyGridMap(const GridParams& params, const LaserSensorParams& laser_params)
	: params(params),
	  laser_params(laser_params),
	  grid_width(static_cast<int>(params.width / params.resolution)),
	  grid_height(static_cast<int>(params.height / params.resolution)),
	  particle_count(params.particle_count),
	  grid_cell_count(grid_width * grid_height),
	  new_born_particle_count(params.new_born_particle_count)
{
	CHECK_ERROR(cudaMallocManaged((void**)&grid_cell_array, grid_cell_count * sizeof(GridCell)));
	CHECK_ERROR(cudaMallocManaged((void**)&particle_array, particle_count * sizeof(Particle)));
	CHECK_ERROR(cudaMallocManaged((void**)&particle_array_next, particle_count * sizeof(Particle)));
	CHECK_ERROR(cudaMalloc((void**)&birth_particle_array, new_born_particle_count * sizeof(Particle)));

	CHECK_ERROR(cudaMallocManaged((void**)&meas_cell_array, grid_cell_count * sizeof(MeasurementCell)));

	CHECK_ERROR(cudaMalloc(&weight_array, particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&birth_weight_array, particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&born_masses_array, grid_cell_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&vel_x_array, particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&vel_y_array, particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&vel_x_squared_array, particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&vel_y_squared_array, particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&vel_xy_array, particle_count * sizeof(float)));
	CHECK_ERROR(cudaMalloc(&rand_array, particle_count * sizeof(float)));

	initialize();
}

OccupancyGridMap::~OccupancyGridMap()
{
	CHECK_ERROR(cudaFree(grid_cell_array));
	CHECK_ERROR(cudaFree(particle_array));
	CHECK_ERROR(cudaFree(particle_array_next));
	CHECK_ERROR(cudaFree(meas_cell_array));

	CHECK_ERROR(cudaFree(weight_array));
	CHECK_ERROR(cudaFree(birth_weight_array));
	CHECK_ERROR(cudaFree(born_masses_array));
	CHECK_ERROR(cudaFree(vel_x_array));
	CHECK_ERROR(cudaFree(vel_y_array));
	CHECK_ERROR(cudaFree(vel_x_squared_array));
	CHECK_ERROR(cudaFree(vel_y_squared_array));
	CHECK_ERROR(cudaFree(vel_xy_array));
	CHECK_ERROR(cudaFree(rand_array));
	
	delete renderer;
}

void OccupancyGridMap::initialize()
{
	initParticlesKernel<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, grid_width, grid_height,
		particle_count);

	initGridCellsKernel<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(grid_cell_array, grid_width, grid_height,
		grid_cell_count);

	CHECK_ERROR(cudaGetLastError());
	
	renderer = new Renderer(grid_width, grid_height, laser_params.fov);
}

void OccupancyGridMap::updateDynamicGrid(float dt)
{
	particlePrediction(dt);
	particleAssignment();
	gridCellOccupancyUpdate();
	updatePersistentParticles();
	initializeNewParticles();
	statisticalMoments();
	resampling();

	CHECK_ERROR(cudaMemcpy(particle_array, particle_array_next, particle_count * sizeof(Particle), cudaMemcpyDeviceToDevice));

	CHECK_ERROR(cudaDeviceSynchronize());
}

void OccupancyGridMap::updateMeasurementGrid(float* measurements, int num_measurements)
{
	std::cout << "OccupancyGridMap::updateMeasurementGrid" << std::endl;

	float* d_measurements;
	CHECK_ERROR(cudaMalloc(&d_measurements, num_measurements * sizeof(float)));
	CHECK_ERROR(cudaMemcpy(d_measurements, measurements, num_measurements * sizeof(float), cudaMemcpyHostToDevice));

	const int polar_width = num_measurements;
	const int polar_height = grid_height;

	dim3 block_dim(32, 32);
	dim3 grid_dim(divUp(polar_width, block_dim.x), divUp(polar_height, block_dim.y));
	dim3 cart_grid_dim(divUp(grid_width, block_dim.x), divUp(grid_height, block_dim.y));

	const float anisotropy_level = 16.0f;
	Texture texture(polar_width, polar_height, anisotropy_level);
	cudaSurfaceObject_t polar_surface;
	
	// create polar texture
	texture.beginCudaAccess(&polar_surface);
	createPolarGridTextureKernel<<<grid_dim, block_dim>>>(polar_surface, d_measurements, polar_width, polar_height, params.resolution);

	CHECK_ERROR(cudaGetLastError());
	texture.endCudaAccess(polar_surface);
	
	// render cartesian image to texture using polar texture
	renderer->renderToTexture(texture);
	
	Framebuffer* framebuffer = renderer->getFrameBuffer();
	cudaSurfaceObject_t cartesian_surface;

	framebuffer->beginCudaAccess(&cartesian_surface);
	// transform RGBA texture to measurement grid
	cartesianGridToMeasurementGridKernel<<<cart_grid_dim, block_dim>>>(meas_cell_array, cartesian_surface, grid_width, grid_height);

	CHECK_ERROR(cudaGetLastError());
	framebuffer->endCudaAccess(cartesian_surface);

	CHECK_ERROR(cudaFree(d_measurements));
	CHECK_ERROR(cudaDeviceSynchronize());
}

void OccupancyGridMap::particlePrediction(float dt)
{
	std::cout << "OccupancyGridMap::particlePrediction" << std::endl;

	glm::mat4x4 transition_matrix(1, 0, dt, 0, 
								  0, 1, 0, dt, 
								  0, 0, 1, 0, 
								  0, 0, 0, 1);

	thrust::default_random_engine rng;
	thrust::normal_distribution<float> dist_pos(0.0f, params.process_noise_position);
	thrust::normal_distribution<float> dist_vel(0.0f, params.process_noise_velocity);

	glm::vec4 process_noise(dist_pos(rng), dist_pos(rng), dist_vel(rng), dist_vel(rng));

	predictKernel<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, grid_width, grid_height, params.p_S,
		transition_matrix, process_noise, particle_count);

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::particleAssignment()
{
	std::cout << "OccupancyGridMap::particleAssignment" << std::endl;

	CHECK_ERROR(cudaDeviceSynchronize());
	thrust::device_ptr<Particle> particles(particle_array);
	thrust::sort(particles, particles + particle_count, GPU_LAMBDA(Particle x, Particle y)
	{
		return x.grid_cell_idx < y.grid_cell_idx;
	});

	particleToGridKernel<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, grid_cell_array, weight_array,
		particle_count);

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::gridCellOccupancyUpdate()
{
	std::cout << "OccupancyGridMap::gridCellOccupancyUpdate" << std::endl;

	CHECK_ERROR(cudaDeviceSynchronize());
	thrust::device_vector<float> weightsAccum(particle_count);
	accumulate(weight_array, weightsAccum);
	float* weight_array_accum = thrust::raw_pointer_cast(weightsAccum.data());

	gridCellPredictionUpdateKernel<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(grid_cell_array, particle_array, weight_array_accum,
		meas_cell_array, born_masses_array, params.p_B, params.p_S, grid_cell_count);

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::updatePersistentParticles()
{
	std::cout << "OccupancyGridMap::updatePersistentParticles" << std::endl;

	updatePersistentParticlesKernel1<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, meas_cell_array,
		weight_array, particle_count);

	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());

	thrust::device_vector<float> weightsAccum(particle_count);
	accumulate(weight_array, weightsAccum);
	float* weight_array_accum = thrust::raw_pointer_cast(weightsAccum.data());

	updatePersistentParticlesKernel2<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(grid_cell_array,
		weight_array_accum, grid_cell_count);

	CHECK_ERROR(cudaGetLastError());

	updatePersistentParticlesKernel3<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, meas_cell_array,
		grid_cell_array, weight_array, particle_count);

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::initializeNewParticles()
{
	std::cout << "OccupancyGridMap::initializeNewParticles" << std::endl;

	CHECK_ERROR(cudaDeviceSynchronize());

	thrust::device_vector<float> particleOrdersAccum(grid_cell_count);
	accumulate(born_masses_array, particleOrdersAccum);
	float* particle_orders_array_accum = thrust::raw_pointer_cast(particleOrdersAccum.data());

	normalize_particle_orders(particle_orders_array_accum, grid_cell_count, params.new_born_particle_count);

	initNewParticlesKernel1<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, grid_cell_array,
		meas_cell_array, weight_array, born_masses_array, birth_particle_array, particle_orders_array_accum, grid_cell_count);

	CHECK_ERROR(cudaGetLastError());

	initNewParticlesKernel2<<<divUp(new_born_particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(birth_particle_array,
		grid_cell_array, birth_weight_array, grid_width, new_born_particle_count);

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::statisticalMoments()
{
	std::cout << "OccupancyGridMap::statisticalMoments" << std::endl;

	statisticalMomentsKernel1<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, weight_array,
		vel_x_array, vel_y_array, vel_x_squared_array, vel_y_squared_array, vel_xy_array, particle_count);

	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());

	thrust::device_vector<float> velXAccum(particle_count);
	accumulate(vel_x_array, velXAccum);
	float* vel_x_array_accum = thrust::raw_pointer_cast(velXAccum.data());

	thrust::device_vector<float> velYAccum(particle_count);
	accumulate(vel_y_array, velYAccum);
	float* vel_y_array_accum = thrust::raw_pointer_cast(velYAccum.data());

	thrust::device_vector<float> velXSquaredAccum(particle_count);
	accumulate(vel_x_squared_array, velXSquaredAccum);
	float* vel_x_squared_array_accum = thrust::raw_pointer_cast(velXSquaredAccum.data());

	thrust::device_vector<float> velYSquaredAccum(particle_count);
	accumulate(vel_y_squared_array, velYSquaredAccum);
	float* vel_y_squared_array_accum = thrust::raw_pointer_cast(velYSquaredAccum.data());

	thrust::device_vector<float> velXYAccum(particle_count);
	accumulate(vel_xy_array, velYSquaredAccum);
	float* vel_xy_array_accum = thrust::raw_pointer_cast(velXYAccum.data());

	statisticalMomentsKernel2<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(grid_cell_array, vel_x_array_accum,
		vel_y_array_accum, vel_x_squared_array_accum, vel_y_squared_array_accum, vel_xy_array_accum, grid_cell_count);

	CHECK_ERROR(cudaGetLastError());
}

void OccupancyGridMap::resampling()
{
	std::cout << "OccupancyGridMap::resampling" << std::endl;

	CHECK_ERROR(cudaDeviceSynchronize());

	float max = static_cast<float>(particle_count + new_born_particle_count);
	thrust::device_vector<float> random_numbers(particle_count);
	thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(particle_count), random_numbers.begin(),
		GPU_LAMBDA(int index)
	{
		thrust::default_random_engine rand_eng;
		thrust::uniform_real_distribution<float> dist(0.0f, max);
		rand_eng.discard(index);
		return dist(rand_eng);
	});
	thrust::sort(random_numbers.begin(), random_numbers.end());

	thrust::device_vector<float> weight_accum(particle_count);
	thrust::device_vector<float> new_born_weight_accum(particle_count);
	accumulate(weight_array, weight_accum);
	accumulate(birth_weight_array, new_born_weight_accum);

	float offset = weight_accum.back();
	thrust::transform(new_born_weight_accum.begin(), new_born_weight_accum.end(), new_born_weight_accum.begin(),
		GPU_LAMBDA(float x)
	{
		return x + offset;
	});

	thrust::device_vector<float> joint_weight_accum(weight_accum.size() + new_born_weight_accum.size());
	joint_weight_accum.insert(joint_weight_accum.end(), weight_accum.begin(), weight_accum.end());
	joint_weight_accum.insert(joint_weight_accum.end(), new_born_weight_accum.begin(), new_born_weight_accum.end());

	thrust::device_vector<int> idx_resampled(particle_count);
	calc_resampled_indeces(joint_weight_accum, random_numbers, idx_resampled);
	int* idx_array_resampled = thrust::raw_pointer_cast(idx_resampled.data());

	resamplingKernel<<<divUp(particle_count, BLOCK_SIZE), BLOCK_SIZE>>>(particle_array, particle_array_next,
		birth_particle_array, idx_array_resampled, particle_count);

	CHECK_ERROR(cudaGetLastError());
}
