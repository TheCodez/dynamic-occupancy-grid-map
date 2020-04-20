// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm.h"
#include "dogm/dogm_types.h"

#include "dogm/kernel/init.h"
#include "dogm/kernel/init_new_particles.h"
#include "dogm/kernel/mass_update.h"
#include "dogm/kernel/measurement_grid.h"
#include "dogm/kernel/particle_to_grid.h"
#include "dogm/kernel/predict.h"
#include "dogm/kernel/resampling.h"
#include "dogm/kernel/statistical_moments.h"
#include "dogm/kernel/update_persistent_particles.h"

#include "dogm/opengl/framebuffer.h"
#include "dogm/opengl/renderer.h"
#include "dogm/opengl/texture.h"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cuda_runtime.h>

namespace dogm
{

constexpr int BLOCK_SIZE = 256;

DOGM::DOGM(const GridParams& params, const LaserSensorParams& laser_params)
    : params(params), laser_params(laser_params), grid_size(static_cast<int>(params.size / params.resolution)),
      particle_count(params.particle_count), grid_cell_count(grid_size * grid_size),
      new_born_particle_count(params.new_born_particle_count), block_dim(BLOCK_SIZE)
{
    int device;
    CHECK_ERROR(cudaGetDevice(&device));

    cudaDeviceProp device_prop;
    CHECK_ERROR(cudaGetDeviceProperties(&device_prop, device));

    int blocks_per_sm = device_prop.maxThreadsPerMultiProcessor / block_dim.x;
    dim3 dim(device_prop.multiProcessorCount * blocks_per_sm);
    particles_grid = birth_particles_grid = grid_map_grid = dim;

    CHECK_ERROR(cudaMallocManaged((void**)&grid_cell_array, grid_cell_count * sizeof(GridCell)));

    particle_array.init(particle_count);
    particle_array_next.init(particle_count);
    birth_particle_array.init(new_born_particle_count);

    CHECK_ERROR(cudaMallocManaged((void**)&meas_cell_array, grid_cell_count * sizeof(MeasurementCell)));

    CHECK_ERROR(cudaMallocManaged((void**)&polar_meas_cell_array, 100 * grid_size * sizeof(MeasurementCell)));

    CHECK_ERROR(cudaMalloc(&weight_array, particle_count * sizeof(float)));
    CHECK_ERROR(cudaMalloc(&birth_weight_array, new_born_particle_count * sizeof(float)));
    CHECK_ERROR(cudaMalloc(&born_masses_array, grid_cell_count * sizeof(float)));

    CHECK_ERROR(cudaMalloc(&rng_states, particles_grid.x * block_dim.x * sizeof(curandState)));

    renderer = std::make_unique<Renderer>(grid_size, laser_params.fov, params.size, laser_params.max_range);

    initialize();
}

DOGM::~DOGM()
{
    CHECK_ERROR(cudaFree(grid_cell_array));
    CHECK_ERROR(cudaFree(meas_cell_array));

    CHECK_ERROR(cudaFree(polar_meas_cell_array));

    CHECK_ERROR(cudaFree(weight_array));
    CHECK_ERROR(cudaFree(birth_weight_array));
    CHECK_ERROR(cudaFree(born_masses_array));

    CHECK_ERROR(cudaFree(rng_states));
}

void DOGM::initialize()
{
    cudaStream_t particles_stream, grid_stream;
    CHECK_ERROR(cudaStreamCreate(&particles_stream));
    CHECK_ERROR(cudaStreamCreate(&grid_stream));

    setupRandomStatesKernel<<<particles_grid, block_dim>>>(rng_states, 123456, particles_grid.x * block_dim.x);

    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());

    initParticlesKernel<<<particles_grid, block_dim, 0, particles_stream>>>(
        particle_array, rng_states, params.velocity_persistent, grid_size, particle_count);

    initGridCellsKernel<<<grid_map_grid, block_dim, 0, grid_stream>>>(grid_cell_array, meas_cell_array, grid_size,
                                                                      grid_cell_count);

    CHECK_ERROR(cudaGetLastError());

    CHECK_ERROR(cudaStreamDestroy(particles_stream));
    CHECK_ERROR(cudaStreamDestroy(grid_stream));
}

void DOGM::sortParticles(ParticleSoA particles)
{
    thrust::device_ptr<int> grid_index_ptr(particles.grid_cell_idx);
    thrust::device_ptr<float> weight_ptr(particles.weight);
    thrust::device_ptr<bool> associated_ptr(particles.associated);
    thrust::device_ptr<glm::vec4> state_ptr(particles.state);

    auto it = thrust::make_zip_iterator(thrust::make_tuple(weight_ptr, associated_ptr, state_ptr));
    thrust::sort_by_key(grid_index_ptr, grid_index_ptr + particles.size, it);
}

void DOGM::updateParticleFilter(float dt)
{
    particlePrediction(dt);
    particleAssignment();
    gridCellOccupancyUpdate();
    updatePersistentParticles();
    initializeNewParticles();
    statisticalMoments();
    resampling();

    //    CHECK_ERROR(
    //        cudaMemcpy(particle_array, particle_array_next, particle_count * sizeof(Particle),
    //        cudaMemcpyDeviceToDevice));
    particle_array = particle_array_next;

    CHECK_ERROR(cudaDeviceSynchronize());

    iteration++;
}

void DOGM::updateMeasurementGridFromArray(const std::vector<float2>& measurements)
{
    thrust::device_vector<float2> d_measurements(measurements);
    float2* d_measurements_array = thrust::raw_pointer_cast(d_measurements.data());

    dim3 dim_block(32, 32);
    dim3 cart_grid_dim(divUp(grid_size, dim_block.x), divUp(grid_size, dim_block.y));

    gridArrayToMeasurementGridKernel<<<cart_grid_dim, dim_block>>>(meas_cell_array, d_measurements_array, grid_size);

    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());
}

void DOGM::updateMeasurementGrid(float* measurements, int num_measurements)
{
    // std::cout << "DOGM::updateMeasurementGrid" << std::endl;

    float* d_measurements;
    CHECK_ERROR(cudaMalloc(&d_measurements, num_measurements * sizeof(float)));
    CHECK_ERROR(cudaMemcpy(d_measurements, measurements, num_measurements * sizeof(float), cudaMemcpyHostToDevice));

    const int polar_width = num_measurements;
    const int polar_height = static_cast<int>(laser_params.max_range / laser_params.resolution);

    dim3 dim_block(32, 32);
    dim3 grid_dim(divUp(polar_width, dim_block.x), divUp(polar_height, dim_block.y));
    dim3 cart_grid_dim(divUp(grid_size, dim_block.x), divUp(grid_size, dim_block.y));

    const float anisotropy_level = 16.0f;
    Texture polar_texture(polar_width, polar_height, anisotropy_level);
    cudaSurfaceObject_t polar_surface;

    // create polar texture
    polar_texture.beginCudaAccess(&polar_surface);
    createPolarGridTextureKernel2<<<grid_dim, dim_block>>>(polar_surface, polar_meas_cell_array, d_measurements,
                                                           polar_width, polar_height, params.resolution);

    CHECK_ERROR(cudaGetLastError());
    polar_texture.endCudaAccess(polar_surface);

    // render cartesian image to texture using polar texture
    renderer->renderToTexture(polar_texture);

    Framebuffer* framebuffer = renderer->getFrameBuffer();
    cudaSurfaceObject_t cartesian_surface;

    framebuffer->beginCudaAccess(&cartesian_surface);
    // transform RGBA texture to measurement grid
    cartesianGridToMeasurementGridKernel<<<cart_grid_dim, dim_block>>>(meas_cell_array, cartesian_surface, grid_size);

    CHECK_ERROR(cudaGetLastError());
    framebuffer->endCudaAccess(cartesian_surface);

    CHECK_ERROR(cudaFree(d_measurements));
    CHECK_ERROR(cudaDeviceSynchronize());
}

void DOGM::particlePrediction(float dt)
{
    // std::cout << "DOGM::particlePrediction" << std::endl;

    // clang-format off
    glm::mat4x4 transition_matrix(1, 0, dt, 0,
                                  0, 1, 0, dt,
                                  0, 0, 1, 0,
                                  0, 0, 0, 1);
    // clang-format on

    // FIXME: glm uses column major, we need row major
    transition_matrix = glm::transpose(transition_matrix);

    predictKernel<<<particles_grid, block_dim>>>(
        particle_array, rng_states, params.velocity_persistent, grid_size, params.persistence_prob, transition_matrix,
        params.process_noise_position, params.process_noise_velocity, particle_count);

    CHECK_ERROR(cudaGetLastError());
}

void DOGM::particleAssignment()
{
    // std::cout << "DOGM::particleAssignment" << std::endl;

    reinitGridParticleIndices<<<grid_map_grid, block_dim>>>(grid_cell_array, grid_cell_count);

    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());

    sortParticles(particle_array);
    particleToGridKernel<<<particles_grid, block_dim>>>(particle_array, grid_cell_array, weight_array, particle_count);

    CHECK_ERROR(cudaGetLastError());
}

void DOGM::gridCellOccupancyUpdate()
{
    // std::cout << "DOGM::gridCellOccupancyUpdate" << std::endl;

    CHECK_ERROR(cudaDeviceSynchronize());

    thrust::device_vector<float> weights_accum(particle_count);
    accumulate(weight_array, weights_accum);
    float* weight_array_accum = thrust::raw_pointer_cast(weights_accum.data());

    gridCellPredictionUpdateKernel<<<grid_map_grid, block_dim>>>(grid_cell_array, particle_array, weight_array,
                                                                 weight_array_accum, meas_cell_array, born_masses_array,
                                                                 params.birth_prob, grid_cell_count);

    CHECK_ERROR(cudaGetLastError());
}

void DOGM::updatePersistentParticles()
{
    // std::cout << "DOGM::updatePersistentParticles" << std::endl;

    updatePersistentParticlesKernel1<<<particles_grid, block_dim>>>(particle_array, meas_cell_array, weight_array,
                                                                    particle_count);

    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());

    thrust::device_vector<float> weights_accum(particle_count);
    accumulate(weight_array, weights_accum);
    float* weight_array_accum = thrust::raw_pointer_cast(weights_accum.data());

    updatePersistentParticlesKernel2<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(
        grid_cell_array, weight_array_accum, grid_cell_count);

    CHECK_ERROR(cudaGetLastError());

    updatePersistentParticlesKernel3<<<particles_grid, block_dim>>>(particle_array, meas_cell_array, grid_cell_array,
                                                                    weight_array, particle_count);

    CHECK_ERROR(cudaGetLastError());
}

void DOGM::initializeNewParticles()
{
    // std::cout << "DOGM::initializeNewParticles" << std::endl;

    initBirthParticlesKernel<<<birth_particles_grid, block_dim>>>(
        birth_particle_array, rng_states, params.velocity_birth, grid_size, new_born_particle_count);

    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());

    thrust::device_vector<float> particle_orders_accum(grid_cell_count);
    accumulate(born_masses_array, particle_orders_accum);
    float* particle_orders_array_accum = thrust::raw_pointer_cast(particle_orders_accum.data());

    normalize_particle_orders(particle_orders_array_accum, grid_cell_count, new_born_particle_count);

    initNewParticlesKernel1<<<grid_map_grid, block_dim>>>(grid_cell_array, meas_cell_array, weight_array,
                                                          born_masses_array, birth_particle_array,
                                                          particle_orders_array_accum, grid_cell_count);

    CHECK_ERROR(cudaGetLastError());

    initNewParticlesKernel2<<<birth_particles_grid, block_dim>>>(
        birth_particle_array, grid_cell_array, rng_states, params.velocity_birth, grid_size, new_born_particle_count);

    CHECK_ERROR(cudaGetLastError());

    copyBirthWeightKernel<<<birth_particles_grid, block_dim>>>(birth_particle_array, birth_weight_array,
                                                               new_born_particle_count);

    CHECK_ERROR(cudaGetLastError());
}

void DOGM::statisticalMoments()
{
    // std::cout << "DOGM::statisticalMoments" << std::endl;

    thrust::device_vector<float> vel_x(particle_count);
    float* vel_x_array = thrust::raw_pointer_cast(vel_x.data());

    thrust::device_vector<float> vel_y(particle_count);
    float* vel_y_array = thrust::raw_pointer_cast(vel_y.data());

    thrust::device_vector<float> vel_x_squared(particle_count);
    float* vel_x_squared_array = thrust::raw_pointer_cast(vel_x_squared.data());

    thrust::device_vector<float> vel_y_squared(particle_count);
    float* vel_y_squared_array = thrust::raw_pointer_cast(vel_y_squared.data());

    thrust::device_vector<float> vel_xy(particle_count);
    float* vel_xy_array = thrust::raw_pointer_cast(vel_xy.data());

    statisticalMomentsKernel1<<<particles_grid, block_dim>>>(particle_array, weight_array, vel_x_array, vel_y_array,
                                                             vel_x_squared_array, vel_y_squared_array, vel_xy_array,
                                                             particle_count);

    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());

    thrust::device_vector<float> vel_x_accum(particle_count);
    accumulate(vel_x_array, vel_x_accum);
    float* vel_x_array_accum = thrust::raw_pointer_cast(vel_x_accum.data());

    thrust::device_vector<float> vel_y_accum(particle_count);
    accumulate(vel_y_array, vel_y_accum);
    float* vel_y_array_accum = thrust::raw_pointer_cast(vel_y_accum.data());

    thrust::device_vector<float> vel_x_squared_accum(particle_count);
    accumulate(vel_x_squared_array, vel_x_squared_accum);
    float* vel_x_squared_array_accum = thrust::raw_pointer_cast(vel_x_squared_accum.data());

    thrust::device_vector<float> vel_y_squared_accum(particle_count);
    accumulate(vel_y_squared_array, vel_y_squared_accum);
    float* vel_y_squared_array_accum = thrust::raw_pointer_cast(vel_y_squared_accum.data());

    thrust::device_vector<float> vel_xy_accum(particle_count);
    accumulate(vel_xy_array, vel_xy_accum);
    float* vel_xy_array_accum = thrust::raw_pointer_cast(vel_xy_accum.data());

    statisticalMomentsKernel2<<<grid_map_grid, block_dim>>>(grid_cell_array, vel_x_array_accum, vel_y_array_accum,
                                                            vel_x_squared_array_accum, vel_y_squared_array_accum,
                                                            vel_xy_array_accum, grid_cell_count);

    CHECK_ERROR(cudaGetLastError());
}

void DOGM::resampling()
{
    // std::cout << "DOGM::resampling" << std::endl;

    CHECK_ERROR(cudaDeviceSynchronize());

    thrust::device_ptr<float> persistent_weights(weight_array);
    thrust::device_ptr<float> new_born_weights(birth_weight_array);

    thrust::device_vector<float> joint_weight_array;
    joint_weight_array.insert(joint_weight_array.end(), persistent_weights, persistent_weights + particle_count);
    joint_weight_array.insert(joint_weight_array.end(), new_born_weights, new_born_weights + new_born_particle_count);

    thrust::device_vector<float> joint_weight_accum(joint_weight_array.size());
    accumulate(joint_weight_array, joint_weight_accum);

    float joint_max = joint_weight_accum.back();

    thrust::device_vector<float> rand_array(particle_count);
    float* rand_ptr = thrust::raw_pointer_cast(rand_array.data());

    resamplingGenerateRandomNumbersKernel<<<particles_grid, block_dim>>>(rand_ptr, rng_states, joint_max,
                                                                         particle_count);

    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());

    thrust::sort(rand_array.begin(), rand_array.end());

    thrust::device_vector<int> idx_resampled(particle_count);
    calc_resampled_indices(joint_weight_accum, rand_array, idx_resampled, joint_max);
    int* idx_array_resampled = thrust::raw_pointer_cast(idx_resampled.data());

    float new_weight = joint_max / particle_count;

    printf("joint_max: %f\n", joint_max);

    resamplingKernel<<<particles_grid, block_dim>>>(particle_array, particle_array_next, birth_particle_array,
                                                    idx_array_resampled, new_weight, particle_count);

    CHECK_ERROR(cudaGetLastError());
}

} /* namespace dogm */
