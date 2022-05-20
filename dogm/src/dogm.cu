// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm.h"
#include "dogm/dogm_types.h"

#include "dogm/kernel/ego_motion_compensation.h"
#include "dogm/kernel/init.h"
#include "dogm/kernel/init_new_particles.h"
#include "dogm/kernel/mass_update.h"
#include "dogm/kernel/particle_to_grid.h"
#include "dogm/kernel/predict.h"
#include "dogm/kernel/resampling.h"
#include "dogm/kernel/statistical_moments.h"
#include "dogm/kernel/update_persistent_particles.h"

#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cuda_runtime.h>

#include <vector>

namespace dogm
{

constexpr int BLOCK_SIZE = 256;

DOGM::DOGM(const Params& params)
    : params(params), grid_size(static_cast<int>(params.size / params.resolution)),
      particle_count(params.particle_count), grid_cell_count(grid_size * grid_size),
      new_born_particle_count(params.new_born_particle_count), block_dim(BLOCK_SIZE), first_pose_received(false),
      first_measurement_received(false), position_x(0.0f), position_y(0.0f)
{
    int device;
    CUDA_CALL(cudaGetDevice(&device));

    cudaDeviceProp device_prop;
    CUDA_CALL(cudaGetDeviceProperties(&device_prop, device));

    int blocks_per_sm = device_prop.maxThreadsPerMultiProcessor / block_dim.x;
    dim3 dim(device_prop.multiProcessorCount * blocks_per_sm);
    particles_grid = birth_particles_grid = grid_map_grid = dim;

    particle_array.init(particle_count, true);
    particle_array_next.init(particle_count, true);
    birth_particle_array.init(new_born_particle_count, true);

    grid_cell_array.init(grid_cell_count, true);
    meas_cell_array.init(grid_cell_count, true);

    CUDA_CALL(cudaMalloc(&weight_array, particle_count * sizeof(float)));
    CUDA_CALL(cudaMalloc(&birth_weight_array, new_born_particle_count * sizeof(float)));
    CUDA_CALL(cudaMalloc(&born_masses_array, grid_cell_count * sizeof(float)));

    CUDA_CALL(cudaMalloc(&vel_x_array, particle_count * sizeof(float)));
    CUDA_CALL(cudaMalloc(&vel_y_array, particle_count * sizeof(float)));
    CUDA_CALL(cudaMalloc(&vel_x_squared_array, particle_count * sizeof(float)));
    CUDA_CALL(cudaMalloc(&vel_y_squared_array, particle_count * sizeof(float)));
    CUDA_CALL(cudaMalloc(&vel_xy_array, particle_count * sizeof(float)));

    CUDA_CALL(cudaMalloc(&rand_array, particle_count * sizeof(float)));

    CUDA_CALL(cudaMalloc(&rng_states, particles_grid.x * block_dim.x * sizeof(curandState)));

    initialize();
}

DOGM::~DOGM()
{
    particle_array.free();
    particle_array_next.free();
    birth_particle_array.free();

    grid_cell_array.free();
    meas_cell_array.free();

    CUDA_CALL(cudaFree(weight_array));
    CUDA_CALL(cudaFree(birth_weight_array));
    CUDA_CALL(cudaFree(born_masses_array));

    CUDA_CALL(cudaFree(vel_x_array));
    CUDA_CALL(cudaFree(vel_y_array));
    CUDA_CALL(cudaFree(vel_x_squared_array));
    CUDA_CALL(cudaFree(vel_y_squared_array));
    CUDA_CALL(cudaFree(vel_xy_array));

    CUDA_CALL(cudaFree(rand_array));

    CUDA_CALL(cudaFree(rng_states));
}

void DOGM::initialize()
{
    cudaStream_t particles_stream, grid_stream;
    CUDA_CALL(cudaStreamCreate(&particles_stream));
    CUDA_CALL(cudaStreamCreate(&grid_stream));

    setupRandomStatesKernel<<<particles_grid, block_dim>>>(rng_states, 123456, particles_grid.x * block_dim.x);

    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    initGridCellsKernel<<<grid_map_grid, block_dim, 0, grid_stream>>>(grid_cell_array, meas_cell_array, grid_size,
                                                                      grid_cell_count);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaStreamDestroy(particles_stream));
    CUDA_CALL(cudaStreamDestroy(grid_stream));
}

void DOGM::updateGrid(MeasurementCellsSoA measurement_grid, float new_x, float new_y, float new_yaw, float dt,
                      bool device)
{
    updateMeasurementGrid(measurement_grid, device);
    updatePose(new_x, new_y, new_yaw);

    particlePrediction(dt);
    particleAssignment();
    gridCellOccupancyUpdate();
    updatePersistentParticles();
    initializeNewParticles();
    statisticalMoments();
    resampling();

    particle_array = particle_array_next;

    CUDA_CALL(cudaDeviceSynchronize());
}

GridCellsSoA DOGM::getGridCells() const
{
    GridCellsSoA grid_cells(grid_cell_count, false);
    grid_cells.copy(grid_cell_array, cudaMemcpyDeviceToHost);

    return grid_cells;
}

MeasurementCellsSoA DOGM::getMeasurementCells() const
{
    MeasurementCellsSoA meas_cells(grid_cell_count, false);
    meas_cells.copy(meas_cell_array, cudaMemcpyDeviceToHost);

    return meas_cells;
}

ParticlesSoA DOGM::getParticles() const
{
    ParticlesSoA particles(particle_count, false);
    particles.copy(particle_array, cudaMemcpyDeviceToHost);

    return particles;
}

void DOGM::updatePose(float new_x, float new_y, float new_yaw)
{
    if (!first_pose_received)
    {
        position_x = new_x;
        position_y = new_y;
        yaw = new_yaw;
        first_pose_received = true;
    }
    else
    {
        const float x_diff = new_x - position_x;
        const float y_diff = new_y - position_y;

        if (fabsf(x_diff) > params.resolution || fabsf(y_diff) > params.resolution)
        {
            const int x_move = -static_cast<int>(x_diff / params.resolution);
            const int y_move = -static_cast<int>(y_diff / params.resolution);

            moveParticlesKernel<<<particles_grid, block_dim>>>(particle_array, x_move, y_move, particle_count);

            dim3 dim_block(32, 32);
            dim3 grid_dim(divUp(grid_size, dim_block.x), divUp(grid_size, dim_block.y));

            GridCellsSoA old_grid_cell_array(grid_cell_count, true);
            old_grid_cell_array.copy(grid_cell_array, cudaMemcpyDeviceToDevice);

            moveMapKernel<<<grid_dim, dim_block>>>(grid_cell_array, old_grid_cell_array, meas_cell_array,
                                                   particle_array, x_move, y_move, grid_size);

            old_grid_cell_array.free();

            position_x = new_x;
            position_y = new_y;
            yaw = new_yaw;
        }
    }
}

void DOGM::updateMeasurementGrid(MeasurementCellsSoA measurement_grid, bool device)
{
    cudaMemcpyKind kind = device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    meas_cell_array.copy(measurement_grid, kind);

    if (!first_measurement_received)
    {
        initializeParticles();
        first_measurement_received = true;
    }
}

void DOGM::initializeParticles()
{
    copyMassesKernel<<<grid_map_grid, block_dim>>>(meas_cell_array, born_masses_array, grid_cell_count);

    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    thrust::device_vector<float> particle_orders_accum(grid_cell_count);
    accumulate(born_masses_array, particle_orders_accum);
    float* particle_orders_array_accum = thrust::raw_pointer_cast(particle_orders_accum.data());

    float new_weight = 1.0f / particle_count;

    normalize_particle_orders(particle_orders_array_accum, grid_cell_count, particle_count);

    initParticlesKernel1<<<grid_map_grid, block_dim>>>(particle_array, particle_orders_array_accum, grid_cell_count);

    CUDA_CALL(cudaGetLastError());

    initParticlesKernel2<<<particles_grid, block_dim>>>(particle_array, rng_states, params.init_max_velocity, grid_size,
                                                        new_weight, particle_count);

    CUDA_CALL(cudaGetLastError());
}

void DOGM::particlePrediction(float dt)
{
    // clang-format off
    glm::mat4x4 transition_matrix(1, 0, dt, 0,
                                  0, 1, 0, dt,
                                  0, 0, 1, 0,
                                  0, 0, 0, 1);
    // clang-format on

    // FIXME: glm uses column major, we need row major
    transition_matrix = glm::transpose(transition_matrix);

    predictKernel<<<particles_grid, block_dim>>>(
        particle_array, rng_states, params.stddev_velocity, grid_size, params.persistence_prob, transition_matrix,
        params.stddev_process_noise_position, params.stddev_process_noise_velocity, particle_count);

    CUDA_CALL(cudaGetLastError());
}

void DOGM::particleAssignment()
{
    reinitGridParticleIndices<<<grid_map_grid, block_dim>>>(grid_cell_array, grid_cell_count);

    CUDA_CALL(cudaGetLastError());
    // CUDA_CALL(cudaDeviceSynchronize());

    // sort particles
    thrust::device_ptr<int> grid_index_ptr(particle_array.grid_cell_idx);
    thrust::device_ptr<float> weight_ptr(particle_array.weight);
    thrust::device_ptr<bool> associated_ptr(particle_array.associated);
    thrust::device_ptr<glm::vec4> state_ptr(particle_array.state);

    auto it = thrust::make_zip_iterator(thrust::make_tuple(weight_ptr, associated_ptr, state_ptr));
    thrust::sort_by_key(grid_index_ptr, grid_index_ptr + particle_count, it);

    particleToGridKernel<<<particles_grid, block_dim>>>(particle_array, grid_cell_array, weight_array, particle_count);

    CUDA_CALL(cudaGetLastError());
}

void DOGM::gridCellOccupancyUpdate()
{
    // CUDA_CALL(cudaDeviceSynchronize());

    thrust::device_vector<float> weights_accum(particle_count);
    accumulate(weight_array, weights_accum);
    float* weight_array_accum = thrust::raw_pointer_cast(weights_accum.data());

    gridCellPredictionUpdateKernel<<<grid_map_grid, block_dim>>>(grid_cell_array, particle_array, weight_array,
                                                                 weight_array_accum, meas_cell_array, born_masses_array,
                                                                 params.birth_prob, grid_cell_count);

    CUDA_CALL(cudaGetLastError());
}

void DOGM::updatePersistentParticles()
{
    updatePersistentParticlesKernel1<<<particles_grid, block_dim>>>(particle_array, meas_cell_array, weight_array,
                                                                    particle_count);

    CUDA_CALL(cudaGetLastError());
    // CUDA_CALL(cudaDeviceSynchronize());

    thrust::device_vector<float> weights_accum(particle_count);
    accumulate(weight_array, weights_accum);
    float* weight_array_accum = thrust::raw_pointer_cast(weights_accum.data());

    updatePersistentParticlesKernel2<<<divUp(grid_cell_count, BLOCK_SIZE), BLOCK_SIZE>>>(
        grid_cell_array, weight_array_accum, grid_cell_count);

    CUDA_CALL(cudaGetLastError());

    updatePersistentParticlesKernel3<<<particles_grid, block_dim>>>(particle_array, meas_cell_array, grid_cell_array,
                                                                    weight_array, particle_count);

    CUDA_CALL(cudaGetLastError());
}

void DOGM::initializeNewParticles()
{
    initBirthParticlesKernel<<<birth_particles_grid, block_dim>>>(
        birth_particle_array, rng_states, params.stddev_velocity, grid_size, new_born_particle_count);

    CUDA_CALL(cudaGetLastError());
    // CUDA_CALL(cudaDeviceSynchronize());

    thrust::device_vector<float> particle_orders_accum(grid_cell_count);
    accumulate(born_masses_array, particle_orders_accum);
    float* particle_orders_array_accum = thrust::raw_pointer_cast(particle_orders_accum.data());

    normalize_particle_orders(particle_orders_array_accum, grid_cell_count, new_born_particle_count);

    initNewParticlesKernel1<<<grid_map_grid, block_dim>>>(grid_cell_array, meas_cell_array, weight_array,
                                                          born_masses_array, birth_particle_array,
                                                          particle_orders_array_accum, grid_cell_count);

    CUDA_CALL(cudaGetLastError());

    initNewParticlesKernel2<<<birth_particles_grid, block_dim>>>(birth_particle_array, grid_cell_array, rng_states,
                                                                 params.stddev_velocity, params.init_max_velocity,
                                                                 grid_size, new_born_particle_count);

    CUDA_CALL(cudaGetLastError());
}

void DOGM::statisticalMoments()
{
    statisticalMomentsKernel1<<<particles_grid, block_dim>>>(particle_array, weight_array, vel_x_array, vel_y_array,
                                                             vel_x_squared_array, vel_y_squared_array, vel_xy_array,
                                                             particle_count);

    CUDA_CALL(cudaGetLastError());
    // CUDA_CALL(cudaDeviceSynchronize());

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

    CUDA_CALL(cudaGetLastError());
}

void DOGM::resampling()
{
    // CUDA_CALL(cudaDeviceSynchronize());

    thrust::device_ptr<float> persistent_weights(weight_array);
    thrust::device_ptr<float> new_born_weights(birth_particle_array.weight);

    thrust::device_vector<float> joint_weight_array;
    joint_weight_array.insert(joint_weight_array.end(), persistent_weights, persistent_weights + particle_count);
    joint_weight_array.insert(joint_weight_array.end(), new_born_weights, new_born_weights + new_born_particle_count);

    thrust::device_vector<float> joint_weight_accum(joint_weight_array.size());
    accumulate(joint_weight_array, joint_weight_accum);

    float joint_max = joint_weight_accum.back();

    resamplingGenerateRandomNumbersKernel<<<particles_grid, block_dim>>>(rand_array, rng_states, joint_max,
                                                                         particle_count);

    CUDA_CALL(cudaGetLastError());
    // CUDA_CALL(cudaDeviceSynchronize());

    thrust::device_ptr<float> rand_ptr(rand_array);
    thrust::device_vector<float> rand_vector(rand_ptr, rand_ptr + particle_count);

    thrust::sort(rand_vector.begin(), rand_vector.end());

    thrust::device_vector<int> idx_resampled(particle_count);
    calc_resampled_indices(joint_weight_accum, rand_vector, idx_resampled, joint_max);
    int* idx_array_resampled = thrust::raw_pointer_cast(idx_resampled.data());

    float new_weight = joint_max / particle_count;

    resamplingKernel<<<particles_grid, block_dim>>>(particle_array, particle_array_next, birth_particle_array,
                                                    idx_array_resampled, new_weight, particle_count);

    CUDA_CALL(cudaGetLastError());
}

} /* namespace dogm */
