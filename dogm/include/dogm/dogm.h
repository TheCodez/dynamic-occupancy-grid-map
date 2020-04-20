// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <memory>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include <vector>

#include "dogm_types.h"

class Renderer;

namespace dogm
{

// struct GridCell;
// struct MeasurementCell;
// struct Particle;
// struct ParticleSoA;

struct GridParams
{
    float size;
    float resolution;
    int particle_count;
    int new_born_particle_count;
    float persistence_prob;
    float process_noise_position;
    float process_noise_velocity;
    float birth_prob;
    float velocity_persistent;
    float velocity_birth;
};

struct LaserSensorParams
{
    float max_range;
    float resolution;
    float fov;
};

class DOGM
{
public:
    DOGM(const GridParams& params, const LaserSensorParams& laser_params);
    ~DOGM();

    void updateMeasurementGridFromArray(const std::vector<float2>& measurements);

    void updateMeasurementGrid(float* measurements, int num_measurements);
    void updateParticleFilter(float dt);

    int getGridSize() const { return grid_size; }
    float getResolution() const { return params.resolution; }

    float getPositionX() const { return 0.0f; }
    float getPositionY() const { return 0.0f; }

    int getIteration() const { return iteration; }

private:
    void initialize();

    void sortParticles(ParticleSoA particles);

public:
    void particlePrediction(float dt);
    void particleAssignment();
    void gridCellOccupancyUpdate();
    void updatePersistentParticles();
    void initializeNewParticles();
    void statisticalMoments();
    void resampling();

public:
    GridParams params;
    LaserSensorParams laser_params;

    std::unique_ptr<Renderer> renderer;

    GridCell* grid_cell_array;
    struct ParticleSoA particle_array;
    struct ParticleSoA particle_array_next;
    struct ParticleSoA birth_particle_array;

    MeasurementCell* polar_meas_cell_array;
    MeasurementCell* meas_cell_array;

    float* weight_array;
    float* birth_weight_array;
    float* born_masses_array;

    curandState* rng_states;

    int grid_size;

    int grid_cell_count;
    int particle_count;
    int new_born_particle_count;

    dim3 block_dim;
    dim3 particles_grid;
    dim3 birth_particles_grid;
    dim3 grid_map_grid;

    int iteration;
};

} /* namespace dogm */
