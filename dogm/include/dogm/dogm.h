// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include "dogm_types.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <memory>

#include <vector>

namespace dogm
{

class DOGM
{
public:
    struct Params
    {
        float size;
        float resolution;
        int particle_count;
        int new_born_particle_count;
        float persistence_prob;
        float stddev_process_noise_position;
        float stddev_process_noise_velocity;
        float birth_prob;
        float stddev_velocity;
        float init_max_velocity;
    };

    DOGM(const Params& params);
    ~DOGM();

    void updatePose(float new_x, float new_y);
    void addMeasurementGrid(MeasurementCell* measurement_grid, bool device);
    void updateGrid(float dt);

    std::vector<GridCell> getGridCells() const;
    std::vector<MeasurementCell> getMeasurementCells() const;

    ParticlesSoA getParticles() const;

    int getGridSize() const { return grid_size; }
    float getResolution() const { return params.resolution; }

    float getPositionX() const { return position_x; }
    float getPositionY() const { return position_y; }

    int getIteration() const { return iteration; }

private:
    void initialize();

public:
    void initializeParticles();

    void particlePrediction(float dt);
    void particleAssignment();
    void gridCellOccupancyUpdate();
    void updatePersistentParticles();
    void initializeNewParticles();
    void statisticalMoments();
    void resampling();

public:
    Params params;

    GridCell* grid_cell_array;
    ParticlesSoA particle_array;
    ParticlesSoA particle_array_next;
    ParticlesSoA birth_particle_array;
    MeasurementCell* meas_cell_array;

    float* weight_array;
    float* birth_weight_array;
    float* born_masses_array;

    float* vel_x_array;
    float* vel_y_array;

    float* vel_x_squared_array;
    float* vel_y_squared_array;
    float* vel_xy_array;

    float* rand_array;

    curandState* rng_states;

    int grid_size;

    int grid_cell_count;
    int particle_count;
    int new_born_particle_count;

    dim3 block_dim;
    dim3 particles_grid;
    dim3 birth_particles_grid;
    dim3 grid_map_grid;

private:
    int iteration;

    bool first_pose_received;
    bool first_measurement_received;
    float position_x;
    float position_y;
};

} /* namespace dogm */
