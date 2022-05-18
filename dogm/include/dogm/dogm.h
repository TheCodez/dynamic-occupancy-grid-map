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
    /**
     * Parameters used for the DOGM
     */
    struct Params
    {
        // Grid size [m]
        float size;

        // Grid cell size [m/cell]
        float resolution;

        // Number of persistent particles
        int particle_count;

        // Number of birth particles
        int new_born_particle_count;

        // Probability of persistence
        float persistence_prob;

        // Process noise position
        float stddev_process_noise_position;

        // Process noise velocity
        float stddev_process_noise_velocity;

        // Probability of birth
        float birth_prob;

        // Velocity to sample birth particles from (normal distribution) [m/s]
        float stddev_velocity;

        // Velocity to sample the initial particles from (uniform distribution) [m/s]
        float init_max_velocity;
    };

    /**
     * Constructor.
     * @params params parameter used for the grid map and the particle filter.
     */
    DOGM(const Params& params);

    /**
     * Destructor.
     */
    ~DOGM();

    /**
     * Updates the grid map and particle filter to the new timestep.
     * @param measurement_grid new measurement grid map.
     * @param new_x new x pose.
     * @param new_y new y pose.
     * @param new_yaw new yaw.
     * @param dt delta time since the last update.
     * @param device whether the measurement grid resides in GPU memory (default: true).
     */
     void updateGrid(MeasurementCellsSoA measurement_grid, float new_x, float new_y, float dt,
                    bool device = true);

    /**
     * Returns the grid map in the host memory.
     *
     * @return grid map.
     */
    GridCellsSoA getGridCells() const;

    /**
     * Returns the measurement grid map in the host memory.
     *
     * @return measurement grid map.
     */
    MeasurementCellsSoA getMeasurementCells() const;

    /**
     * Returns the persistent particles of the particle filter.
     *
     * @return particle array.
     */
    ParticlesSoA getParticles() const;

    /**
     * Returns the grid map size in cells.
     *
     * @return grid size in cells.
     */
    int getGridSize() const { return grid_size; }

    /**
     * Returns the grid map resolution.
     *
     * @return resolution [m/cell].
     */
    float getResolution() const { return params.resolution; }

    /**
     * Returns the x position.
     *
     * @return x position.
     */
    float getPositionX() const { return position_x; }

    /**
     * Returns the vehicles yaw.
     *
     * @return yaw.
     */
    float getYaw() const { return yaw; }

    /**
     * Returns the y position.
     *
     * @return y position.
     */
    float getPositionY() const { return position_y; }

    int getIteration() const { return iteration; }

private:
    void initialize();

    void updatePose(float new_x, float new_y);
    void updateMeasurementGrid(MeasurementCellsSoA measurement_grid, bool device);

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

    GridCellsSoA grid_cell_array;
    ParticlesSoA particle_array;
    ParticlesSoA particle_array_next;
    ParticlesSoA birth_particle_array;
    MeasurementCellsSoA meas_cell_array;

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
