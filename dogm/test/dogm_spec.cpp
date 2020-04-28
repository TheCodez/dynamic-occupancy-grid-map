// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/dogm.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

TEST(DOGM, Predict)
{
    dogm::GridParams grid_params;
    grid_params.size = 10.0f;
    grid_params.resolution = 1.0f;
    grid_params.particle_count = 2;
    grid_params.new_born_particle_count = 1;
    grid_params.persistence_prob = 0.5f;
    grid_params.process_noise_position = 0.0f;
    grid_params.process_noise_velocity = 0.0f;
    grid_params.birth_prob = 0.02f;
    grid_params.velocity_persistent = 10.0f;
    grid_params.velocity_birth = 10.0f;

    dogm::LaserSensorParams laser_params;
    laser_params.fov = 120.0f;
    laser_params.max_range = 1.0f;
    laser_params.resolution = 1.0f;

    float delta_time = 0.1f;

    dogm::DOGM dogm(grid_params, laser_params);
    cudaDeviceSynchronize();

    dogm::ParticlesSoA particles = dogm.getParticles();

    glm::vec4 old_state = particles[0].state;
    glm::vec4 pred_state = old_state + delta_time * glm::vec4(old_state[2], old_state[3], 0, 0);
    float old_weight = particles[0].weight;

    dogm.particlePrediction(delta_time);
    cudaDeviceSynchronize();

    dogm::ParticlesSoA new_particles = dogm.getParticles();

    EXPECT_EQ(pred_state, new_particles[0].state);
    EXPECT_EQ(old_weight * grid_params.persistence_prob, new_particles[0].weight);
}
