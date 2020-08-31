// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/dogm.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

TEST(DOGM, EgoMotionCompensation)
{
    dogm::DOGM::Params grid_params;
    grid_params.size = 10.0f;
    grid_params.resolution = 1.0f;
    grid_params.particle_count = 2;
    grid_params.new_born_particle_count = 1;
    grid_params.persistence_prob = 0.5f;
    grid_params.stddev_process_noise_position = 0.0f;
    grid_params.stddev_process_noise_velocity = 0.0f;
    grid_params.birth_prob = 0.02f;
    grid_params.stddev_velocity = 10.0f;
    grid_params.init_max_velocity = 30.0f;

    dogm::DOGM dogm(grid_params);
    cudaDeviceSynchronize();

    dogm::ParticlesSoA particles = dogm.getParticles();
    glm::vec4 old_state = particles.state[0];
    glm::vec2 pose{10.0f, 10.0f};

    // Set initial pose (no position update)
    dogm.updatePose(pose.x, pose.y);
    cudaDeviceSynchronize();
    EXPECT_EQ(pose.x, dogm.getPositionX());
    EXPECT_EQ(pose.y, dogm.getPositionY());
    dogm::ParticlesSoA new_particles = dogm.getParticles();
    EXPECT_EQ(old_state, new_particles.state[0]);

    // Change lower than resolution doesn't lead to update after initial position is set
    dogm.updatePose(pose.x + 0.5f, pose.y + 0.5f);
    cudaDeviceSynchronize();
    EXPECT_EQ(pose.x, dogm.getPositionX());
    EXPECT_EQ(pose.y, dogm.getPositionY());
    dogm::ParticlesSoA new_particles2 = dogm.getParticles();
    EXPECT_EQ(old_state, new_particles2.state[0]);

    // Update pose -> position update
    const float x_change = 3.0f;
    pose.x += x_change;
    dogm.updatePose(pose.x, pose.y);
    cudaDeviceSynchronize();
    EXPECT_EQ(pose.x, dogm.getPositionX());
    EXPECT_EQ(pose.y, dogm.getPositionY());
    /*
    Position is added here, because the change in updatePose is negative. In the moveParticleKernel the negative move
    is subtracted, which is the same as the addition here.
    */
    old_state.x += x_change;
    dogm::ParticlesSoA new_particles3 = dogm.getParticles();
    EXPECT_EQ(old_state, new_particles3.state[0]);

    particles.free();
    new_particles.free();
    new_particles2.free();
    new_particles3.free();
}

TEST(DOGM, Predict)
{
    dogm::DOGM::Params grid_params;
    grid_params.size = 10.0f;
    grid_params.resolution = 1.0f;
    grid_params.particle_count = 2;
    grid_params.new_born_particle_count = 1;
    grid_params.persistence_prob = 0.5f;
    grid_params.stddev_process_noise_position = 0.0f;
    grid_params.stddev_process_noise_velocity = 0.0f;
    grid_params.birth_prob = 0.02f;
    grid_params.stddev_velocity = 10.0f;
    grid_params.init_max_velocity = 30.0f;

    float delta_time = 0.1f;

    dogm::DOGM dogm(grid_params);
    cudaDeviceSynchronize();

    dogm::ParticlesSoA particles = dogm.getParticles();

    glm::vec4 old_state = particles.state[0];
    glm::vec4 pred_state = old_state + delta_time * glm::vec4(old_state[2], old_state[3], 0, 0);
    float old_weight = particles.weight[0];

    dogm.particlePrediction(delta_time);
    cudaDeviceSynchronize();

    dogm::ParticlesSoA new_particles = dogm.getParticles();

    EXPECT_EQ(pred_state, new_particles.state[0]);
    EXPECT_EQ(old_weight * grid_params.persistence_prob, new_particles.weight[0]);

    particles.free();
    new_particles.free();
}
