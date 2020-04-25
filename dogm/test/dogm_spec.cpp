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

    glm::vec4 old_state = dogm.particle_array.state[0];
    glm::vec4 pred_state = old_state + glm::vec4(old_state[2] * delta_time, old_state[3] * delta_time, 0, 0);
    float old_weight = dogm.particle_array.weight[0];

    dogm.particlePrediction(delta_time);
    cudaDeviceSynchronize();

    EXPECT_EQ(pred_state, dogm.particle_array.state[0]);
    EXPECT_EQ(old_weight * grid_params.persistence_prob, dogm.particle_array.weight[0]);
}
