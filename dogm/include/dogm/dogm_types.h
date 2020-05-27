// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include "cuda_utils.h"
#include <glm/vec4.hpp>

namespace dogm
{

struct GridCell
{
    int start_idx;
    int end_idx;
    float new_born_occ_mass;
    float pers_occ_mass;
    float free_mass;
    float occ_mass;
    float pred_occ_mass;
    float mu_A;
    float mu_UA;

    float w_A;
    float w_UA;

    float mean_x_vel;
    float mean_y_vel;
    float var_x_vel;
    float var_y_vel;
    float covar_xy_vel;
};

struct MeasurementCell
{
    float free_mass;
    float occ_mass;
    float likelihood;
    float p_A;
};

struct Particle
{
    int grid_cell_idx;
    float weight;
    bool associated;
    glm::vec4 state;
};

struct ParticlesSoA
{
    glm::vec4* state;
    int* grid_cell_idx;
    float* weight;
    bool* associated;

    int size;
    bool device;

    ParticlesSoA() : size(0), device(true) {}

    ParticlesSoA(int new_size, bool is_device) { init(new_size, is_device); }

    void init(int new_size, bool is_device)
    {
        size = new_size;
        device = is_device;
        if (device)
        {
            CHECK_ERROR(cudaMalloc((void**)&state, size * sizeof(glm::vec4)));
            CHECK_ERROR(cudaMalloc((void**)&grid_cell_idx, size * sizeof(int)));
            CHECK_ERROR(cudaMalloc((void**)&weight, size * sizeof(float)));
            CHECK_ERROR(cudaMalloc((void**)&associated, size * sizeof(bool)));
        }
        else
        {
            state = (glm::vec4*)malloc(size * sizeof(glm::vec4));
            grid_cell_idx = (int*)malloc(size * sizeof(int));
            weight = (float*)malloc(size * sizeof(float));
            associated = (bool*)malloc(size * sizeof(bool));
        }
    }

    void free()
    {
        if (device)
        {
            CHECK_ERROR(cudaFree(state));
            CHECK_ERROR(cudaFree(grid_cell_idx));
            CHECK_ERROR(cudaFree(weight));
            CHECK_ERROR(cudaFree(associated));
        }
        else
        {
            ::free(state);
            ::free(grid_cell_idx);
            ::free(weight);
            ::free(associated);
        }
    }

    void copy(const ParticlesSoA& other, cudaMemcpyKind kind)
    {
        CHECK_ERROR(cudaMemcpy(grid_cell_idx, other.grid_cell_idx, size * sizeof(int), kind));
        CHECK_ERROR(cudaMemcpy(weight, other.weight, size * sizeof(float), kind));
        CHECK_ERROR(cudaMemcpy(associated, other.associated, size * sizeof(bool), kind));
        CHECK_ERROR(cudaMemcpy(state, other.state, size * sizeof(glm::vec4), kind));
    }

    ParticlesSoA& operator=(const ParticlesSoA& other)
    {
        if (this != &other)
        {
            copy(other, cudaMemcpyDeviceToDevice);
        }

        return *this;
    }

    __device__ void copy(const ParticlesSoA& other, int index, int other_index)
    {
        grid_cell_idx[index] = other.grid_cell_idx[other_index];
        weight[index] = other.weight[other_index];
        associated[index] = other.associated[other_index];
        state[index] = other.state[other_index];
    }
};

} /* namespace dogm */
