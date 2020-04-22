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
    int* grid_cell_idx;
    float* weight;
    bool* associated;
    glm::vec4* state;

    int size;

    ParticlesSoA() : size(0) {}

    __host__ __device__ void init(int new_size)
    {
        size = new_size;
        CHECK_ERROR(cudaMalloc((void**)&grid_cell_idx, size * sizeof(int)));
        CHECK_ERROR(cudaMalloc((void**)&weight, size * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void**)&associated, size * sizeof(bool)));
        CHECK_ERROR(cudaMallocManaged((void**)&state, size * sizeof(glm::vec4)));
    }

    __host__ __device__ void free()
    {
        CHECK_ERROR(cudaFree(grid_cell_idx));
        CHECK_ERROR(cudaFree(weight));
        CHECK_ERROR(cudaFree(associated));
        CHECK_ERROR(cudaFree(state));
    }

    __host__ __device__ ParticlesSoA& operator=(const ParticlesSoA& other)
    {
        if (this != &other)
        {
            CHECK_ERROR(cudaMemcpy(grid_cell_idx, other.grid_cell_idx, size * sizeof(int), cudaMemcpyDeviceToDevice));
            CHECK_ERROR(cudaMemcpy(weight, other.weight, size * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_ERROR(cudaMemcpy(associated, other.associated, size * sizeof(bool), cudaMemcpyDeviceToDevice));
            CHECK_ERROR(cudaMemcpy(state, other.state, size * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));
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
