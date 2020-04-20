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

struct GridCellSoA
{
    int* start_idx;
    int* end_idx;
    float* new_born_occ_mass;
    float* pers_occ_mass;
    float* free_mass;
    float* occ_mass;
    float* mu_A;
    float* mu_UA;

    float* w_A;
    float* w_UA;

    float* mean_x_vel;
    float* mean_y_vel;
    float* var_x_vel;
    float* var_y_vel;
    float* covar_xy_vel;

    int size;

    GridCellSoA() : size(0) {}

    GridCellSoA(int size) : size(size) {}

    void init(int new_size)
    {
        size = new_size;
        CHECK_ERROR(cudaMalloc((void**)&start_idx, size * sizeof(int)));
        CHECK_ERROR(cudaMalloc((void**)&end_idx, size * sizeof(int)));

        CHECK_ERROR(cudaMalloc((void**)&new_born_occ_mass, size * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void**)&pers_occ_mass, size * sizeof(float)));
        CHECK_ERROR(cudaMallocManaged((void**)&free_mass, size * sizeof(float)));
        CHECK_ERROR(cudaMallocManaged((void**)&occ_mass, size * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void**)&mu_A, size * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void**)&mu_UA, size * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void**)&w_A, size * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void**)&w_UA, size * sizeof(float)));
        CHECK_ERROR(cudaMallocManaged((void**)&mean_x_vel, size * sizeof(float)));
        CHECK_ERROR(cudaMallocManaged((void**)&mean_y_vel, size * sizeof(float)));
        CHECK_ERROR(cudaMallocManaged((void**)&var_x_vel, size * sizeof(float)));
        CHECK_ERROR(cudaMallocManaged((void**)&var_y_vel, size * sizeof(float)));
        CHECK_ERROR(cudaMallocManaged((void**)&covar_xy_vel, size * sizeof(float)));
    }

    /*
    GridCellSoA& operator=(const GridCellSoA& other)
    {
        if (this != &other)
        {
            CHECK_ERROR(cudaMemcpy(grid_cell_idx, other.grid_cell_idx, size * sizeof(int), cudaMemcpyDeviceToDevice));
            CHECK_ERROR(cudaMemcpy(weight, other.weight, size * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        return *this;
    }
    */
};

struct ParticleSoA
{
    int* grid_cell_idx;
    float* weight;
    bool* associated;
    glm::vec4* state;

    int size;

    ParticleSoA() : size(0) {}

    ParticleSoA(int size) : size(size) {}

    void init(int new_size)
    {
        size = new_size;
        CHECK_ERROR(cudaMalloc((void**)&grid_cell_idx, size * sizeof(int)));
        CHECK_ERROR(cudaMalloc((void**)&weight, size * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void**)&associated, size * sizeof(bool)));
        CHECK_ERROR(cudaMallocManaged((void**)&state, size * sizeof(glm::vec4)));
    }

    ParticleSoA& operator=(const ParticleSoA& other)
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
};

} /* namespace dogm */
