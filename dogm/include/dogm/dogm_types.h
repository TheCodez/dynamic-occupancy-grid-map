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
        CHECK_ERROR(cudaMalloc((void**)&state, size * sizeof(glm::vec4)));
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
