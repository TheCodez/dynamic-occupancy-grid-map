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
    struct
    {
        glm::vec4* state;
        int* grid_cell_idx;
        float* weight;
        bool* associated;
    };

    void* memory_block;
    int size;
    bool device;

    ParticlesSoA() : size(0), device(true) {}

    ParticlesSoA(int new_size, bool is_device) { init(new_size, is_device); }

    void init(int new_size, bool is_device)
    {
        size = new_size;
        device = is_device;
        const size_t num_bytes = size * sizeof(Particle);

        if (device)
        {
            CHECK_ERROR(cudaMalloc((void**)&memory_block, num_bytes));
        }
        else
        {
            memory_block = malloc(num_bytes);
        }

        assignPointers();
    }

    void free()
    {
        assert(size);
        if (device)
        {
            CHECK_ERROR(cudaFree(memory_block));
        }
        else
        {
            ::free(memory_block);
        }

        memory_block = nullptr;
        size = 0;
    }

    void copy(const ParticlesSoA& other)
    {
        assert(size && size == other.size);

        cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
        if (device != other.device)
        {
            kind = device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        }

        const size_t num_bytes = size * sizeof(Particle);
        CHECK_ERROR(cudaMemcpy(memory_block, other.memory_block, num_bytes, kind));
        assignPointers();
    }

    ParticlesSoA& operator=(const ParticlesSoA& other)
    {
        if (this != &other)
        {
            copy(other);
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

private:
    void assignPointers()
    {
        state = reinterpret_cast<glm::vec4*>(memory_block);
        grid_cell_idx = reinterpret_cast<int*>(state + size);
        weight = reinterpret_cast<float*>(grid_cell_idx + size);
        associated = reinterpret_cast<bool*>(weight + size);
    }
};

} /* namespace dogm */
