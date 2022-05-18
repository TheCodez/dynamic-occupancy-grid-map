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

struct GridCellsSoA
{
    int* start_idx;
    int* end_idx;
    float* new_born_occ_mass;
    float* pers_occ_mass;
    float* free_mass;
    float* occ_mass;
    float* pred_occ_mass;
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
    bool device;

    GridCellsSoA() : size(0), device(true) {}

    GridCellsSoA(int new_size, bool is_device) { init(new_size, is_device); }

    void init(int new_size, bool is_device)
    {
        size = new_size;
        device = is_device;
        if (device)
        {
            CUDA_CALL(cudaMalloc((void**)&start_idx, size * sizeof(int)));
            CUDA_CALL(cudaMalloc((void**)&end_idx, size * sizeof(int)));
            CUDA_CALL(cudaMalloc((void**)&new_born_occ_mass, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&pers_occ_mass, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&free_mass, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&occ_mass, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&pred_occ_mass, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&mu_A, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&mu_UA, size * sizeof(float)));

            CUDA_CALL(cudaMalloc((void**)&w_A, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&w_UA, size * sizeof(float)));

            CUDA_CALL(cudaMalloc((void**)&mean_x_vel, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&mean_y_vel, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&var_x_vel, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&var_y_vel, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&covar_xy_vel, size * sizeof(float)));
        }
        else
        {
            start_idx = (int*)malloc(size * sizeof(int));
            end_idx = (int*)malloc(size * sizeof(int));
            new_born_occ_mass = (float*)malloc(size * sizeof(float));
            pers_occ_mass = (float*)malloc(size * sizeof(float));
            free_mass = (float*)malloc(size * sizeof(float));
            occ_mass = (float*)malloc(size * sizeof(float));
            pred_occ_mass = (float*)malloc(size * sizeof(float));
            mu_A = (float*)malloc(size * sizeof(float));
            mu_UA = (float*)malloc(size * sizeof(float));

            w_A = (float*)malloc(size * sizeof(float));
            w_UA = (float*)malloc(size * sizeof(float));

            mean_x_vel = (float*)malloc(size * sizeof(float));
            mean_y_vel = (float*)malloc(size * sizeof(float));
            var_x_vel = (float*)malloc(size * sizeof(float));
            var_y_vel = (float*)malloc(size * sizeof(float));
            covar_xy_vel = (float*)malloc(size * sizeof(float));
        }
    }

    void free()
    {
        if (device)
        {
            CUDA_CALL(cudaFree(start_idx));
            CUDA_CALL(cudaFree(end_idx));
            CUDA_CALL(cudaFree(new_born_occ_mass));
            CUDA_CALL(cudaFree(pers_occ_mass));
            CUDA_CALL(cudaFree(free_mass));
            CUDA_CALL(cudaFree(occ_mass));
            CUDA_CALL(cudaFree(pred_occ_mass));
            CUDA_CALL(cudaFree(mu_A));
            CUDA_CALL(cudaFree(mu_UA));

            CUDA_CALL(cudaFree(w_A));
            CUDA_CALL(cudaFree(w_UA));

            CUDA_CALL(cudaFree(mean_x_vel));
            CUDA_CALL(cudaFree(mean_y_vel));
            CUDA_CALL(cudaFree(var_x_vel));
            CUDA_CALL(cudaFree(var_y_vel));
            CUDA_CALL(cudaFree(covar_xy_vel));
        }
        else
        {
            ::free(start_idx);
            ::free(end_idx);
            ::free(new_born_occ_mass);
            ::free(pers_occ_mass);
            ::free(free_mass);
            ::free(occ_mass);
            ::free(pred_occ_mass);
            ::free(mu_A);
            ::free(mu_UA);

            ::free(w_A);
            ::free(w_UA);

            ::free(mean_x_vel);
            ::free(mean_y_vel);
            ::free(var_x_vel);
            ::free(var_y_vel);
            ::free(covar_xy_vel);
        }
    }

    void copy(const GridCellsSoA& other, cudaMemcpyKind kind)
    {
        CUDA_CALL(cudaMemcpy(start_idx, other.start_idx, size * sizeof(int), kind));
        CUDA_CALL(cudaMemcpy(end_idx, other.end_idx, size * sizeof(int), kind));
        CUDA_CALL(cudaMemcpy(new_born_occ_mass, other.new_born_occ_mass, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(pers_occ_mass, other.pers_occ_mass, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(free_mass, other.free_mass, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(occ_mass, other.occ_mass, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(pred_occ_mass, other.pred_occ_mass, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(mu_A, other.mu_A, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(mu_UA, other.mu_UA, size * sizeof(float), kind));

        CUDA_CALL(cudaMemcpy(w_A, other.w_A, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(w_UA, other.w_UA, size * sizeof(float), kind));

        CUDA_CALL(cudaMemcpy(mean_x_vel, other.mean_x_vel, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(mean_y_vel, other.mean_y_vel, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(var_x_vel, other.var_x_vel, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(var_y_vel, other.var_y_vel, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(covar_xy_vel, other.covar_xy_vel, size * sizeof(float), kind));
    }

    GridCellsSoA& operator=(const GridCellsSoA& other)
    {
        if (this != &other)
        {
            copy(other, cudaMemcpyDeviceToDevice);
        }

        return *this;
    }

    __device__ void copy(const GridCellsSoA& other, int index, int other_index)
    {
        start_idx[index] = other.start_idx[other_index];
        end_idx[index] = other.end_idx[other_index];
        new_born_occ_mass[index] = other.new_born_occ_mass[other_index];
        pers_occ_mass[index] = other.pers_occ_mass[other_index];
        free_mass[index] = other.free_mass[other_index];
        occ_mass[index] = other.occ_mass[other_index];
        pred_occ_mass[index] = other.pred_occ_mass[other_index];
        mu_A[index] = other.mu_A[other_index];
        mu_UA[index] = other.mu_UA[other_index];

        w_A[index] = other.w_A[other_index];
        w_UA[index] = other.w_UA[other_index];

        mean_x_vel[index] = other.mean_x_vel[other_index];
        mean_y_vel[index] = other.mean_y_vel[other_index];
        var_x_vel[index] = other.var_x_vel[other_index];
        var_y_vel[index] = other.var_y_vel[other_index];
        covar_xy_vel[index] = other.covar_xy_vel[other_index];
    }
};

struct MeasurementCellsSoA
{
    float* free_mass;
    float* occ_mass;
    float* likelihood;
    float* p_A;

    int size;
    bool device;

    MeasurementCellsSoA() : size(0), device(true) {}

    MeasurementCellsSoA(int new_size, bool is_device) { init(new_size, is_device); }

    void init(int new_size, bool is_device)
    {
        size = new_size;
        device = is_device;
        if (device)
        {
            CUDA_CALL(cudaMalloc((void**)&free_mass, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&occ_mass, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&likelihood, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&p_A, size * sizeof(float)));
        }
        else
        {
            free_mass = (float*)malloc(size * sizeof(float));
            occ_mass = (float*)malloc(size * sizeof(float));
            likelihood = (float*)malloc(size * sizeof(float));
            p_A = (float*)malloc(size * sizeof(float));
        }
    }

    void free()
    {
        if (device)
        {
            CUDA_CALL(cudaFree(free_mass));
            CUDA_CALL(cudaFree(occ_mass));
            CUDA_CALL(cudaFree(likelihood));
            CUDA_CALL(cudaFree(p_A));
        }
        else
        {
            ::free(free_mass);
            ::free(occ_mass);
            ::free(likelihood);
            ::free(p_A);
        }
    }

    void copy(const MeasurementCellsSoA& other, cudaMemcpyKind kind)
    {
        CUDA_CALL(cudaMemcpy(free_mass, other.free_mass, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(occ_mass, other.occ_mass, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(likelihood, other.likelihood, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(p_A, other.p_A, size * sizeof(float), kind));
    }

    MeasurementCellsSoA& operator=(const MeasurementCellsSoA& other)
    {
        if (this != &other)
        {
            copy(other, cudaMemcpyDeviceToDevice);
        }

        return *this;
    }

    __device__ void copy(const MeasurementCellsSoA& other, int index, int other_index)
    {
        free_mass[index] = other.free_mass[other_index];
        occ_mass[index] = other.occ_mass[other_index];
        likelihood[index] = other.likelihood[other_index];
        p_A[index] = other.p_A[other_index];
    }
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
            CUDA_CALL(cudaMalloc((void**)&state, size * sizeof(glm::vec4)));
            CUDA_CALL(cudaMalloc((void**)&grid_cell_idx, size * sizeof(int)));
            CUDA_CALL(cudaMalloc((void**)&weight, size * sizeof(float)));
            CUDA_CALL(cudaMalloc((void**)&associated, size * sizeof(bool)));
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
            CUDA_CALL(cudaFree(state));
            CUDA_CALL(cudaFree(grid_cell_idx));
            CUDA_CALL(cudaFree(weight));
            CUDA_CALL(cudaFree(associated));
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
        CUDA_CALL(cudaMemcpy(grid_cell_idx, other.grid_cell_idx, size * sizeof(int), kind));
        CUDA_CALL(cudaMemcpy(weight, other.weight, size * sizeof(float), kind));
        CUDA_CALL(cudaMemcpy(associated, other.associated, size * sizeof(bool), kind));
        CUDA_CALL(cudaMemcpy(state, other.state, size * sizeof(glm::vec4), kind));
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
