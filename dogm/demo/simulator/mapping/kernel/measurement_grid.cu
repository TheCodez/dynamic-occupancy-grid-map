// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/dogm_types.h"
#include "mapping/kernel/measurement_grid.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PI 3.14159265358979323846f

__device__ float2 combine_masses(float2 prior, float2 meas)
{
    // Masses: mOcc, mFree
    float occ = prior.x;
    float free = prior.y;

    float meas_occ = meas.x;
    float meas_free = meas.y;

    float unknown_pred = 1.0f - occ - free;
    float meas_cell_unknown = 1.0f - meas_occ - meas_free;
    float K = free * meas_occ + occ * meas_free;

    float2 res;
    res.x = (occ * meas_cell_unknown + unknown_pred * meas_occ + occ * meas_occ) / (1.0f - K);
    res.y = (free * meas_cell_unknown + unknown_pred * meas_free + free * meas_free) / (1.0f - K);

    return res;
}

__device__ float pFree(int i, float p_min, float p_max, int max_range)
{
    return p_min + i * (p_max - p_min) / max_range;
}

__device__ float pOcc(int r, float zk, int index, float resolution)
{
    float occ_max = 0.95f;
    float delta = 0.6f / resolution;

    return occ_max * exp(-0.5f * (index - r) * (index - r) / (delta * delta));
}

__device__ float2 inverse_sensor_model(int i, float resolution, float zk, float r_max)
{
    // Masses: mOcc, mFree

    const float free = pFree(i, 0.15f, 1.0f, r_max);

    if (isfinite(zk))
    {
        const int r = static_cast<int>(zk / resolution);
        const float occ = pOcc(r, zk, i, resolution);

        if (i <= r)
        {
            return occ > free ? make_float2(occ, 0.0f) : make_float2(0.0f, 1.0f - free);
        }
        else
        {
            return occ > 0.5f ? make_float2(occ, 0.0f) : make_float2(0.0f, 0.0f);
        }
    }
    else
    {
        return make_float2(0.0f, 1.0f - free);
    }
}

__global__ void createPolarGridTextureKernel(cudaSurfaceObject_t polar, const float* __restrict__ measurements,
                                             int width, int height, float resolution)
{
    const int theta = blockIdx.x * blockDim.x + threadIdx.x;
    const int range = blockIdx.y * blockDim.y + threadIdx.y;

    if (theta < width && range < height)
    {
        const float epsilon = 0.00001f;
        const float zk = measurements[theta];

        float2 masses = inverse_sensor_model(range, resolution, zk, height);
        masses.x = max(epsilon, min(1.0f - epsilon, masses.x));
        masses.y = max(epsilon, min(1.0f - epsilon, masses.y));

        surf2Dwrite(masses, polar, theta * sizeof(float2), range);
    }
}

__global__ void fusePolarGridTextureKernel(cudaSurfaceObject_t polar, const float* __restrict__ measurements, int width,
                                           int height, float resolution)
{
    const int theta = blockIdx.x * blockDim.x + threadIdx.x;
    const int range = blockIdx.y * blockDim.y + threadIdx.y;

    if (theta < width && range < height)
    {
        const float epsilon = 0.00001f;
        const float zk = measurements[theta];

        float2 prior = surf2Dread<float2>(polar, theta * sizeof(float2), range);
        float2 masses = inverse_sensor_model(range, resolution, zk, height);
        masses.x = max(epsilon, min(1.0f - epsilon, masses.x));
        masses.y = max(epsilon, min(1.0f - epsilon, masses.y));

        float2 new_masses = combine_masses(prior, masses);
        // new_masses.x = max(epsilon, min(1.0f - epsilon, new_masses.x));
        // new_masses.y = max(epsilon, min(1.0f - epsilon, new_masses.y));

        surf2Dwrite(new_masses, polar, theta * sizeof(float2), range);
    }
}

__global__ void cartesianGridToMeasurementGridKernel(dogm::MeasurementCell* __restrict__ meas_grid,
                                                     cudaSurfaceObject_t cart, int grid_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int index = (grid_size - y - 1) * grid_size + x;

    if (x < grid_size && y < grid_size)
    {
        float4 color = surf2Dread<float4>(cart, x * sizeof(float4), y);

        meas_grid[index].occ_mass = color.x;
        meas_grid[index].free_mass = color.y;

        meas_grid[index].likelihood = 1.0f;
        meas_grid[index].p_A = 1.0f;
    }
}

__global__ void gridArrayToMeasurementGridKernel(dogm::MeasurementCell* __restrict__ meas_grid,
                                                 const float2* __restrict__ grid, int grid_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int index = grid_size * y + x;

    if (x < grid_size && y < grid_size)
    {
        float2 masses = grid[index];

        meas_grid[index].occ_mass = masses.x;
        meas_grid[index].free_mass = masses.y;

        meas_grid[index].likelihood = 1.0f;
        meas_grid[index].p_A = 1.0f;
    }
}
