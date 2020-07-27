// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/dogm_types.h"
#include "mapping/kernel/measurement_grid.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PI 3.14159265358979323846f

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

__device__ float2 multi_inverse_sensor_model(int cell_index, int theta, float resolution, float* meas_points,
                                             int num_meas_points, float r_max)
{
    // Masses: mOcc, mFree
    float m_free = 0.0f, m_occ = 0.0f;
    float cell_pos = cell_index * resolution;

    // if cell is inside sensor range
    if (cell_pos < r_max)
    {
        float delta = 0.2f;
        float min_range = meas_points[theta * num_meas_points];

        for (int j = 0; j < num_meas_points; j++)
        {
            // find min range measurement
            if (meas_points[theta * num_meas_points + j] < min_range)
            {
                min_range = meas_points[theta * num_meas_points + j];
            }

            float zk = meas_points[theta * num_meas_points + j];
            float occ = 0.95f * exp(-0.5f * (cell_pos - zk) * (cell_pos - zk) / (delta * delta));
            m_occ = max(occ, m_occ);
        }

        if (cell_pos < min_range)
        {
            float free = 1.0f - pFree(cell_index, 0.2f, 1.0f, r_max);
            m_free = max(free - m_occ, 0.0f);
        }
    }
    else
    {
        m_occ = 0.0f;
        m_free = 0.0f;
    }

    return make_float2(m_occ, m_free);
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

__global__ void createPolarGridTextureKernel(cudaSurfaceObject_t polar, float* __restrict__ measurements, int width,
                                             int height, float resolution)
{
    const int theta = blockIdx.x * blockDim.x + threadIdx.x;
    const int range = blockIdx.y * blockDim.y + threadIdx.y;

    if (theta < width && range < height)
    {
        const float epsilon = 0.00001f;
        const float zk = measurements[theta];
        const int num_meas_points = 64;
        // float* meas_points = measurements[theta];//[num_meas_points] = {zk, zk + 5.0f, zk + 8.0f};

        float2 masses = multi_inverse_sensor_model(range, theta, resolution, measurements, num_meas_points, height);
        masses.x = max(epsilon, min(1.0f - epsilon, masses.x));
        masses.y = max(epsilon, min(1.0f - epsilon, masses.y));

        surf2Dwrite(masses, polar, theta * sizeof(float2), range);
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
