// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#define GPU_LAMBDA [=] __host__ __device__

#ifndef CUDA_CALL
#define CUDA_CALL(call)\
{\
    auto status = static_cast<cudaError_t>(call);\
    if (status != cudaSuccess)\
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n",\
            #call, __LINE__, __FILE__, cudaGetErrorString(status), status);\
}
#endif

inline int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

inline __device__ float curand_uniform(curandState* state, float min, float max)
{
    // Change from (0, 1] to [0, 1)
    return min + (max - min) * (1.0f - curand_uniform(state));
}

inline __device__ float curand_normal(curandState* state, float mean, float stddev)
{
    return curand_normal(state) * stddev + mean;
}
