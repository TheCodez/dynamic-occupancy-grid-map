/*
MIT License

Copyright (c) 2019 Michael Kösel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#define GPU_LAMBDA [=] __host__ __device__

#define CHECK_ERROR(ans)                                                                                               \
    {                                                                                                                  \
        checkError((ans), __FILE__, __LINE__);                                                                         \
    }

inline void checkError(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        printf("GPU Kernel Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }

inline __device__ float curand_uniform(curandState* state, float min, float max)
{
    // Change from (0, 1] to [0, 1)
    return min + (max - min) * (1.0f - curand_uniform(state));
}

inline __device__ float curand_normal(curandState* state, float mean, float stddev)
{
    return curand_normal(state) * stddev + mean;
}
