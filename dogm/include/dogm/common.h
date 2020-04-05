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
#include <thrust/device_vector.h>
#include <thrust/scan.h>

template <typename T> inline void accumulate(T* arr, thrust::device_vector<T>& result)
{
    thrust::device_ptr<T> ptr(arr);
    thrust::inclusive_scan(ptr, ptr + result.size(), result.begin());
}

template <typename T> inline void accumulate(thrust::device_vector<T>& arr, thrust::device_vector<T>& result)
{
    thrust::inclusive_scan(arr.begin(), arr.end(), result.begin());
}

template <typename T> inline __device__ __host__ T subtract(T* accum_array, int start_idx, int end_idx)
{
    if (start_idx == 0)
    {
        return accum_array[end_idx];
    }
    return accum_array[end_idx] - accum_array[start_idx - 1];
}

template <typename T> inline __device__ __host__ T clamp(T a, T lower, T upper) { return max(min(a, upper), lower); }

inline __host__ __device__ unsigned int hash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}
