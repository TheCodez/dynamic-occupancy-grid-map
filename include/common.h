#pragma once

#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

template <typename T>
static __host__ thrust::device_vector<T> accumulate(T* arr)
{
	thrust::device_ptr<T> ptr(arr);
	thrust::device_vector<T> result;
	thrust::inclusive_scan(ptr, ptr + ARRAY_SIZE(arr), result.begin());

	return result;
}

static __device__ __host__ float subtract(float* accum_array, int start_idx, int end_idx)
{
	return accum_array[end_idx] - accum_array[start_idx];
}

template <typename T>
static __device__ __host__ T min(T a, T b)
{
	return a < b ? a : b;
}

template <typename T>
static __device__ __host__ T max(T a, T b)
{
	return a > b ? a : b;
}

template <typename T>
static __device__ __host__ T clamp(T a, T lower, T upper)
{
	return max(min(a, upper), lower);
}
