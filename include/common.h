#pragma once

#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

template <typename T>
static void accumulate(T* arr, thrust::device_vector<T>& result)
{
	thrust::device_ptr<T> ptr(arr);
	thrust::inclusive_scan(ptr, ptr + result.size(), result.begin());
}

static __device__ __host__ float subtract(float* accum_array, int start_idx, int end_idx)
{
	return accum_array[end_idx] - accum_array[start_idx - 1];
}

template <typename T>
static __device__ __host__ T clamp(T a, T lower, T upper)
{
	return max(min(a, upper), lower);
}
