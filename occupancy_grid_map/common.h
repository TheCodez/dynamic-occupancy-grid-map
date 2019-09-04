#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

template <typename T>
__device__ __host__ thrust::device_vector<T> accumulate(T* arr)
{
	thrust::device_ptr<T> ptr(arr);
	thrust::device_vector<T> result;
	thrust::inclusive_scan(ptr, ptr + ARRAY_SIZE(array), result.begin());

	return result;
}

__device__ __host__ float subtract(float* accum_array, int start_idx, int end_idx)
{
	return accum_array[end_idx] - accum_array[start_idx];
}
