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

#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

#define GPU_LAMBDA [=] __host__ __device__

#define CHECK_ERROR(ans) { checkError((ans), __FILE__, __LINE__); }

static __host__ void checkError(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		printf("GPU Kernel Error: %s %s %d\n", cudaGetErrorString(code), file, line);
	}
}

static inline int divUp(int total, int grain)
{
	return (total + grain - 1) / grain;
}

template <typename T>
class KernelArray
{
public:
	KernelArray(thrust::device_vector<T>& vector)
	{
		data = thrust::raw_pointer_cast(vector.data());
		length = static_cast<int>(vector.size());
	}

	__device__ __host__ T* get() const { return data; }
	__device__ __host__ int size() const { return length; }

	__device__ __host__ T& operator [] (int index) { return data[index]; }
	__device__ __host__ T operator [] (int index) const { return data[index]; }

private:
	T* data;
	int length;
};
