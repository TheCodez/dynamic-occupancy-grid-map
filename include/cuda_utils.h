#pragma once

#include "cuda_runtime.h"
#include <stdio.h>
#include <thrust/random.h>

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

__host__ __device__ inline unsigned int hash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}
