#include "kernel/measurement_grid.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PI 3.14159265358979323846f

__device__ float pFree(int i, float p_min, float p_max, int max_range)
{
	return p_min + i * (p_max - p_min) / max_range;
}

__device__ float pOcc(float r, float zk, float i)
{
	float alpha = 2.0; //0.6f * (1.0f - min(1.0, (1.0f / max_range) * zk));
	float delta = 0.9f; //1.f + 0.015f * zk;

	return (alpha / (delta * sqrt(2 * PI))) * exp(-pow(i - r, 2) / 2 * pow(delta, 2));
}

__device__ float inverse_sensor_model(int i, float resolution, float zk, float max_range)
{
	const int r_max = (int)(max_range / resolution);

	if (isfinite(zk))
	{
		int r = (int)(zk / resolution);

		if (i < r)
		{
			return max(pFree(i, 0.01, 0.4f, r_max), pOcc(r, zk, i));
		}

		return max(0.5f, pOcc(r, zk, i));
	}
	else
	{
		return pFree(i, 0.01, 0.4f, r_max);
	}
}

__device__ float2 probability_to_masses(float prob)
{
	// Masses: mOcc, mFree
	if (prob == 0.5f)
	{
		return make_float2(0.0f, 0.0f);
	}
	else
	{
		return make_float2(prob, 1.0f - prob);
	}
}

__global__ void createPolarGridMapKernel(cudaSurfaceObject_t polar, float* measurements, int width, int height, float resolution,
	float max_range)
{
	const int theta = blockIdx.x * blockDim.x + threadIdx.x;
	const int range = blockIdx.y * blockDim.y + threadIdx.y;

	if (theta < width && range < height)
	{
		const float epsilon = 0.00001f;
		const float zk = measurements[theta];

		float prob = inverse_sensor_model(range, resolution, zk, max_range);
		prob = max(epsilon, min(1.0f - epsilon, prob));

		surf2Dwrite(prob, polar, theta * sizeof(float), range);
	}
}

__global__ void cartesianGridToMeasurementGridKernel(MeasurementCell* meas_grid, cudaSurfaceObject_t cart, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index = (height - y - 1) * width + x;

	if (x < width && y < height)
	{
		float4 color = surf2Dread<float4>(cart, x * sizeof(float4), y);
		float2 masses = probability_to_masses(color.x);

		meas_grid[index].occ_mass = masses.x;
		meas_grid[index].free_mass = masses.y;

		meas_grid[index].likelihood = 1.0f;
		meas_grid[index].p_A = 1.0f;
	}
}
