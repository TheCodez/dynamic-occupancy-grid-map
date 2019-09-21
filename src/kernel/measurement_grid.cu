#include "kernel/measurement_grid.h"

#include <device_launch_parameters.h>
//#include <math_constants.h>

#define CUDART_PI_F 3.14f

__device__ float pFree(float d, float dMin, float dMax)
{
	if (d >= 0 && d <= dMin)
	{
		return (0.4f * dMin) / dMax;
	}
	else if (d >= dMin && d <= dMax)
	{
		return (0.4f / dMax) * d;
	}

	return 0.5f;
}

__device__ float pOcc(float z, float mI)
{
	float alpha = 1.0; //0.6f * (1.0f - min(1.0, (1.0f / max_range) * z));
	float delta = 0.015f; //1.f + 0.015f * z;

	float nominator = -pow(mI - z, 2);
	float denominator = 2 * pow(delta, 2);

	return (alpha / (delta * sqrt(2 * CUDART_PI_F))) * exp(nominator / denominator);
}

__device__ float inverse_sensor_model(int i, float resolution, float z, float dMin, float dMax)
{
	int zk = (int)(z / resolution);

	if (i < zk)
	{
		return max(pFree(i, (int)(dMin / resolution), (int)(dMax / resolution)), pOcc(zk, i));
	}

	return max(0.5f, pOcc(zk, i));
}

__global__ void createPolarGridMapKernel(float2* polar, float* measurements, int width, int height, float resolution,
	float min_range, float max_range)
{
	const int theta = blockIdx.x * blockDim.x + threadIdx.x; // angle
	const int range = blockIdx.y * blockDim.y + threadIdx.y; // range
	const int index = range * width + theta;

	if (theta >= width || range >= height)
	{
		return;
	}

	const float epsilon = 0.00001;
	const float zk = measurements[theta];
	float prob = 0.0f;

	if (isfinite(zk))
	{
		prob = inverse_sensor_model(range, resolution, zk, min_range, max_range);
	}
	else
	{
		prob = pFree(range, (int)(min_range / resolution), (int)(max_range / resolution));
	}

	prob = max(epsilon, min(1.0f - epsilon, prob));

	// Masses: mOcc, mFree
	if (prob == 0.5f)
	{
		polar[index] = make_float2(0.0f, 0.0f);
	}
	else
	{
		polar[index] = make_float2(prob, 1.0f - prob);
	}
}

__global__ void polarToCartesianGridMapKernel(MeasurementCell* cartesian, float2* polar, int width, int height,
	int polar_w, int polar_h)
{
	const int _x = blockIdx.x * blockDim.x + threadIdx.x;
	const int _y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index = _y * width + _x;

	if (_x >= width || _y >= height)
	{
		return;
	}

	int x = _x - width / 2;
	int y = _y - height;

	const int theta = static_cast<int>((atan2((float)y, (float)x)) * polar_w + polar_w * 2 + CUDART_PI_F);// + 2045);// + 10;
	int r = static_cast<int>(sqrt((float)(x * x + y * y)));

	r = height - r;

	const int polar_index = r * polar_w + theta;


	if (theta >= polar_w || theta < 0 || r >= polar_h || r < 0)
	{
		cartesian[index].free_mass = 0.0f;
		cartesian[index].occ_mass = 0.0f;
		return;
	}

	float2 val = polar[polar_index];
	cartesian[index].occ_mass = val.x;
	cartesian[index].free_mass = val.y;

	cartesian[index].likelihood = 1.0f;
	cartesian[index].p_A = 1.0f;
}
