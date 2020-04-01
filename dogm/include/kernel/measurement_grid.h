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

#include <device_launch_parameters.h>

namespace dogm
{

struct MeasurementCell;

__global__ void createPolarGridTextureKernel(cudaSurfaceObject_t polar, const float* __restrict__ measurements, int width, int height,
	float resolution);

__global__ void createPolarGridTextureKernel2(cudaSurfaceObject_t polar, MeasurementCell* __restrict__ polar_meas_grid, 
	const float* __restrict__ measurements, int width, int height, float resolution);

__global__ void fusePolarGridTextureKernel(cudaSurfaceObject_t polar, const float* __restrict__ measurements, int width, int height,
	float resolution);

__global__ void cartesianGridToMeasurementGridKernel(MeasurementCell* __restrict__ meas_grid, cudaSurfaceObject_t cart, int grid_size);

__global__ void gridArrayToMeasurementGridKernel(MeasurementCell* __restrict__ meas_grid, const float2* __restrict__ grid, int grid_size);

} /* namespace dogm */
