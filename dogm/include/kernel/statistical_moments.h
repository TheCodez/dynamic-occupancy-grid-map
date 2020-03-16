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

#include "cuda_utils.h"
#include <device_launch_parameters.h>

struct GridCell;
struct Particle;

__global__ void statisticalMomentsKernel1(KernelArray<Particle> particle_array, KernelArray<float> weight_array,
	KernelArray<float> vel_x_array, KernelArray<float> vel_y_array, KernelArray<float> vel_x_squared_array,
	KernelArray<float> vel_y_squared_array, KernelArray<float> vel_xy_array);

__global__ void statisticalMomentsKernel2(KernelArray<GridCell> grid_cell_array, KernelArray<float> vel_x_array_accum,
	KernelArray<float> vel_y_array_accum, KernelArray<float> vel_x_squared_array_accum, KernelArray<float> vel_y_squared_array_accum,
	KernelArray<float> vel_xy_array_accum);

