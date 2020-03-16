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

#include "dogm.h"
#include "cuda_utils.h"
#include <device_launch_parameters.h>

struct GridCell;
struct MeasurementCell;
struct Particle;

void normalize_particle_orders(thrust::device_vector<float>& particle_orders_array_accum, int particle_orders_count, int v_B);

__global__ void initNewParticlesKernel1(KernelArray<Particle> particle_array, KernelArray<GridCell> grid_cell_array,
	KernelArray<MeasurementCell> meas_cell_array, KernelArray<float> weight_array, KernelArray<float> born_masses_array,
	KernelArray<Particle> birth_particle_array, KernelArray<float> particle_orders_array_accum);

__global__ void initNewParticlesKernel2(KernelArray<Particle> birth_particle_array, KernelArray<GridCell> grid_cell_array, int grid_size);

__global__ void copyBirthWeightKernel(KernelArray<Particle> birth_particle_array, KernelArray<float> birth_weight_array);
