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
#include "kernel/mass_update.h"
#include "common.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/execution_policy.h>

__device__ double predict_free_mass(GridCell& grid_cell, double m_occ_pred, double alpha = 0.9)
{
	double m_free_pred = min(alpha * grid_cell.free_mass, 1.0 - m_occ_pred);

	// limit free mass
	double sum = m_free_pred + m_occ_pred;
	if (sum > 1.0)
	{
		double diff = sum - 1.0;
		m_free_pred -= diff;
	}

	return m_free_pred;
}

__device__ double update_o(double m_occ_pred, double m_free_pred, const MeasurementCell& meas_cell)
{
	double unknown_pred = 1.0 - m_occ_pred - m_free_pred;
	double meas_cell_unknown = 1.0 - meas_cell.free_mass - meas_cell.occ_mass;
	double K = m_free_pred * meas_cell.occ_mass + m_occ_pred * meas_cell.free_mass;

	return (m_occ_pred * meas_cell_unknown + unknown_pred * meas_cell.occ_mass + m_occ_pred * meas_cell.occ_mass) / (1.0 - K);
}

__device__ double update_f(double m_occ_pred, double m_free_pred, const MeasurementCell& meas_cell)
{
	double unknown_pred = 1.0 - m_occ_pred - m_free_pred;
	double meas_cell_unknown = 1.0 - meas_cell.free_mass - meas_cell.occ_mass;
	double K = m_free_pred * meas_cell.occ_mass + m_occ_pred * meas_cell.free_mass;

	return (m_free_pred * meas_cell_unknown + unknown_pred * meas_cell.free_mass + m_free_pred * meas_cell.free_mass) / (1.0 - K);
}

__device__ double separate_newborn_part(double m_occ_pred, double m_occ_up, double p_B)
{
	return (m_occ_up * p_B * (1.0 - m_occ_pred)) / (m_occ_pred + p_B * (1.0 - m_occ_pred));
}

__device__ void store_values(double rho_b, double rho_p, double m_free_up, double m_occ_up, GridCell* grid_cell_array, int i)
{
	grid_cell_array[i].pers_occ_mass = rho_p;
	grid_cell_array[i].new_born_occ_mass = rho_b;
	grid_cell_array[i].free_mass = m_free_up;
	grid_cell_array[i].occ_mass = m_occ_up;
}

__device__ void normalize_to_pS(Particle* particle_array, double* weight_array, double p_S, int start_idx, int end_idx)
{
	double sum = 0.0f;
	for (int i = start_idx; i < end_idx + 1; i++)
	{
		sum += weight_array[i];
	}

	for (int i = start_idx; i < end_idx + 1; i++)
	{
		weight_array[i] = weight_array[i] / sum * p_S;
		particle_array[i].weight = weight_array[i];
	}
}

__global__ void gridCellPredictionUpdateKernel(GridCell* grid_cell_array, Particle* particle_array, double* weight_array,
	double* weight_array_accum, MeasurementCell* meas_cell_array, double* born_masses_array, double p_B, double p_S, int cell_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
	{
		int start_idx = grid_cell_array[i].start_idx;
		int end_idx = grid_cell_array[i].end_idx;

		double m_occ_pred = subtract(weight_array_accum, start_idx, end_idx);

		if (m_occ_pred > p_S)
		{
			m_occ_pred = p_S;
			normalize_to_pS(particle_array, weight_array, p_S, start_idx, end_idx);
		}

		double m_free_pred = predict_free_mass(grid_cell_array[i], m_occ_pred);
		double m_occ_up = update_o(m_occ_pred, m_free_pred, meas_cell_array[i]);
		double m_free_up = update_f(m_occ_pred, m_free_pred, meas_cell_array[i]);
		double rho_b = separate_newborn_part(m_occ_pred, m_occ_up, p_B);
		double rho_p = m_occ_up - rho_b;
		born_masses_array[i] = rho_b;
		store_values(rho_b, rho_p, m_free_up, m_occ_up, grid_cell_array, i);
	}
}
