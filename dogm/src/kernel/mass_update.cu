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
#include "dogm_types.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dogm
{

__device__ float predict_free_mass(GridCell& grid_cell, float m_occ_pred, float alpha = 0.9)
{
	float m_free_pred = min(alpha * grid_cell.free_mass, 1.0 - m_occ_pred);

	// limit free mass
	float sum = m_free_pred + m_occ_pred;
	if (sum > 1.0)
	{
		//printf("Sum of masses exceeds 1. Limiting free mass.\n");
		float diff = sum - 1.0;
		m_free_pred -= diff;
	}

	return m_free_pred;
}

__device__ float update_o(float m_occ_pred, float m_free_pred, const MeasurementCell& meas_cell)
{
	float unknown_pred = 1.0 - m_occ_pred - m_free_pred;
	float meas_cell_unknown = 1.0 - meas_cell.free_mass - meas_cell.occ_mass;
	float K = m_free_pred * meas_cell.occ_mass + m_occ_pred * meas_cell.free_mass;

	return (m_occ_pred * meas_cell_unknown + unknown_pred * meas_cell.occ_mass + m_occ_pred * meas_cell.occ_mass) / (1.0 - K);
}

__device__ float update_f(float m_occ_pred, float m_free_pred, const MeasurementCell& meas_cell)
{
	float unknown_pred = 1.0 - m_occ_pred - m_free_pred;
	float meas_cell_unknown = 1.0 - meas_cell.free_mass - meas_cell.occ_mass;
	float K = m_free_pred * meas_cell.occ_mass + m_occ_pred * meas_cell.free_mass;

	return (m_free_pred * meas_cell_unknown + unknown_pred * meas_cell.free_mass + m_free_pred * meas_cell.free_mass) / (1.0 - K);
}

__device__ float separate_newborn_part(float m_occ_pred, float m_occ_up, float p_B)
{
	return (m_occ_up * p_B * (1.0 - m_occ_pred)) / (m_occ_pred + p_B * (1.0 - m_occ_pred));
}

__device__ void store_values(float rho_b, float rho_p, float m_free_up, float m_occ_up, GridCell* grid_cell_array, int i)
{
	grid_cell_array[i].pers_occ_mass = rho_p;
	grid_cell_array[i].new_born_occ_mass = rho_b;
	grid_cell_array[i].free_mass = m_free_up;
	grid_cell_array[i].occ_mass = m_occ_up;
}

__device__ void normalize_to_pS(Particle* particle_array, float* weight_array, float p_S, int start_idx, int end_idx)
{
	float sum = 0.0f;
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

__global__ void gridCellPredictionUpdateKernel(GridCell* grid_cell_array, Particle* particle_array, float* weight_array,
	float* weight_array_accum, MeasurementCell* meas_cell_array, float* born_masses_array, float p_B, float p_S, int cell_count)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
	{
		int start_idx = grid_cell_array[i].start_idx;
		int end_idx = grid_cell_array[i].end_idx;

		if (start_idx != -1)
		{
			float m_occ_pred = subtract(weight_array_accum, start_idx, end_idx);

			if (m_occ_pred > p_S)
			{
				//printf("Predicted mass greater pS. Mass is: %f\n", m_occ_pred);
				m_occ_pred = p_S;
				normalize_to_pS(particle_array, weight_array, p_S, start_idx, end_idx);
			}

			float m_free_pred = predict_free_mass(grid_cell_array[i], m_occ_pred);
			float m_occ_up = update_o(m_occ_pred, m_free_pred, meas_cell_array[i]);
			float m_free_up = update_f(m_occ_pred, m_free_pred, meas_cell_array[i]);
			float rho_b = separate_newborn_part(m_occ_pred, m_occ_up, p_B);
			float rho_p = m_occ_up - rho_b;
			born_masses_array[i] = rho_b;

			//printf("Rho B: %f\n", rho_b);

			store_values(rho_b, rho_p, m_free_up, m_occ_up, grid_cell_array, i);
		}
		else
		{
			float m_occ = grid_cell_array[i].occ_mass;
			float m_free = predict_free_mass(grid_cell_array[i], m_occ);
			float m_occ_up = update_o(m_occ, m_free, meas_cell_array[i]);
			float m_free_up = update_f(m_occ, m_free, meas_cell_array[i]);
			born_masses_array[i] = 0.0f;
			store_values(0.0f, m_occ_up, m_free_up, m_occ_up, grid_cell_array, i);
		}
	}
}

} /* namespace dogm */
