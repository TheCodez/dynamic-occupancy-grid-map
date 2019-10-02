#include "kernel/mass_update.h"
#include "common.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float predict_free_mass(GridCell& grid_cell, float occPred, float alpha = 0.9f)
{
	return min(alpha * grid_cell.free_mass, 1.0f - occPred);
}

__device__ float update_o(float occ_pred, float free_pred, const MeasurementCell& meas)
{
	float unknown_pred = 1.0f - occ_pred - free_pred;
	float meas_cell_unknown = 1.0f - meas.free_mass - meas.occ_mass;
	float K = free_pred * meas.occ_mass + occ_pred * meas.free_mass;

	return (occ_pred * meas_cell_unknown + unknown_pred * meas.occ_mass + occ_pred * meas.occ_mass) / (1.0f - K);
}

__device__ float update_f(float occPred, float freePred, const MeasurementCell& meas)
{
	float unknown_pred = 1.0f - occPred - freePred;
	float meas_cell_unknown = 1.0f - meas.free_mass - meas.occ_mass;
	float K = freePred * meas.occ_mass + occPred * meas.free_mass;

	return (freePred * meas_cell_unknown + unknown_pred * meas.free_mass + freePred * meas.free_mass) / (1.0f - K);
}

__device__ float separate_newborn_part(float occPred, float occUp, float pb)
{
	return (occUp * pb * (1.0f - occPred)) / (occPred + pb * (1.0f - occPred));
}

__device__ void store_values(float rhoB, float rhoP, float freeUp, float occUp, GridCell* grid_cell_array, int i)
{
	grid_cell_array[i].pers_occ_mass = rhoP;
	grid_cell_array[i].new_born_occ_mass = rhoB;
	grid_cell_array[i].free_mass = freeUp;
}

__global__ void gridCellPredictionUpdateKernel(GridCell* grid_cell_array, float* weight_array_accum, MeasurementCell* meas_cell_array,
	float* born_masses_array, float pb, int cell_count)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < cell_count)
	{
		int start_idx = grid_cell_array[i].start_idx;
		int end_idx = grid_cell_array[i].end_idx;
		float occ_pred = subtract(weight_array_accum, start_idx, end_idx);
		float free_pred = predict_free_mass(grid_cell_array[i], occ_pred);
		float occ_up = update_o(occ_pred, free_pred, meas_cell_array[i]);
		float free_up = update_f(occ_pred, free_pred, meas_cell_array[i]);
		float rho_b = separate_newborn_part(occ_pred, occ_up, pb);
		float rho_p = occ_up - rho_b;
		born_masses_array[i] = rho_b;
		store_values(rho_b, rho_p, free_up, occ_up, grid_cell_array, i);
	}
}
