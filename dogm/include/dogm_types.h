#pragma once

#include <glm/vec4.hpp>

struct GridCell
{
	int start_idx;
	int end_idx;
	float new_born_occ_mass;
	float pers_occ_mass;
	float free_mass;
	float occ_mass;
	float mu_A;
	float mu_UA;

	float w_A;
	float w_UA;

	float mean_x_vel;
	float mean_y_vel;
	float var_x_vel;
	float var_y_vel;
	float covar_xy_vel;
};

struct MeasurementCell
{
	float free_mass;
	float occ_mass;
	float likelihood;
	float p_A;
};

struct Particle
{
	int grid_cell_idx;
	float weight;
	bool associated;
	glm::vec4 state;
};