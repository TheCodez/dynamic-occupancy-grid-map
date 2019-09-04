#pragma once

#include <thrust/device_vector.h>
#include <Eigen/Dense>

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
	Eigen::Vector4f state;
};

struct GridParams
{
	int width;
	int height;
	int resolution;
	int v;
	int vb;
	float ps;
	float process_noise_position;
	float process_noise_velocity;
	float pb;
};

class OccupancyGridMap
{
public:
	OccupancyGridMap(const GridParams& params);
	~OccupancyGridMap();

	void update(float dt);

private:

	void initialize();

	void particlePrediction(float dt);
	void particleAssignment();
	void gridCellOccupancyUpdate();
	void updatePersistentParticles();
	void initializeNewParticles();
	void statisticalMoments();
	void resampling();

private:

	GridParams params;

	GridCell* grid_cell_array;
	Particle* particle_array;
	Particle* birth_particle_array;

	float* weight_array;
	float* birth_weight_array;
	MeasurementCell* meas_cell_array;
	float* born_masses_array;
	float* particle_orders_array_accum;

	float* vel_x_array;
	float* vel_y_array;

	float* vel_x_squared_array;
	float* vel_y_squared_array;
	float* vel_xy_array;

	float* rand_array;
};
