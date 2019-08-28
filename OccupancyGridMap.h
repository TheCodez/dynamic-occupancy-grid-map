#pragma once

#include <thrust/device_vector.h>
#include <Eigen/Dense>

struct GridCell
{
	int startIdx;
	int endIdx;
	float newBornOccMass;
	float persOccMass;
	float freeMass;
	float occMass;
	float muA;
	float muUA;

	float wA;
	float wUA;

	int x;
	int y;

	float mean_x_vel;
	float mean_y_vel;
	float var_x_vel;
	float var_y_vel;
	float covar_xy_vel;
};

struct MeasurementCell
{
	float freeMass;
	float occMass;
	float likelihood;
	float pA;
};

struct Particle
{
	int gridCellIdx;
	float weight;

	Eigen::Vector4f state;

	bool associated;
};

struct GridParams
{
	int width;
	int height;
	int resolution;
	int v;
	int vB;
	float pS;
	float processNoisePosition;
	float processNoiseVelocity;
	float pB;
};

class OccupancyGridMap
{
public:
	OccupancyGridMap(const GridParams& params);
	~OccupancyGridMap();

	void update(float t);

private:

	void particlePrediction(float t);
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

	float pS;
	float pB;

	int vB;
};

