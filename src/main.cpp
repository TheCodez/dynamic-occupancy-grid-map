#include "occupancy_grid_map.h"

int main(int argc, const char** argv) 
{
	GridParams params;
	params.width = 80;//120;
	params.height = 80;//120;
	params.resolution = 0.1f;
	params.particle_count = 300000;//2 * static_cast<int>(10e6);
	params.new_born_particle_count = 200000;//2 * static_cast<int>(10e5);
	params.ps = 0.99f;
	params.process_noise_position = 0.02f;
	params.process_noise_velocity = 0.8f;
	params.pb = 0.02f;

	OccupancyGridMap grid_map(params);

	float ranges[] = {
		25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f,
		37.5f, 37.5f, 37.5f,
		25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f,
		45.0f,
		25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f,
		INFINITY, INFINITY, INFINITY, INFINITY, INFINITY,
		25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f,
		INFINITY, INFINITY,
		25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f, 25.0f,
		12.5f, 12.5f, 12.5f, 12.5f,
		25.0f, 25.0f, 25.0f,
		6.25f,
		25.0f, 25.0f, 25.0f,
	};

	grid_map.update(0.1f, ranges);
}