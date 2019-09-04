#include "occupancy_grid_map/occupancy_grid_map.h"

int main(int argc, const char** argv) 
{
	GridParams params;
	params.width = 120;
	params.height = 120;
	params.resolution = 0.1f;
	params.v = 2 * 10e6;
	params.vb = 2 * 10e5;
	params.ps = 0.99f;
	params.process_noise_position = 0.02f;
	params.process_noise_velocity = 0.8f;
	params.pb = 0.02f;

	OccupancyGridMap gridMap(params);
}