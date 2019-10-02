#include "occupancy_grid_map.h"

#include <iostream>
#include <stdio.h>

float pignistic_transformation(float free_mass, float occ_mass)
{
	return occ_mass + 0.5f * (1.0f - occ_mass - free_mass);
}

template <typename T>
void save_image(const char* filename, T* grid, int width, int height)
{
	FILE* pgmimg;
	pgmimg = fopen(filename, "wb");

	// Writing Magic Number to the File 
	fprintf(pgmimg, "P2\n");

	// Writing Width and Height 
	fprintf(pgmimg, "%d %d\n", width, height);

	// Writing the maximum gray value 
	fprintf(pgmimg, "255\n");

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int index = y * width + x;

			const T& cell = grid[index];
			float occ = pignistic_transformation(cell.free_mass, cell.occ_mass);
			int temp = (int)ceil(occ * 255);
			fprintf(pgmimg, "%d ", 255 - temp);
		}
		fprintf(pgmimg, "\n");
	}

	fclose(pgmimg);
}

int main(int argc, const char** argv) 
{
	GridParams params;
	params.width = 50;//120;
	params.height = 50;//120;
	params.resolution = 0.2f;
	params.particle_count = 3000;//2 * static_cast<int>(10e6);
	params.new_born_particle_count = 2000;//2 * static_cast<int>(10e5);
	params.ps = 0.99f;
	params.process_noise_position = 0.02f;
	params.process_noise_velocity = 0.8f;
	params.pb = 0.02f;

	LaserSensorParams laser_params;
	laser_params.fov = 70.0f;
	laser_params.max_range = 50.0f;

	OccupancyGridMap grid_map(params, laser_params);

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

	const int width = static_cast<int>(params.width / params.resolution);
	const int height = static_cast<int>(params.height / params.resolution);
	
	grid_map.updateMeasurementGrid(ranges, 70);
	grid_map.updateDynamicGrid(0.1f);

	save_image("result_measurement_grid.pgm", grid_map.meas_cell_array, width, height);
	save_image("result_dynamic_grid.pgm", grid_map.grid_cell_array, width, height);
}