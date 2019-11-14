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
#include "occupancy_grid_map.h"

#include <glm/glm.hpp>

#include <iostream>
#include <stdio.h>

#include <chrono>
#include <cmath>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "measurements.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

using namespace std;

#define PI 3.14159265358979323846f

void HSVtoRGB(int H, double S, double V, int output[3])
{
	double C = S * V;
	double X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
	double m = V - C;
	double Rs, Gs, Bs;

	if (H >= 0 && H < 60) 
	{
		Rs = C;
		Gs = X;
		Bs = 0;
	}
	else if (H >= 60 && H < 120)
	{
		Rs = X;
		Gs = C;
		Bs = 0;
	}
	else if (H >= 120 && H < 180)
	{
		Rs = 0;
		Gs = C;
		Bs = X;
	}
	else if (H >= 180 && H < 240)
	{
		Rs = 0;
		Gs = X;
		Bs = C;
	}
	else if (H >= 240 && H < 300)
	{
		Rs = X;
		Gs = 0;
		Bs = C;
	}
	else 
	{
		Rs = C;
		Gs = 0;
		Bs = X;
	}

	output[0] = (Rs + m) * 255;
	output[1] = (Gs + m) * 255;
	output[2] = (Bs + m) * 255;
}

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
	params.size = 50;
	params.resolution = 0.1f;
	params.particle_count = 2 * static_cast<int>(10e5);
	params.new_born_particle_count = 2 * static_cast<int>(10e4);
	params.p_S = 0.99f;
	params.process_noise_position = 0.02f;
	params.process_noise_velocity = 0.8f;
	params.p_B = 0.02f;

	LaserSensorParams laser_params;
	laser_params.fov = 120.0f;
	laser_params.max_range = 50.0f;

	OccupancyGridMap grid_map(params, laser_params);

	float* measurements[9] = { ranges, ranges2, ranges3, ranges4, ranges4, ranges4, ranges4, ranges4, ranges4 };
//	float* measurements[5] = { ranges4, ranges3, ranges2, ranges, ranges };
	int size = sizeof(ranges) / sizeof(ranges[0]);

	for (float* meas : measurements)
	{
		auto begin = chrono::high_resolution_clock::now();

		grid_map.updateMeasurementGrid(meas, size);
		//for (int i = 0; i < 10; i++)
		{
			grid_map.updateDynamicGrid(0.1f);
		}
		auto end = chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::cout << "Iteration took: " << ms << " ms" << std::endl;
	}

	cv::Mat particle_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < grid_map.particle_count; i++)
	{
		const Particle& part = grid_map.particle_array[i];
		float x = part.state[0];
		float y = part.state[1];

		if ((x >= 0 && x < grid_map.getGridSize()) && (y >= 0 && y < grid_map.getGridSize()))
		{
			particle_img.at<cv::Vec3b>(static_cast<int>(y), static_cast<int>(x)) = cv::Vec3b(0, 0, 255);
		}
	}

	cv::Mat grid_img(grid_map.getGridSize(), grid_map.getGridSize(), CV_8UC3);
	for (int y = 0; y < grid_map.getGridSize(); y++)
	{
		for (int x = 0; x < grid_map.getGridSize(); x++)
		{
			int index = y * grid_map.getGridSize() + x;

			const GridCell& cell = grid_map.grid_cell_array[index];
			float occ = pignistic_transformation(cell.free_mass, cell.occ_mass);
			uchar temp = (uchar) floor(occ * 255);
			
			cv::Mat vel_img(2, 1, CV_32FC1);
			vel_img.at<float>(0) = cell.mean_x_vel;
			vel_img.at<float>(1) = cell.mean_y_vel;

			cv::Mat covar_img(2, 2, CV_32FC1);
			covar_img.at<float>(0, 0) = cell.var_x_vel;
			covar_img.at<float>(1, 0) = cell.covar_xy_vel;
			covar_img.at<float>(0, 1) = cell.covar_xy_vel;
			covar_img.at<float>(1, 1) = cell.var_y_vel;

			cv::Mat mdist = vel_img.t() * covar_img.inv() * vel_img;

			if (occ >= 0.7f)// && mdist.at<float>(0, 0) > 4.0)
			{
				float angle = atan2(cell.mean_y_vel, cell.mean_x_vel) * (180.0f / PI);

				int color[3];
				HSVtoRGB((int)ceil(angle), 1.0, 1.0, color);

				grid_img.at<cv::Vec3b>(y, x) = cv::Vec3b(color[2], color[1], color[0]);

			}
			else
			{
				grid_img.at<cv::Vec3b>(y, x) = cv::Vec3b(255 - temp, 255 - temp, 255 - temp);
			}
		}
	}
	cv::namedWindow("dynamic_grid", cv::WINDOW_NORMAL);
	cv::imshow("dynamic_grid", grid_img);
	
	cv::namedWindow("particles", cv::WINDOW_NORMAL);
	cv::imshow("particles", particle_img);
	cv::waitKey(0);

	save_image("result_measurement_grid.pgm", grid_map.meas_cell_array, grid_map.getGridSize(), grid_map.getGridSize());
	save_image("result_dynamic_grid.pgm", grid_map.grid_cell_array, grid_map.getGridSize(), grid_map.getGridSize());
}