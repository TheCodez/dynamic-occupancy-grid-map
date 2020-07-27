// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "mapping/laser_to_meas_grid.h"
#include "mapping/opengl/renderer.h"

#include "mapping/kernel/measurement_grid.h"

#include <thrust/device_vector.h>

LaserMeasurementGrid::LaserMeasurementGrid(const Params& params, float grid_length, float resolution)
    : grid_size(static_cast<int>(grid_length / resolution)), params(params)
{
    int grid_cell_count = grid_size * grid_size;

    CHECK_ERROR(cudaMalloc(&meas_grid, grid_cell_count * sizeof(dogm::MeasurementCell)));

    renderer = std::make_unique<Renderer>(grid_size, params.fov, grid_length, params.max_range);
}

LaserMeasurementGrid::~LaserMeasurementGrid()
{
    CHECK_ERROR(cudaFree(meas_grid));
}

dogm::MeasurementCell* LaserMeasurementGrid::generateGrid(const std::vector<float>& measurements)
{
    const int num_measurements = measurements.size();

    thrust::device_vector<float> meas_points(num_measurements * params.num_layers, INFINITY);
    std::vector<int> point_count(num_measurements);

    for (int i = 0; i < num_measurements; i++)
    {
        float range = measurements[i];

        int count = point_count[i];
        for (int j = 0; j < params.num_layers; j++)
        {
            meas_points[i * params.num_layers + j] = range + j * 5;
            point_count[i]++;
        }
    }

    float* d_measurements = thrust::raw_pointer_cast(meas_points.data());

    const int polar_width = num_measurements;
    const int polar_height = static_cast<int>(params.max_range / params.resolution);

    dim3 dim_block(32, 32);
    dim3 grid_dim(divUp(polar_width, dim_block.x), divUp(polar_height, dim_block.y));
    dim3 cart_grid_dim(divUp(grid_size, dim_block.x), divUp(grid_size, dim_block.y));

    const float anisotropy_level = 16.0f;
    Texture polar_texture(polar_width, polar_height, anisotropy_level);
    cudaSurfaceObject_t polar_surface;

    // create polar texture
    polar_texture.beginCudaAccess(&polar_surface);
    createPolarGridTextureKernel<<<grid_dim, dim_block>>>(polar_surface, d_measurements, polar_width, polar_height,
                                                          params.resolution, params.stddev_range, params.num_layers);

    CHECK_ERROR(cudaGetLastError());
    polar_texture.endCudaAccess(polar_surface);

    // render cartesian image to texture using polar texture
    renderer->renderToTexture(polar_texture);

    Framebuffer* framebuffer = renderer->getFrameBuffer();
    cudaSurfaceObject_t cartesian_surface;

    framebuffer->beginCudaAccess(&cartesian_surface);
    // transform RGBA texture to measurement grid
    cartesianGridToMeasurementGridKernel<<<cart_grid_dim, dim_block>>>(meas_grid, cartesian_surface, grid_size);

    CHECK_ERROR(cudaGetLastError());
    framebuffer->endCudaAccess(cartesian_surface);

    CHECK_ERROR(cudaDeviceSynchronize());

    return meas_grid;
}

#include <fstream>
#include <iostream>

struct LidarPoint
{
    float x, y, z;
    float intensity;
};

std::vector<LidarPoint> loadLidar(int i)
{
    std::vector<LidarPoint> cloud;

    // const char* path = "C:/Development/SelfDrivingCar/catkin_ws/src/argoverse-api/demo_usage/cloud.bin";
    const char* path = "C:/Development/_PytorchDev/pytorch-LiLa/data/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync/"
                       "velodyne_points/data/0000000135.bin";

    std::ifstream input(path, std::ios_base::binary);

    if (!input.good())
    {
        std::cerr << "Cannot open file" << std::endl;
        return {};
    }

    for (int i = 0; input.good() && !input.eof(); i++)
    {
        LidarPoint point;
        input.read((char*)&point.x, 3 * sizeof(float));
        input.read((char*)&point.intensity, sizeof(float));
        cloud.push_back(point);
    }

    return cloud;
}

dogm::MeasurementCell* LaserMeasurementGrid::generateGrid()  // pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud)
{
    std::vector<LidarPoint> cloud = loadLidar(0);

    float angle_min = -M_PI;
    float angle_max = M_PI;
    float angle_increment = M_PI / 180.0f;

    int ranges_size = std::ceil((angle_max - angle_min) / angle_increment);

    thrust::device_vector<float> meas_points(ranges_size * params.num_layers, 110);
    std::vector<int> point_count(ranges_size);

    for (int i = 0; i < cloud.size(); i++)
    {
        float x = cloud[i].x;
        float y = cloud[i].y;
        float z = cloud[i].z;

        // printf("x: %f, y: %f, z: %f\n", x, y, z);

        if (std::isnan(x) || std::isnan(y) || std::isnan(z))
        {
            continue;
        }

        if (z > 4.0f || z < 0.5f)
        {
            continue;
        }

        double range = hypot(x, y);
        if (range < 0.0f || range > 100.0f)
        {
            continue;
        }

        double angle = atan2(y, x);
        if (angle < angle_min || angle > angle_max)
        {
            continue;
        }

        int index = (angle - angle_min) / angle_increment;
        int count = point_count[index];
        if (count < params.num_layers)
        {
            meas_points[index * params.num_layers + count] = range;
            point_count[index]++;
        }
    }

    float* d_meas_points = thrust::raw_pointer_cast(meas_points.data());

    const int polar_width = ranges_size;
    const int polar_height = static_cast<int>(params.max_range / params.resolution);

    dim3 dim_block(32, 32);
    dim3 grid_dim(divUp(polar_width, dim_block.x), divUp(polar_height, dim_block.y));
    dim3 cart_grid_dim(divUp(grid_size, dim_block.x), divUp(grid_size, dim_block.y));

    const float anisotropy_level = 16.0f;
    Texture polar_texture(polar_width, polar_height, anisotropy_level);
    cudaSurfaceObject_t polar_surface;

    // create polar texture
    polar_texture.beginCudaAccess(&polar_surface);
    createPolarGridTextureKernel<<<grid_dim, dim_block>>>(polar_surface, d_meas_points, polar_width, polar_height,
                                                          params.resolution, params.stddev_range, params.num_layers);

    CHECK_ERROR(cudaGetLastError());
    polar_texture.endCudaAccess(polar_surface);

    // render cartesian image to texture using polar texture
    renderer->renderToTexture(polar_texture);

    Framebuffer* framebuffer = renderer->getFrameBuffer();
    cudaSurfaceObject_t cartesian_surface;

    framebuffer->beginCudaAccess(&cartesian_surface);
    // transform RGBA texture to measurement grid
    cartesianGridToMeasurementGridKernel<<<cart_grid_dim, dim_block>>>(meas_grid, cartesian_surface, grid_size);

    CHECK_ERROR(cudaGetLastError());
    framebuffer->endCudaAccess(cartesian_surface);

    CHECK_ERROR(cudaDeviceSynchronize());

    return meas_grid;
}
