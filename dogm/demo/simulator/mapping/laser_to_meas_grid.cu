// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "mapping/laser_to_meas_grid.h"
#include "mapping/opengl/renderer.h"

#include "mapping/kernel/measurement_grid.h"

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

    float* d_measurements;
    CHECK_ERROR(cudaMalloc(&d_measurements, num_measurements * sizeof(float)));
    CHECK_ERROR(
        cudaMemcpy(d_measurements, measurements.data(), num_measurements * sizeof(float), cudaMemcpyHostToDevice));

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
                                                          params.resolution);

    CHECK_ERROR(cudaGetLastError());
    polar_texture.endCudaAccess(polar_surface);

    // render cartesian image to texture using polar texture
    renderer->renderToTexture(polar_texture);

    auto framebuffer = renderer->getFrameBuffer();
    cudaSurfaceObject_t cartesian_surface;

    framebuffer->beginCudaAccess(&cartesian_surface);
    // transform RGBA texture to measurement grid
    cartesianGridToMeasurementGridKernel<<<cart_grid_dim, dim_block>>>(meas_grid, cartesian_surface, grid_size);

    CHECK_ERROR(cudaGetLastError());
    framebuffer->endCudaAccess(cartesian_surface);

    CHECK_ERROR(cudaFree(d_measurements));
    CHECK_ERROR(cudaDeviceSynchronize());

    return meas_grid;
}
