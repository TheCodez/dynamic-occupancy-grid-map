// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef LASER_TO_MEAS_GRID_H
#define LASER_TO_MEAS_GRID_H

#include "dogm/dogm_types.h"
#include "mapping/opengl/renderer.h"
#include <memory>
#include <vector>

class LaserMeasurementGrid
{
public:
    struct Params
    {
        float max_range;
        float resolution;
        float fov;
    };

    LaserMeasurementGrid(const Params& params, float grid_length, float resolution);
    ~LaserMeasurementGrid();

    dogm::MeasurementCell* generateGrid(const std::vector<float>& measurements);

private:
    dogm::MeasurementCell* meas_grid;
    int grid_size;

    Params params;
    std::unique_ptr<Renderer> renderer;
};

#endif  // LASER_TO_MEAS_GRID_H