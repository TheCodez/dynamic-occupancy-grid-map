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
#pragma once

#include <memory>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include <vector>

class Renderer;

namespace dogm
{

struct GridCell;
struct MeasurementCell;
struct Particle;

struct GridParams
{
    float size;
    float resolution;
    int particle_count;
    int new_born_particle_count;
    float persistence_prob;
    float process_noise_position;
    float process_noise_velocity;
    float birth_prob;
    float velocity_persistent;
    float velocity_birth;
};

struct LaserSensorParams
{
    float max_range;
    float fov;
};

class DOGM
{
public:
    DOGM(const GridParams& params, const LaserSensorParams& laser_params);
    ~DOGM();

    void updateMeasurementGridFromArray(const std::vector<float2>& measurements);

    void updateMeasurementGrid(float* measurements, int num_measurements);
    void updateParticleFilter(float dt);

    int getGridSize() const { return grid_size; }
    float getResolution() const { return params.resolution; }

    float getPositionX() const { return 0.0f; }
    float getPositionY() const { return 0.0f; }

    int getIteration() const { return iteration; }

private:
    void initialize();

public:
    void particlePrediction(float dt);
    void particleAssignment();
    void gridCellOccupancyUpdate();
    void updatePersistentParticles();
    void initializeNewParticles();
    void statisticalMoments();
    void resampling();

public:
    GridParams params;
    LaserSensorParams laser_params;

    std::unique_ptr<Renderer> renderer;

    GridCell* grid_cell_array;
    Particle* particle_array;
    Particle* particle_array_next;
    Particle* birth_particle_array;

    MeasurementCell* polar_meas_cell_array;
    MeasurementCell* meas_cell_array;

    float* weight_array;
    float* birth_weight_array;
    float* born_masses_array;

    curandState* rng_states;

    int grid_size;

    int grid_cell_count;
    int particle_count;
    int new_born_particle_count;

    dim3 block_dim;
    dim3 particles_grid;
    dim3 birth_particles_grid;
    dim3 grid_map_grid;

    int iteration;
};

} /* namespace dogm */
