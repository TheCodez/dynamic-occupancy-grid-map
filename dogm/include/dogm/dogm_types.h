#pragma once

#include <glm/vec4.hpp>
#include <cuda_runtime.h>

namespace dogm
{

struct GridCell
{
    // x: start_idx, y: end_idx
    int2 particle_indices;

    // x: pers_occ_mass, y: new_born_occ_mass
    float2 rho_masses;

    // x: occ_mass, y: free_mass
    float2 masses;

    // x: mu_A, y: mu_UA
    float2 norm_constants;

    // x: w_A, y: w_UA
    float2 birth_weights;

    float2 mean_vel;
    float2 var_vel;
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
    glm::vec4 state;
};

} /* namespace dogm */
