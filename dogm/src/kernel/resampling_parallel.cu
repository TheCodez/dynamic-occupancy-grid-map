#include "dogm/common.h"
#include "dogm/cuda_utils.h"
#include "dogm/dogm_types.h"
#include "dogm/kernel/resampling_parallel.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/binary_search.h>

namespace dogm 
{

__global__ void resampleIndexKernel(const ParticlesSoA particle_array, ParticlesSoA particle_array_next,
    const ParticlesSoA birth_particle_array, const int* __restrict__ idx_array_up,
    const int* __restrict__ idx_array_down, float new_weight, int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        int idx = i + idx_array_up[i] + idx_array_down[i];

        // if (idx_array_up[i] != 0 || idx_array_down[i] != 0)
        //     printf("\t(%d, %d,%d, %d)", i, idx_array_up[i], idx_array_down[i], idx);

        if (idx < particle_count)
        {
            particle_array_next.copy(particle_array, i, idx);
        }
        else
        {
            particle_array_next.copy(birth_particle_array, i, idx - particle_count);
            // printf("!");
        }

        particle_array_next.weight[i] = new_weight;
    }
}

// Systematic / Stratified max optimized

__global__ void __launch_bounds__(kTRI) resampleSystematicIndexUp(int const num_particles,
    unsigned long long int const seed, int* __restrict__ resampling_index_up, float* __restrict__ prefix_sum) {
    auto const tile_32 = cg::tiled_partition<kWarpSize>(cg::this_thread_block());

    __shared__ float s_warp_0[kTRI]; // strange diff *2
    __shared__ float s_warp_1[kTRI]; // strange diff *2

    // Setting prefix_sum[n - 1] in each block versus call a separate kernel
    // beforehand. Set last value in prefix-sum to 1.0f
    if ( threadIdx.x == 0 ) {
        prefix_sum[num_particles - 1] = 1.0f;  //
    }

    for ( unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_particles;
          tid += blockDim.x * gridDim.x ) {

        curandStateXORWOW_t local_state {};

        float distro {};

        if (systematic) {
            curand_init( seed, 0, 0, &local_state );
            distro = curand_uniform( &local_state );
        } else {
            curand_init( seed + tid, 0, 0, &local_state );
            distro = curand_uniform( &local_state );
        }

        if ( threadIdx.x < kWarpSize ) {
            ResamplingUpPerWarp(tile_32, tid, num_particles, distro, s_warp_0, prefix_sum, resampling_index_up);
        } else {
            ResamplingUpPerWarp(tile_32, tid, num_particles, distro, s_warp_1, prefix_sum, resampling_index_up);
        }
    }
}

__device__ void ResamplingUpPerWarp(cg::thread_block_tile<kWarpSize> const &tile_32,
    unsigned int const &tid, int const &num_particles, float const &distro,
    float* shared, float* __restrict__ prefix_sum, int* __restrict__ resampling_index_up) {

    float const    tidf { static_cast<float>( tid ) };
    auto const t { tile_32.thread_rank( ) };

    int l {0};
    int idx {0};
    float   a {};
    float   b {};

    bool mask { true };

    if ( tid < num_particles - kWarpSize - l ) { // strange diff kWarpSize or kTRI
        shared[t] = prefix_sum[tid + l];     
        shared[t + kWarpSize] = prefix_sum[tid + kWarpSize + l];   // strange diff kWarpSize || kTRI
    }

    // Distribution will be the same for each Monte Carlo
    float const draw = ( distro + tidf ) / num_particles;

    tile_32.sync();

    while (tile_32.any(mask)) {
        if (tid < num_particles - (kTRI) - l) {  // strange diff (+ kWarpSize)

            a = prefix_sum[tid + kWarpSize + l];
            b = prefix_sum[tid + kTRI + l]; // strange diff + kWarpSize

            #pragma unroll         // strange diff  kWarpSize or kTRI
            for ( int i = 0; i < kWarpSize; i++ ) {          // strange diff kWarpSize or kTRI
                mask = shared[t + i] < draw;
                if ( mask ) {
                    idx++;
                }
            }
            l += kWarpSize;          // strange diff kWarpSize or kTRI
            shared[t] = a;
            shared[t + kWarpSize] = b;           // strange diff kWarpSize or kTRI

            tile_32.sync();
        } else {
            while ( mask && tid < ( num_particles - l ) ) {
                mask = prefix_sum[tid + l] < draw;
                if ( mask ) {
                    idx++;
                }
                l++;
            }
        }

        tile_32.sync( );
    }
    resampling_index_up[tid] = idx;
}

__global__ void __launch_bounds__(kTRI) resampleSystematicIndexDown(int const num_particles,
    unsigned long long int const seed, int *__restrict__ resampling_index_down, float *__restrict__ prefix_sum) {

    auto const tile_32 = cg::tiled_partition<kWarpSize>( cg::this_thread_block( ) );

    __shared__ float s_warp_0[kTRI]; // strange diff *2
    __shared__ float s_warp_1[kTRI]; // strange diff *2

    // Setting prefix_sum_particle_weights[n - 1] in each block versus call a
    // separate kernel beforehand
    if ( threadIdx.x == 0 ) {
        prefix_sum[num_particles - 1] = 1.0f;
    }

    for ( unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_particles;
          tid += blockDim.x * gridDim.x ) {

        curandStateXORWOW_t local_state {};

        float distro {};

        if ( systematic ) {
            curand_init( seed, 0, 0, &local_state );
            distro = curand_uniform( &local_state );
        } else {
            curand_init( seed + tid, 0, 0, &local_state );
            distro = curand_uniform( &local_state );
        }

        if ( threadIdx.x < kWarpSize ) {
            ResamplingDownPerWarp( tile_32, tid, num_particles, distro, s_warp_0, prefix_sum, resampling_index_down );
        } else {
            ResamplingDownPerWarp( tile_32, tid, num_particles, distro, s_warp_1, prefix_sum, resampling_index_down );
        }
    }
}

__device__ void ResamplingDownPerWarp( cg::thread_block_tile<kWarpSize> const &tile_32,
    unsigned int const &tid, int const &num_particles, float const &distro,
    float *shared, float *__restrict__ prefix_sum, int *__restrict__ resampling_index_down ) {

    float const tidf { static_cast<float>( tid ) };
    auto const t { tile_32.thread_rank( ) };

    int l {1};
    int idx {0};
    float a{};
    float b{};

    bool mask { false };

    // Preload in into shared memory
    if ( tid >= kWarpSize + l ) { // strange diff kWarpSize or kTRI
        shared[t] = prefix_sum[tid - kWarpSize - l]; // strange diff kWarpSize or kTRI
        shared[t + kWarpSize] = prefix_sum[tid - l]; // strange diff kWarpSize or kTRI
    }

    // Distribution will be the same for each Monte Carlo
    float const draw = ( distro + tidf ) / num_particles;

    tile_32.sync( );

    while ( !tile_32.all( mask ) ) {

        if ( tid >= kTRI + l ) { // strange diff  + kWarpSize
            a = prefix_sum[tid - ( kTRI )-l]; // strange diff + kWarpSize)
            b = prefix_sum[tid - kWarpSize - l];

            #pragma unroll
            for ( int i = 1; i < kWarpSize + 1; i++ ) { // strange diff kWarpSize or kTRI
                mask = shared[t + kWarpSize - i] < draw; // strange diff kWarpSize or kTRI
                if ( !mask ) {
                    idx--;
                }
            }
            l += kWarpSize; // strange diff kWarpSize or kTRI
            shared[t]             = a;
            shared[t + kWarpSize] = b; // strange diff kWarpSize or kTRI
            tile_32.sync( );

        } else {

            while ( !mask ) {
                if ( tid > l ) {
                    mask = prefix_sum[tid - ( l + 1 )] < draw;
                } else {
                    mask = true;
                }
                if ( !mask ) {
                    idx--;
                }
                l++;
            }
        }

        tile_32.sync( );
    }
    resampling_index_down[tid] = idx;
}

}