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

#ifndef PRECISION_EVALUATOR_H
#define PRECISION_EVALUATOR_H

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dbscan.h"
#include "dogm/dogm_types.h"
#include "simulator.h"

class PrecisionEvaluator
{
public:
    explicit PrecisionEvaluator(const SimulationData _sim_data, const float _resolution)
        : sim_data{_sim_data}, resolution{_resolution}
    {
        // step_errors.reserve(sim_data.size());
    }

    void evaluateAndStoreStep(int simulation_step_index, const std::vector<Point<dogm::GridCell>>& cells_with_velocity,
                              bool print_current_precision = false)
    {
        if (cells_with_velocity.size() > 0)
        {
            for (const auto& vehicle : sim_data[simulation_step_index].vehicles)
            {
                std::cout << "\nGround Truth: vel: " << vehicle.vel[0] << " " << vehicle.vel[1]
                          << ", pos: " << vehicle.pos[0] << " " << vehicle.pos[1] << "\n";
            }

            const auto clusters = computeDbscanClusters(cells_with_velocity);
            int cluster_id = 0;
            for (const auto& cluster : clusters)
            {
                float x_vel = 0.0f, y_vel = 0.0f, x_pos = 0.0f, y_pos = 0.0f;
                for (auto& point : cluster)
                {
                    x_vel += point.data.mean_x_vel;
                    y_vel += point.data.mean_y_vel;
                    x_pos += point.x;
                    y_pos += point.y;
                }

                float mean_x_vel = (x_vel / cluster.size()) * resolution;
                float mean_y_vel = -(y_vel / cluster.size()) * resolution;
                float mean_x_pos = (x_pos / cluster.size()) * resolution;
                float mean_y_pos = (y_pos / cluster.size()) * resolution;

                // Ground truth velocities are in polar coordinates so convert them
                mean_y_vel = sqrtf(powf(mean_x_vel, 2) + powf(mean_y_vel, 2));
                mean_x_vel = atan2(mean_y_vel, mean_x_vel);

                // mean_y_pos = sqrtf(powf(mean_x_pos, 2) + powf(mean_y_pos, 2));
                // mean_x_pos = atan2(mean_y_pos, mean_x_pos);

                if (print_current_precision)
                {
                    // std::cout << "Cluster ID: " << cluster_id << "\n";
                    std::cout << std::setprecision(2);
                    std::cout << "Est. values: vel: " << mean_x_vel << " " << mean_y_vel;
                    std::cout << " , pos: " << mean_x_pos << " " << mean_y_pos << "\n\n";
                }

                // Find matching ground truth vehicle: compute center location of cluster. Find vehicles with distance
                // smaller than eps; from those, take vehicle with smallest dist Compute velocity error, store

                cluster_id++;
            }
        }
    }

    Clusters<dogm::GridCell> computeDbscanClusters(const std::vector<Point<dogm::GridCell>>& cells_with_velocity)
    {
        DBSCAN<dogm::GridCell> dbscan(3.0f, 5);
        return dbscan.cluster(cells_with_velocity);
    }

    void printSummary() { std::cout << "\nPrecision evaluator prints no summary yet.\n"; }

private:
    SimulationData sim_data;
    float resolution;
    // std::vector<?> step_errors;
};

#endif  // PRECISION_EVALUATOR_H
