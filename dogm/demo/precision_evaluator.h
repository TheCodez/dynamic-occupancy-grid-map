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
#include <iostream>
#include <vector>

#include "dbscan.h"
#include "dogm/dogm_types.h"
#include "simulator.h"

class PrecisionEvaluator
{
public:
    explicit PrecisionEvaluator(const SimulationData _sim_data) : sim_data{_sim_data} {}

    void evaluateAndStoreStep(int simulation_step_index, const std::vector<Point<dogm::GridCell>>& cells_with_velocity,
                              bool print_current_precision = false)
    {
        if (cells_with_velocity.size() > 0)
        {
            DBSCAN<dogm::GridCell> dbscan(cells_with_velocity);
            dbscan.cluster(3.0f, 5);

            std::vector<Point<dogm::GridCell>> cluster_points = dbscan.getPoints();
            int num_cluster = dbscan.getNumCluster();
            std::map<int, std::vector<Point<dogm::GridCell>>> clustered_map = dbscan.getClusteredPoints();

            for (const auto& iter : clustered_map)
            {
                int cluster_id = iter.first;
                std::vector<Point<dogm::GridCell>> cluster = iter.second;

                float y_vel = 0.0f, x_vel = 0.0f;
                for (auto& point : cluster)
                {
                    x_vel += point.data.mean_x_vel;
                    y_vel += point.data.mean_y_vel;
                }

                float resolution = 0.2f;
                float mean_x_vel = (x_vel / cluster.size()) * resolution;
                float mean_y_vel = -(y_vel / cluster.size()) * resolution;

                // Ground truth velocities are in polar coordinates so convert them
                mean_y_vel = sqrtf(powf(mean_x_vel, 2) + powf(mean_y_vel, 2));
                mean_x_vel = atan2(mean_y_vel, mean_x_vel);

                if (print_current_precision)
                    std::cout << "Cluster ID: " << cluster_id << " , est. x-velocity: " << mean_x_vel
                              << ", est. y-velocity: " << mean_y_vel << "\n";
            }
        }
    }

    void printSummary() { std::cout << "\nPrecision evaluator prints no summary yet.\n"; }

private:
    SimulationData sim_data;
};

#endif  // PRECISION_EVALUATOR_H
