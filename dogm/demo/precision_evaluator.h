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
        cumulative_error_x_pos = 0.0f;
        cumulative_error_y_pos = 0.0f;
        cumulative_error_x_vel = 0.0f;
        cumulative_error_y_vel = 0.0f;
        number_of_iterations = 0;
        // step_errors.reserve(sim_data.size());
    }

    void evaluateAndStoreStep(int simulation_step_index, const std::vector<Point<dogm::GridCell>>& cells_with_velocity,
                              bool print_current_precision = false)
    {
        if (cells_with_velocity.size() > 0 && sim_data[simulation_step_index].vehicles.size() > 0)
        {
            for (const auto& vehicle : sim_data[simulation_step_index].vehicles)
            {
                std::cout << "\nGround Truth: vel: " << vehicle.vel[0] << " " << vehicle.vel[1]
                          << ", pos: " << vehicle.pos[0] << " " << vehicle.pos[1] << "\n";
            }

            const auto clustered_map = computeDbscanClusters(cells_with_velocity);
            for (const auto& iter : clustered_map)
            {
                int cluster_id = iter.first;
                std::vector<Point<dogm::GridCell>> cluster = iter.second;

                float x_vel = 0.0f, y_vel = 0.0f, x_pos = 0.0f, y_pos = 0.0f;
                for (auto& point : cluster)
                {
                    x_vel += point.data.mean_x_vel;
                    y_vel += point.data.mean_y_vel;
                    x_pos += point.x;
                    y_pos += point.y;
                }

                float mean_x_vel = (x_vel / cluster.size()) * resolution;
                float mean_y_vel = (y_vel / cluster.size()) * resolution;
                float mean_x_pos = (x_pos / cluster.size()) * resolution;
                float mean_y_pos = (y_pos / cluster.size()) * resolution;

                const float x_offset = 22.0f;
                const float y_offset = 2.0f * (25.0f - mean_y_pos);  // motivated by numerical experiments
                mean_x_pos += x_offset;
                mean_y_pos += y_offset;

                if (print_current_precision)
                {
                    // std::cout << "Cluster ID: " << cluster_id << "\n";
                    std::cout << std::setprecision(2);
                    std::cout << "Est. values: vel: " << mean_x_vel << " " << mean_y_vel;
                    std::cout << " , pos: " << mean_x_pos << " " << mean_y_pos << "\n\n";
                }

                // TODO use absolute error once you're not looking into systematic errors anymore
                const auto vehicle = sim_data[simulation_step_index].vehicles[0];
                cumulative_error_x_pos += mean_x_pos - vehicle.pos[0];
                cumulative_error_y_pos += mean_y_pos - vehicle.pos[1];
                cumulative_error_x_vel += mean_x_vel - vehicle.vel[0];
                cumulative_error_y_vel += mean_y_vel - vehicle.vel[1];
                ++number_of_iterations;

                // Find matching ground truth vehicle: compute center location of cluster. Find vehicles with distance
                // smaller than eps; from those, take vehicle with smallest dist Compute velocity error, store
            }
        }
    }

    std::map<int, std::vector<Point<dogm::GridCell>>>
    computeDbscanClusters(const std::vector<Point<dogm::GridCell>>& cells_with_velocity)
    {
        DBSCAN<dogm::GridCell> dbscan(cells_with_velocity);
        dbscan.cluster(3.0f, 5);

        std::vector<Point<dogm::GridCell>> cluster_points = dbscan.getPoints();
        int num_cluster = dbscan.getNumCluster();
        return dbscan.getClusteredPoints();
    }

    void printSummary()
    {
        std::cout << "Mean errors: \n";
        std::cout << "Position: " << cumulative_error_x_pos / number_of_iterations << " "
                  << cumulative_error_y_pos / number_of_iterations << "\n";
        std::cout << "Velocity: " << cumulative_error_x_vel / number_of_iterations << " "
                  << cumulative_error_y_vel / number_of_iterations << "\n";
        // TODO add reporting of relative error
    }

private:
    SimulationData sim_data;
    float resolution;
    float cumulative_error_x_pos;
    float cumulative_error_y_pos;
    float cumulative_error_x_vel;
    float cumulative_error_y_vel;
    int number_of_iterations;
    // std::vector<?> step_errors;
};

#endif  // PRECISION_EVALUATOR_H
