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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dbscan.h"
#include "dogm/dogm_types.h"
#include "simulator.h"

const float kMaximumAssignmentDistance = 5.0f;

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
        number_of_detections = 0;
        number_of_missed_detections = 0;
        // step_errors.reserve(sim_data.size());
    }

    void evaluateAndStoreStep(int simulation_step_index, const std::vector<Point<dogm::GridCell>>& cells_with_velocity,
                              bool print_current_precision = false)
    {
        const auto groundtruth_vehicles = sim_data[simulation_step_index].vehicles;
        if (cells_with_velocity.size() > 0 && groundtruth_vehicles.size() > 0)
        {
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
                float mean_y_vel = -(y_vel / cluster.size()) * resolution;
                float mean_x_pos = (x_pos / cluster.size()) * resolution;
                float mean_y_pos = (y_pos / cluster.size()) * resolution;

                const float x_offset = 22.0f;                        // motivated by numerical experiments
                const float y_offset = 2.0f * (25.0f - mean_y_pos);  // motivated by numerical experiments
                mean_x_pos += x_offset;
                mean_y_pos += y_offset;

                std::vector<Vehicle> matching_groundtruth_vehicles{};
                for (const auto& vehicle : groundtruth_vehicles)
                {
                    const float distance =
                        sqrtf(powf(mean_x_pos - vehicle.pos[0], 2.0f) + powf(mean_y_pos - vehicle.pos[1], 2.0f));
                    if (distance < kMaximumAssignmentDistance)
                    {
                        matching_groundtruth_vehicles.push_back(vehicle);
                    }
                }

                if (matching_groundtruth_vehicles.size() == 0)
                {
                    ++number_of_missed_detections;
                    continue;
                }

                std::sort(matching_groundtruth_vehicles.begin(), matching_groundtruth_vehicles.end(),
                          [mean_x_pos, mean_y_pos](const Vehicle& a, const Vehicle& b) {
                              const float distance_a =
                                  sqrtf(powf(mean_x_pos - a.pos[0], 2.0f) + powf(mean_y_pos - a.pos[1], 2.0f));
                              const float distance_b =
                                  sqrtf(powf(mean_x_pos - b.pos[0], 2.0f) + powf(mean_y_pos - b.pos[1], 2.0f));
                              return distance_a < distance_b;
                          });

                const auto vehicle = matching_groundtruth_vehicles[0];

                if (print_current_precision)
                {
                    std::cout << std::setprecision(2);
                    std::cout << "\nCluster ID=" << cluster_id << "\n";
                    std::cout << "Vel. Err.: " << vehicle.vel[0] - mean_x_vel << " " << vehicle.vel[1] - mean_y_vel
                              << ", Pos. Err.: " << vehicle.pos[0] - mean_x_pos << " " << vehicle.pos[1] - mean_y_pos
                              << "\n";
                }

                cumulative_error_x_pos += std::abs(mean_x_pos - vehicle.pos[0]);
                cumulative_error_y_pos += std::abs(mean_y_pos - vehicle.pos[1]);
                cumulative_error_x_vel += std::abs(mean_x_vel - vehicle.vel[0]);
                cumulative_error_y_vel += std::abs(mean_y_vel - vehicle.vel[1]);
                ++number_of_detections;
            }
        }
    }

    void printSummary()
    {
        std::cout << "\nMean absolute errors (x,y): \n";
        std::cout << "Position: " << cumulative_error_x_pos / number_of_detections << " "
                  << cumulative_error_y_pos / number_of_detections << "\n";
        std::cout << "Velocity: " << cumulative_error_x_vel / number_of_detections << " "
                  << cumulative_error_y_vel / number_of_detections << "\n\n";
        std::cout << "Detections unassigned by evaluator: " << number_of_missed_detections << "\n";
        std::cout << "Maximum possible detections: " << sim_data[0].vehicles.size() * sim_data.size() << "\n";
    }

private:
    std::map<int, std::vector<Point<dogm::GridCell>>>
    computeDbscanClusters(const std::vector<Point<dogm::GridCell>>& cells_with_velocity)
    {
        DBSCAN<dogm::GridCell> dbscan(cells_with_velocity);
        dbscan.cluster(3.0f, 5);

        std::vector<Point<dogm::GridCell>> cluster_points = dbscan.getPoints();
        int num_cluster = dbscan.getNumCluster();
        return dbscan.getClusteredPoints();
    }

    SimulationData sim_data;
    float resolution;
    float cumulative_error_x_pos;
    float cumulative_error_y_pos;
    float cumulative_error_x_vel;
    float cumulative_error_y_vel;
    int number_of_detections;
    int number_of_missed_detections;
    // std::vector<?> step_errors;
};

#endif  // PRECISION_EVALUATOR_H
