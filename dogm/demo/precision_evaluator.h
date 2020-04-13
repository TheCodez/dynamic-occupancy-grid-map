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

struct PointWithVelocity
{
    float x{0.0f};
    float y{0.0f};
    float v_x{0.0f};
    float v_y{0.0f};
};

class PrecisionEvaluator
{
public:
    explicit PrecisionEvaluator(const SimulationData _sim_data, const float _resolution)
        : sim_data{_sim_data}, resolution{_resolution}
    {
        number_of_detections = 0;
        number_of_unassigned_detections = 0;
    }

    void evaluateAndStoreStep(int simulation_step_index, const std::vector<Point<dogm::GridCell>>& cells_with_velocity,
                              bool print_current_precision = false)
    {
        const auto groundtruth_vehicles = sim_data[simulation_step_index].vehicles;
        if (cells_with_velocity.size() > 0 && groundtruth_vehicles.size() > 0)
        {
            const auto clusters = computeDbscanClusters(cells_with_velocity);
            int cluster_id = 0;
            for (const auto& cluster : clusters)
            {
                PointWithVelocity cluster_mean = computeClusterMean(cluster);

                std::vector<Vehicle> matching_groundtruth_vehicles{};
                for (const auto& vehicle : groundtruth_vehicles)
                {
                    const float distance = sqrtf(powf(cluster_mean.x - vehicle.pos[0], 2.0f) +
                                                 powf(cluster_mean.y - vehicle.pos[1], 2.0f));
                    if (distance < kMaximumAssignmentDistance)
                    {
                        matching_groundtruth_vehicles.push_back(vehicle);
                    }
                }

                if (matching_groundtruth_vehicles.size() == 0)
                {
                    ++number_of_unassigned_detections;
                    continue;
                }

                std::sort(matching_groundtruth_vehicles.begin(), matching_groundtruth_vehicles.end(),
                          [&cluster_mean](const Vehicle& a, const Vehicle& b) {
                              const float distance_a =
                                  sqrtf(powf(cluster_mean.x - a.pos[0], 2.0f) + powf(cluster_mean.y - a.pos[1], 2.0f));
                              const float distance_b =
                                  sqrtf(powf(cluster_mean.x - b.pos[0], 2.0f) + powf(cluster_mean.y - b.pos[1], 2.0f));
                              return distance_a < distance_b;
                          });

                const auto closest_vehicle = matching_groundtruth_vehicles[0];

                accumulateErrors(cluster_mean, closest_vehicle);

                if (print_current_precision)
                {
                    std::cout << std::setprecision(2);
                    std::cout << "\nCluster ID=" << cluster_id << "\n";
                    std::cout << "Vel. Err.: " << closest_vehicle.vel[0] - cluster_mean.v_x << " "
                              << closest_vehicle.vel[1] - cluster_mean.v_y
                              << ", Pos. Err.: " << closest_vehicle.pos[0] - cluster_mean.x << " "
                              << closest_vehicle.pos[1] - cluster_mean.y << "\n";
                }
                cluster_id++;
            }
        }
    }

    void printSummary()
    {
        std::cout << "\nMean absolute errors (x,y): \n";
        std::cout << "Position: " << cumulative_error.x / number_of_detections << " "
                  << cumulative_error.y / number_of_detections << "\n";
        std::cout << "Velocity: " << cumulative_error.v_x / number_of_detections << " "
                  << cumulative_error.v_y / number_of_detections << "\n\n";
        std::cout << "Detections unassigned by evaluator: " << number_of_unassigned_detections << "\n";
        std::cout << "Maximum possible detections: " << sim_data[0].vehicles.size() * sim_data.size() << "\n";
    }

private:
    Clusters<dogm::GridCell> computeDbscanClusters(const std::vector<Point<dogm::GridCell>>& cells_with_velocity)
    {
        DBSCAN<dogm::GridCell> dbscan(3.0f, 5);
        return dbscan.cluster(cells_with_velocity);
    }

    void accumulateErrors(const PointWithVelocity& cluster_mean, const Vehicle& vehicle)
    {
        cumulative_error.x += std::abs(cluster_mean.x - vehicle.pos[0]);
        cumulative_error.y += std::abs(cluster_mean.y - vehicle.pos[1]);
        cumulative_error.v_x += std::abs(cluster_mean.v_x - vehicle.vel[0]);
        cumulative_error.v_y += std::abs(cluster_mean.v_y - vehicle.vel[1]);
        ++number_of_detections;
    }

    PointWithVelocity computeClusterMean(const Cluster<dogm::GridCell>& cluster)
    {
        PointWithVelocity cluster_mean;
        for (auto& point : cluster)
        {
            cluster_mean.x += point.x;
            cluster_mean.y += point.y;
            cluster_mean.v_x += point.data.mean_x_vel;
            cluster_mean.v_y += point.data.mean_y_vel;
        }

        cluster_mean.x = (cluster_mean.x / cluster.size()) * resolution;
        cluster_mean.y = (cluster_mean.y / cluster.size()) * resolution;
        cluster_mean.v_x = (cluster_mean.v_x / cluster.size()) * resolution;
        cluster_mean.v_y = -(cluster_mean.v_y / cluster.size()) * resolution;

        const float x_offset = 22.0f;                            // motivated by numerical experiments
        const float y_offset = 2.0f * (25.0f - cluster_mean.y);  // motivated by numerical experiments
        cluster_mean.x += x_offset;
        cluster_mean.y += y_offset;

        return cluster_mean;
    }

    SimulationData sim_data;
    float resolution;
    PointWithVelocity cumulative_error;
    int number_of_detections;
    int number_of_unassigned_detections;
};

#endif  // PRECISION_EVALUATOR_H
