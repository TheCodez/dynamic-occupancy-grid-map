// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dbscan.h"
#include "metrics.h"
#include "precision_evaluator.h"

const float kMaximumAssignmentDistance = 5.0f;
const float kMaximumDbscanNeighborDistance = 3.0f;
const int kMinimumNumberOfNeighbors = 5;

static Clusters<dogm::GridCell> computeDbscanClusters(const std::vector<Point<dogm::GridCell>>& cells_with_velocity)
{
    const DBSCAN<dogm::GridCell> dbscan(kMaximumDbscanNeighborDistance, kMinimumNumberOfNeighbors);
    return dbscan.cluster(cells_with_velocity);
}

PrecisionEvaluator::PrecisionEvaluator(const SimulationData _sim_data, const float _resolution, const float _grid_size)
    : sim_data{_sim_data}, resolution{_resolution}, grid_size{_grid_size}
{
    number_of_unassigned_detections = 0;
}

void PrecisionEvaluator::registerMetric(const std::string& name, std::unique_ptr<Metric> metric)
{
    metrics.emplace(name, std::move(metric));
}

void PrecisionEvaluator::evaluateAndStoreStep(int simulation_step_index,
                                              const std::vector<Point<dogm::GridCell>>& cells_with_velocity,
                                              bool print_current_precision)
{
    const auto groundtruth_vehicles = sim_data[simulation_step_index].vehicles;
    if (!cells_with_velocity.empty() && !groundtruth_vehicles.empty())
    {
        const auto clusters = computeDbscanClusters(cells_with_velocity);
        int cluster_id = 0;
        for (const auto& cluster : clusters)
        {
            PointWithVelocity cluster_mean = computeClusterMean(cluster);

            std::vector<Vehicle> matching_groundtruth_vehicles{};
            for (const auto& vehicle : groundtruth_vehicles)
            {
                const float distance =
                    sqrtf(powf(cluster_mean.x - vehicle.pos[0], 2.0f) + powf(cluster_mean.y - vehicle.pos[1], 2.0f));
                if (distance < kMaximumAssignmentDistance)
                {
                    matching_groundtruth_vehicles.push_back(vehicle);
                }
            }

            if (matching_groundtruth_vehicles.empty())
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

            PointWithVelocity current_error{};

            for (auto& metric : metrics)
            {
                // error should be the same for all metrics
                current_error = metric.second->addObjectDetection(cluster_mean, closest_vehicle);
            }

            if (print_current_precision)
            {
                std::cout << std::setprecision(2);
                std::cout << std::endl << "Cluster ID=" << cluster_id << std::endl;
                std::cout << "Vel. Err.: " << current_error.v_x << " " << current_error.v_y
                          << ", Pos. Err.: " << current_error.x << " " << current_error.y << std::endl;
            }
            cluster_id++;
        }
    }
}

PointWithVelocity PrecisionEvaluator::computeClusterMean(const Cluster<dogm::GridCell>& cluster)
{
    // TODO check if median is more precise than mean
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
    cluster_mean.v_y = (cluster_mean.v_y / cluster.size()) * resolution;

    // y as grid index is pointing downwards from top left corner.
    // y in world coordinates is pointing upwards from bottom left corner.
    // Therefore, vectors (velocity) just needs to be inverted. Positions (mean) must be inverted and translated.
    cluster_mean.v_y = -cluster_mean.v_y;
    cluster_mean.y = grid_size - cluster_mean.y;

    return cluster_mean;
}

void PrecisionEvaluator::printSummary()
{
    for (auto& metric : metrics)
    {
        std::cout << std::endl << metric.first << ": " << std::endl;
        PointWithVelocity error = metric.second->computeErrorStatistic();

        std::cout << "Position: " << error.x << " " << error.y << std::endl;
        std::cout << "Velocity: " << error.v_x << " " << error.v_y << std::endl;

        std::cout << std::endl;
    }

    std::cout << "Detections unassigned by evaluator: " << number_of_unassigned_detections << std::endl;
    std::cout << "Maximum possible detections: " << sim_data[0].vehicles.size() * sim_data.size() << std::endl;
}
