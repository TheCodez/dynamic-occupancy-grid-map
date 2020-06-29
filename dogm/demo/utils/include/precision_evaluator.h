// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef PRECISION_EVALUATOR_H
#define PRECISION_EVALUATOR_H

#include "dbscan.h"
#include "dogm/dogm_types.h"
#include "simulator.h"
#include "types.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

class Metric;

class PrecisionEvaluator
{
public:
    PrecisionEvaluator(const SimulationData _sim_data, const float _resolution, const float _grid_size);

    void registerMetric(const std::string& name, std::unique_ptr<Metric> metric);

    void evaluateAndStoreStep(int simulation_step_index, const std::vector<Point<dogm::GridCell>>& cells_with_velocity,
                              bool print_current_precision = false);
    void printSummary();

private:
    PointWithVelocity computeClusterMean(const Cluster<dogm::GridCell>& cluster);

    SimulationData sim_data;
    float resolution;
    float grid_size;
    int number_of_unassigned_detections;

    std::map<std::string, std::unique_ptr<Metric>> metrics;
};

#endif  // PRECISION_EVALUATOR_H
