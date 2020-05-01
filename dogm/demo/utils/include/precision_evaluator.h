// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef PRECISION_EVALUATOR_H
#define PRECISION_EVALUATOR_H

#include "dbscan.h"
#include "dogm/dogm_types.h"
#include "simulator.h"

#include <vector>

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
    PrecisionEvaluator(const SimulationData sim_data, const float resolution, const float grid_size);
    void evaluateAndStoreStep(int simulation_step_index, const std::vector<Point<dogm::GridCell>>& cells_with_velocity,
                              bool print_current_precision = false);
    void printSummary();

private:
    void accumulateErrors(const PointWithVelocity& error);
    PointWithVelocity computeClusterMean(const Cluster<dogm::GridCell>& cluster);

    SimulationData sim_data;
    float resolution;
    float grid_size;
    PointWithVelocity cumulative_error;
    int number_of_detections;
    int number_of_unassigned_detections;
};

#endif  // PRECISION_EVALUATOR_H
