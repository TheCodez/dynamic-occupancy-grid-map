// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef METRICS_H
#define METRICS_H

#include "precision_evaluator.h"

class Metric
{
public:
    Metric() {}
    virtual ~Metric(){};

    virtual void reset() {}
    virtual void update(const PointWithVelocity& cluster_mean, const Vehicle& vehicle) {}
    virtual void compute() {}

protected:
    PointWithVelocity computeError(const PointWithVelocity& cluster_mean, const Vehicle& vehicle)
    {
        PointWithVelocity error{};
        error.x += cluster_mean.x - vehicle.pos[0];
        error.y += cluster_mean.y - vehicle.pos[1];
        error.v_x += cluster_mean.v_x - vehicle.vel[0];
        error.v_y += cluster_mean.v_y - vehicle.vel[1];

        return error;
    }
};

class MeanSquaredError : public Metric
{
public:
    MeanSquaredError();
    virtual ~MeanSquaredError(){};

    virtual void update(const PointWithVelocity& cluster_mean, const Vehicle& vehicle) override;
    virtual void compute() override;

private:
    PointWithVelocity cumulative_error;
    int number_of_detections;
};

#endif  // METRICS_H