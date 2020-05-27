// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef METRICS_H
#define METRICS_H

#include "simulator.h"
#include "types.h"

class Metric
{
public:
    Metric() : cumulative_error{}, number_of_detections(0) {}
    virtual ~Metric() = default;

    virtual void reset()
    {
        cumulative_error = {};
        number_of_detections = 0;
    }
    virtual PointWithVelocity addObjectDetection(const PointWithVelocity& cluster_mean, const Vehicle& vehicle) = 0;
    virtual PointWithVelocity computeErrorStatistic() = 0;

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

    PointWithVelocity cumulative_error;
    int number_of_detections;
};

class MAE : public Metric
{
public:
    MAE() = default;
    virtual ~MAE() = default;

    virtual PointWithVelocity addObjectDetection(const PointWithVelocity& cluster_mean,
                                                 const Vehicle& vehicle) override;
    virtual PointWithVelocity computeErrorStatistic() override;
};

class RMSE : public Metric
{
public:
    RMSE() = default;
    virtual ~RMSE() = default;

    virtual PointWithVelocity addObjectDetection(const PointWithVelocity& cluster_mean,
                                                 const Vehicle& vehicle) override;
    virtual PointWithVelocity computeErrorStatistic() override;
};

#endif  // METRICS_H