// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>

template <typename T>
struct Point
{
    float x, y;
    T data;
    int cluster_id;

    bool operator==(const Point<T>& other) const { return x == other.x && y == other.y; }
};

constexpr int UNCLASSIFIED = -2;
constexpr int NOISE = -1;

template <typename T>
using Cluster = std::vector<Point<T>>;

template <typename T>
using Clusters = std::vector<Cluster<T>>;

template <typename T>
class DBSCAN
{
public:
    DBSCAN(float eps, int min_cells) : eps(eps), min_cells(min_cells) {}
    Clusters<T> cluster(const std::vector<Point<T>>& points) const;

private:
    bool expandCluster(std::vector<Point<T>>& points, Point<T>& point, int cluster_id) const;
    std::vector<Point<T>> regionQuery(const std::vector<Point<T>>& points, const Point<T>& q) const;

    float eps;
    int min_cells;
};

#endif  // DBSCAN_H
