// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dbscan.h"
#include "dogm/dogm_types.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

// Template instantiation to avoid linker errors, see
// https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
template Clusters<dogm::GridCell>
DBSCAN<dogm::GridCell>::cluster(const std::vector<Point<dogm::GridCell>>& points) const;

template <typename T>
static float distance(const Point<T>& q, const Point<T>& p)
{
    return sqrtf(powf(q.x - p.x, 2) + powf(q.y - p.y, 2));
}

template <typename T>
Clusters<T> DBSCAN<T>::cluster(const std::vector<Point<T>>& points) const
{
    std::vector<Point<T>> result_points = points;

    int cluster_id = NOISE + 1;

    for (auto& point : result_points)
    {
        if (point.cluster_id == UNCLASSIFIED)
        {
            if (expandCluster(result_points, point, cluster_id))
            {
                cluster_id++;
            }
        }
    }

    Clusters<T> clusters(cluster_id, Cluster<T>());
    for (auto& point : result_points)
    {
        for (int i = 0; i < clusters.size(); i++)
        {
            if (point.cluster_id == i)
            {
                clusters.at(i).push_back(point);
            }
        }
    }

    return clusters;
}

template <typename T>
bool DBSCAN<T>::expandCluster(std::vector<Point<T>>& points, Point<T>& point, int cluster_id) const
{
    std::vector<Point<T>> seeds = regionQuery(points, point);
    if (seeds.size() < min_cells)
    {
        point.cluster_id = NOISE;
        return false;
    }

    for (auto seeds_iter = seeds.begin(); seeds_iter != seeds.end(); seeds_iter++)
    {
        points.at(std::distance(seeds.begin(), seeds_iter)).cluster_id = cluster_id;
    }
    seeds.erase(std::remove(seeds.begin(), seeds.end(), point), seeds.end());

    while (!seeds.empty())
    {
        Point<T> current_cell = seeds.front();

        std::vector<Point<T>> result = regionQuery(points, current_cell);
        if (result.size() >= min_cells)
        {
            for (auto& result_point : result)
            {
                if (result_point.cluster_id == UNCLASSIFIED || result_point.cluster_id == NOISE)
                {
                    if (result_point.cluster_id == UNCLASSIFIED)
                    {
                        seeds.push_back(result_point);
                    }

                    auto elem_it = std::find(points.begin(), points.end(), result_point);
                    points.at(std::distance(points.begin(), elem_it)).cluster_id = cluster_id;
                }
            }
        }
        seeds.erase(std::remove(seeds.begin(), seeds.end(), current_cell), seeds.end());
    }

    return true;
}

template <typename T>
std::vector<Point<T>> DBSCAN<T>::regionQuery(const std::vector<Point<T>>& points, const Point<T>& q) const
{
    std::vector<Point<T>> neighbors;

    for (auto& p : points)
    {
        if (distance(q, p) <= eps)
        {
            neighbors.push_back(p);
        }
    }

    return neighbors;
}
