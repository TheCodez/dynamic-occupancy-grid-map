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

#ifndef DBSCAN_H
#define DBSCAN_H

#include <algorithm>
#include <map>
#include <vector>

template <typename T>
struct Point
{
    float x, y;
    T data;
    int cluster_id;

    bool operator==(const Point<T>& other) { return x == other.x && y == other.y; }
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

    Clusters<T> cluster(const std::vector<Point<T>>& points)
    {
        std::vector<Point<T>> result_points = points;

        int cluster_id = NOISE + 1;

        for (auto& point : result_points)
        {
            if (point.cluster_id == UNCLASSIFIED)
            {
                if (expandCluster(result_points, point, cluster_id, eps, min_cells))
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

private:
    bool expandCluster(std::vector<Point<T>>& points, Point<T>& point, int cluster_id, float eps, int min_cells)
    {
        std::vector<Point<T>> seeds = regionQuery(points, point, eps);
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

            std::vector<Point<T>> result = regionQuery(points, current_cell, eps);
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

    std::vector<Point<T>> regionQuery(const std::vector<Point<T>>& points, const Point<T>& q, float eps)
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

    float distance(const Point<T>& q, const Point<T>& p) { return sqrtf(powf(q.x - p.x, 2) + powf(q.y - p.y, 2)); }

    float eps;
    int min_cells;
};

#endif  // DBSCAN_H
