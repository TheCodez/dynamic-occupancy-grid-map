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

constexpr int UNCLASSIFIED = -1;
constexpr int NOISE = -2;

template <typename T>
class DBSCAN
{
public:
    DBSCAN(const std::vector<Point<T>>& points) : points(points) {}

    void cluster(float eps, int min_cells)
    {
        int cluster_id = 0;

        for (auto& point : points)
        {
            if (point.cluster_id == UNCLASSIFIED)
            {
                if (expandCluster(point, cluster_id, eps, min_cells))
                {
                    cluster_id++;
                }
            }
        }

        num_clusters = cluster_id;
    }

    std::map<int, std::vector<Point<T>>> getClusteredPoints() const
    {
        std::map<int, std::vector<Point<T>>> result;
        for (int i = 0; i < getNumCluster(); i++)
        {
            result.emplace(i, std::vector<Point<T>>());
        }

        for (auto& point : getPoints())
        {
            for (int i = 0; i < getNumCluster(); i++)
            {
                if (point.cluster_id == i)
                {
                    auto& cluster = result.at(i);
                    cluster.push_back(point);
                }
            }
        }

        return result;
    }

    std::vector<Point<T>> getPoints() const { return points; }
    int getNumCluster() const { return num_clusters; }

private:
    bool expandCluster(Point<T>& point, int cluster_id, float eps, int min_cells)
    {
        std::vector<Point<T>> seeds = regionQuery(point, eps);
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

            std::vector<Point<T>> result = regionQuery(current_cell, eps);
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

    std::vector<Point<T>> regionQuery(const Point<T>& q, float eps)
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

    std::vector<Point<T>> points;
    int num_clusters;
};

#endif  // DBSCAN_H
