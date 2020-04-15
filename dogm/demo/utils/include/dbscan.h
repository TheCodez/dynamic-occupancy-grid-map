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
    Clusters<T> cluster(const std::vector<Point<T>>& points);

private:
    bool expandCluster(std::vector<Point<T>>& points, Point<T>& point, int cluster_id);
    std::vector<Point<T>> regionQuery(const std::vector<Point<T>>& points, const Point<T>& q);

    float eps;
    int min_cells;
};

#endif  // DBSCAN_H
