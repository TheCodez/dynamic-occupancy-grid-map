// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>
#include <vector>

class Timer
{
public:
    Timer(const std::string& name) : m_name{name} { tic(); }
    void tic();
    void toc(const bool print_split = false);

    template<typename FunctionType, typename... ArgumentTypes>
    void timeVoidFunctionCall(const bool print_split, FunctionType&& function, ArgumentTypes&&... args);

    int getLastSplitMs() const;
    void printLastSplitMs() const;
    void printStatsMs() const;

private:
    const std::string m_name;
    std::vector<std::chrono::nanoseconds> m_splits;
    std::chrono::high_resolution_clock::time_point m_current_start;
};

#endif  // TIMER_H
