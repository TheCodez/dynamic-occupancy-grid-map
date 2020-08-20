// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "timer.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>

static int castToMilliseconds(const std::chrono::nanoseconds& nanoseconds)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(nanoseconds).count();
}

void Timer::tic()
{
    m_current_start = m_clock->getCurrentTime();
}

void Timer::toc(const bool print_split)
{
    m_splits.push_back(m_clock->getCurrentTime() - m_current_start);
    if (print_split)
    {
        printLastSplitMs();
    }
    tic();
}

int Timer::getLastSplitMs() const
{
    return castToMilliseconds(m_splits.back());
}

void Timer::printLastSplitMs() const
{
    std::cout << m_name << " took " << getLastSplitMs() << "ms\n";
}

void Timer::printStatsMs() const
{
    std::cout << m_name << " stats (" << m_splits.size() << " splits):\n";
    if (!m_splits.empty())
    {
        std::vector<unsigned int> splits_ms{};
        for (const auto& split : m_splits)
        {
            splits_ms.push_back(castToMilliseconds(split));
        }

        std::sort(splits_ms.begin(), splits_ms.end());

        const auto minimum = splits_ms.front();
        const auto median = splits_ms[splits_ms.size() / 2];
        const auto mean = std::accumulate(splits_ms.begin(), splits_ms.end(), 0.0f) / splits_ms.size();
        const auto maximum = splits_ms.back();

        std::cout << "  Minimum: " << minimum << "ms\n";
        std::cout << "  Median:  " << median << "ms\n";
        std::cout << "  Mean:    " << mean << "ms\n";
        std::cout << "  Maximum: " << maximum << "ms\n\n";
    }
}
