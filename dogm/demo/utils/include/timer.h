// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "iclock.h"

class Timer
{
public:
    Timer(std::string name, std::unique_ptr<IClock> clock) : m_name{std::move(name)}, m_clock{std::move(clock)}
    {
        tic();
    }

    void tic();
    void toc(const bool print_split = false);

    template <typename FunctionType, typename... ArgumentTypes>
    auto timeFunctionCall(const bool print_split, FunctionType&& function, ArgumentTypes&&... arguments)
        -> decltype(function(std::declval<ArgumentTypes>()...))
    {
        tic();
        constexpr auto has_function_void_return_type =
            std::is_same<void, decltype(function(std::declval<ArgumentTypes>()...))>::value;
        if constexpr (has_function_void_return_type)
        {
            function(std::forward<decltype(arguments)>(arguments)...);
            toc(print_split);
            return;
        }
        else
        {
            const auto result = function(std::forward<decltype(arguments)>(arguments)...);
            toc(print_split);
            return result;
        }
    }

    int getLastSplitMs() const;
    void printLastSplitMs() const;
    void printStatsMs() const;

private:
    const std::string m_name;
    std::vector<std::chrono::nanoseconds> m_splits;
    std::chrono::steady_clock::time_point m_current_start;
    std::unique_ptr<IClock> m_clock;
};

#endif  // TIMER_H
