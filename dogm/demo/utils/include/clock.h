// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef CLOCK_H
#define CLOCK_H

#include "iclock.h"
#include <chrono>

class Clock : public IClock
{
public:
    std::chrono::steady_clock::time_point getCurrentTime() final { return std::chrono::steady_clock::now(); }
};

#endif  // CLOCK_H
