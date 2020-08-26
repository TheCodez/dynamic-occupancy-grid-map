// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#ifndef ICLOCK_H
#define ICLOCK_H

#include <chrono>

class IClock
{
public:
    virtual std::chrono::steady_clock::time_point getCurrentTime() = 0;
    virtual ~IClock() = default;
};

#endif  // ICLOCK_H
