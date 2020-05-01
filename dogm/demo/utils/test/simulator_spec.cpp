// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "simulator.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

TEST(Vehicle, Constructor)
{
    const int width{3};
    const glm::vec2 position{-10, 20};
    const glm::vec2 velocity{15, -25};

    Vehicle unit{width, position, velocity};

    EXPECT_EQ(position, unit.pos);
    EXPECT_EQ(velocity, unit.vel);
}
