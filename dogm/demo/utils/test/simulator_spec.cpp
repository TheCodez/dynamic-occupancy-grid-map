// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "simulator.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>
#include <vector>

class VehicleSpecBase
{
protected:
    const float m_width{3.5};
    const glm::vec2 m_position{-12.3, 20.2};
    const glm::vec2 m_velocity{5, -10};
    Vehicle m_unit{m_width, m_position, m_velocity};
};

class VehicleSpec : public ::testing::Test, public VehicleSpecBase
{
};

TEST_F(VehicleSpec, Constructor)
{
    EXPECT_EQ(m_position, m_unit.pos);
    EXPECT_EQ(m_velocity, m_unit.vel);
}

TEST_F(VehicleSpec, GetFacingSide)
{
    const float resolution = 1.5f;
    const glm::vec2 leftmost_point = m_position + glm::vec2{-m_width * 0.5, 0.0f};
    const std::vector<glm::vec2> expected_points{leftmost_point, leftmost_point + glm::vec2{resolution, 0.0f},
                                                 leftmost_point + glm::vec2{2.0f * resolution, 0.0f}};
    EXPECT_EQ(expected_points, m_unit.getPointsOnFacingSide(resolution));
}

class VehicleSpecParametrized : public ::testing::TestWithParam<float>, public VehicleSpecBase
{
};

TEST_P(VehicleSpecParametrized, Move)
{
    const auto delta_time = GetParam();
    const auto expected_position = m_position + delta_time * m_velocity;
    m_unit.move(delta_time);
    EXPECT_EQ(expected_position, m_unit.pos);
    EXPECT_EQ(m_velocity, m_unit.vel);
}

INSTANTIATE_TEST_CASE_P(VehicleMoveTests, VehicleSpecParametrized, ::testing::Values(-3.2f, 0.0f, 3.4f, 10.5f));
