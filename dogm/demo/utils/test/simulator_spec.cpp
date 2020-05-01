// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "simulator.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

class VehicleSpec : public ::testing::Test
{
protected:
    const float m_width{3.5};
    const glm::vec2 m_position{-10, 20};
    const glm::vec2 m_velocity{5, -10};
    Vehicle m_unit{m_width, m_position, m_velocity};
};

TEST_F(VehicleSpec, Constructor)
{
    EXPECT_EQ(m_position, m_unit.pos);
    EXPECT_EQ(m_velocity, m_unit.vel);
}

class VehicleSpecParametrized : public ::testing::TestWithParam<float>
{
protected:
    const float m_width{3.5};
    const glm::vec2 m_position{-12.3, 20.2};
    const glm::vec2 m_velocity{5, -10};
    Vehicle m_unit{m_width, m_position, m_velocity};
};

TEST_P(VehicleSpecParametrized, Move)
{
    const auto delta_time = GetParam();
    const auto expected_position = m_position + delta_time * m_velocity;
    m_unit.move(delta_time);
    EXPECT_EQ(expected_position, m_unit.pos);
    EXPECT_EQ(m_velocity, m_unit.vel);
}

INSTANTIATE_TEST_CASE_P(VehcileMoveTests, VehicleSpecParametrized, ::testing::Values(-3.2f, 0.0f, 3.4f, 10.5f));

TEST_F(VehicleSpec, GetFacingSide)
{
    EXPECT_TRUE(true);
}