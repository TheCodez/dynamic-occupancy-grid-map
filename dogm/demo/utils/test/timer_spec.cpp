// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include <gtest/gtest.h>

#include "timer.h"
#include <chrono>
#include <string>
#include <thread>

void mysleep(const unsigned int milliseconds)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

TEST(Timer, ConstructorCallsTic)
{
    std::string unit_name{"name"};
    std::string expected_output{unit_name + ": 1ms"};

    Timer unit{unit_name};
    mysleep(1);
    unit.toc();

    testing::internal::CaptureStdout();
    unit.printLastSplitMs();
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(expected_output, output);
}

class TimerFixture : public ::testing::Test
{
protected:
    std::string m_unit_name{"name"};
    Timer m_unit{m_unit_name};

    void addSplit(const unsigned int milliseconds)
    {
        m_unit.tic();
        mysleep(milliseconds);
        m_unit.toc();
    }

    std::string getStdoutOfSplit()
    {
        testing::internal::CaptureStdout();
        m_unit.printLastSplitMs();
        return testing::internal::GetCapturedStdout();
    }

    std::string getStdoutOfStats()
    {
        testing::internal::CaptureStdout();
        m_unit.printStatsMs();
        return testing::internal::GetCapturedStdout();
    }
};

TEST_F(TimerFixture, TocCanCallPrint)
{
    std::string expected_output{m_unit_name + ": 1ms"};
    testing::internal::CaptureStdout();
    m_unit.tic();
    mysleep(1);
    m_unit.toc(true);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(expected_output, output);
}

TEST_F(TimerFixture, TocCallsTic)
{
    std::string expected_output{m_unit_name + ": 2ms"};

    mysleep(1);
    m_unit.toc();
    mysleep(2);
    m_unit.toc();

    std::string output = getStdoutOfSplit();

    EXPECT_EQ(expected_output, output);
}

TEST_F(TimerFixture, OneMillisecond)
{
    addSplit(1);
    std::string expected_output{m_unit_name + ": 1ms"};

    std::string output = getStdoutOfSplit();

    EXPECT_EQ(expected_output, output);
}

TEST_F(TimerFixture, TwoMilliseconds)
{
    addSplit(2);
    std::string expected_output{m_unit_name + ": 2ms"};

    std::string output = getStdoutOfSplit();

    EXPECT_EQ(expected_output, output);
}

TEST_F(TimerFixture, PrintStatsEmpty)
{
    std::string expected_output{m_unit_name + " stats (0 splits):\n"};

    std::string output = getStdoutOfStats();
    EXPECT_EQ(expected_output, output);
}

TEST_F(TimerFixture, PrintStats)
{
    addSplit(1);
    addSplit(1);
    addSplit(5);
    addSplit(1);
    std::string expected_output{m_unit_name + " stats (4 splits):\n" + "  Minimum: 1ms\n" + "  Median:  1ms\n" +
                                "  Mean:    2ms\n" + "  Maximum: 5ms\n"};

    std::string output = getStdoutOfStats();
    EXPECT_EQ(expected_output, output);
}
