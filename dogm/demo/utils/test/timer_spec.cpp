// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include <gtest/gtest.h>

#include "timer.h"
#include "IClock.h"
#include <chrono>
#include <string>
#include <thread>
#include <queue>

class ClockStub : public IClock {
public:
    ClockStub(std::initializer_list<std::size_t> time_diffs) {
        auto current_time = std::chrono::steady_clock::now();
        for (const auto &time_diff : time_diffs) {
            m_fake_time_points.push(current_time);
            current_time += std::chrono::milliseconds(time_diff);
            m_fake_time_points.push(current_time);
            current_time += std::chrono::milliseconds(1);  // TODO check if necessary
        }
        m_fake_time_points.push(current_time);  // for tic() call in toc()
    }

    std::chrono::steady_clock::time_point getCurrentTime() final {
        const auto result = m_fake_time_points.front();
        m_fake_time_points.pop();
        return result;
    }

private:
    std::queue<std::chrono::steady_clock::time_point> m_fake_time_points{};
};

TEST(Timer, ConstructorCallsTic) {
    std::string unit_name{"name"};
    std::string expected_output{unit_name + " took 1ms\n"};

    Timer unit{unit_name, std::make_unique<ClockStub>(std::initializer_list<std::size_t>{1U})};
    unit.toc();

    testing::internal::CaptureStdout();
    unit.printLastSplitMs();
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(expected_output, output);
}

class TimerFixture : public ::testing::Test {
protected:
    std::string m_unit_name{"name"};
//    Timer m_unit{m_unit_name};
    Timer m_unit{m_unit_name, std::make_unique<ClockStub>(std::initializer_list<std::size_t>{1U, 2U, 5U, 0U})};

    std::string getStdoutOfSplit() {
        testing::internal::CaptureStdout();
        m_unit.printLastSplitMs();
        return testing::internal::GetCapturedStdout();
    }

    std::string getStdoutOfStats() {
        testing::internal::CaptureStdout();
        m_unit.printStatsMs();
        return testing::internal::GetCapturedStdout();
    }
};

TEST_F(TimerFixture, TocCanCallPrint) {
    std::string expected_output{m_unit_name + " took 1ms\n"};

    testing::internal::CaptureStdout();
    m_unit.tic();
    m_unit.toc(true);

    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(expected_output, output);
}

TEST_F(TimerFixture, TocCallsTic) {
    std::string expected_output{m_unit_name + " took 2ms\n"};

    m_unit.toc();
    m_unit.toc();

    std::string output = getStdoutOfSplit();
    EXPECT_EQ(expected_output, output);
}

#if 0
void sleepFor2Ms() {
    sleepMilliseconds(2);
}

TEST_F(TimerFixture, TimeVoidFunctionCallWithoutArguments) {
    std::string expected_output{m_unit_name + " took 2ms\n"};
    testing::internal::CaptureStdout();

    m_unit.timeFunctionCall(true, sleepFor2Ms);

    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(expected_output, output);
}

TEST_F(TimerFixture, TimeVoidFunctionCallRecordsSplit) {
    std::string expected_output{m_unit_name + " took 2ms\n"};

    m_unit.timeFunctionCall(true, sleepFor2Ms);

    std::string output = getStdoutOfSplit();
    EXPECT_EQ(expected_output, output);
}

TEST_F(TimerFixture, TimeVoidFunctionCallWithoutArgumentsNoOutput) {
    testing::internal::CaptureStdout();

    m_unit.timeFunctionCall(false, sleepFor2Ms);

    const std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(std::string{}, output);
}

void sleepFor2MsWithDummyArguments(int /*unused*/, double /*unused*/) {
    sleepMilliseconds(2);
}

TEST_F(TimerFixture, TimeVoidFunctionCallWithArguments) {
    const std::string expected_output{m_unit_name + " took 2ms\n"};
    testing::internal::CaptureStdout();

    m_unit.timeFunctionCall(true, sleepFor2MsWithDummyArguments, 2, 3.4);

    const std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(expected_output, output);
}

int add(int a, int b) {
    return a + b;
}

TEST_F(TimerFixture, TimeFunctionCallWithArguments) {
    const std::string expected_output{m_unit_name + " took 0ms\n"};
    testing::internal::CaptureStdout();

    const auto result = m_unit.timeFunctionCall(true, add, 2, 3);

    ASSERT_EQ(5, result);
    const std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(expected_output, output);
}

TEST_F(TimerFixture, TimeFunctionCallWithArgumentsNoOutput) {
    testing::internal::CaptureStdout();

    const auto result = m_unit.timeFunctionCall(false, add, 2, 3);

    ASSERT_EQ(5, result);
    const std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(std::string{}, output);
}

TEST_F(TimerFixture, OneMillisecond) {
    std::string expected_output{m_unit_name + " took 1ms\n"};
    addSplit(1);

    std::string output = getStdoutOfSplit();
    EXPECT_EQ(expected_output, output);
}

TEST_F(TimerFixture, TwoMilliseconds) {
    std::string expected_output{m_unit_name + " took 2ms\n"};
    addSplit(2);

    std::string output = getStdoutOfSplit();
    EXPECT_EQ(expected_output, output);
}

TEST_F(TimerFixture, PrintStatsEmpty) {
    std::string expected_output{m_unit_name + " stats (0 splits):\n"};

    std::string output = getStdoutOfStats();
    EXPECT_EQ(expected_output, output);
}

TEST_F(TimerFixture, PrintStats) {
    std::string expected_output{m_unit_name + " stats (4 splits):\n" + "  Minimum: 1ms\n" + "  Median:  1ms\n" +
                                "  Mean:    2ms\n" + "  Maximum: 5ms\n\n"};

    addSplit(1);
    addSplit(1);
    addSplit(5);
    addSplit(1);

    std::string output = getStdoutOfStats();
    EXPECT_EQ(expected_output, output);
}
#endif