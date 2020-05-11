SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

ADD_EXECUTABLE (simulator_spec simulator_spec.cpp)
TARGET_LINK_LIBRARIES (
    simulator_spec
    PRIVATE simulator
    PRIVATE gtest_main)

ADD_GTEST(simulator_spec)
