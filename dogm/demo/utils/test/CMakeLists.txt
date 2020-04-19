ADD_EXECUTABLE (utils_spec simulator_spec.cpp)
TARGET_LINK_LIBRARIES (
    utils_spec
    PRIVATE utils
    PRIVATE gtest_main)

ADD_GTEST(utils_spec)
