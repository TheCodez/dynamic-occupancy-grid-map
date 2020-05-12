SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

ADD_EXECUTABLE (utils_spec timer_spec.cpp)
TARGET_LINK_LIBRARIES (
    utils_spec
    PRIVATE utils
    PRIVATE gtest_main)

ADD_GTEST(utils_spec)
