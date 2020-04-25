SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

ADD_EXECUTABLE (dogm_spec dogm_spec.cpp)
TARGET_LINK_LIBRARIES (
    dogm_spec
    PRIVATE dogm
    PRIVATE gtest_main)

ADD_GTEST(dogm_spec)
