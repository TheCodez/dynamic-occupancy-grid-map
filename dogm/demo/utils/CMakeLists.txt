cmake_minimum_required(VERSION 3.16)

project(templib)

option(BUILD_TESTS "Build unit tests" ON)

LIST (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)
INCLUDE(GoogleTestDownloadAndBuild)

SET(HEADER_FILES
    include/timer.h
)

SET(SRC_FILES
    timer.cpp
)

ADD_LIBRARY(utils STATIC ${SRC_FILES} ${HEADER_FILES})

TARGET_COMPILE_FEATURES(utils PUBLIC cxx_std_17)

TARGET_INCLUDE_DIRECTORIES(utils
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
