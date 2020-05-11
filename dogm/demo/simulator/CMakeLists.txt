SET(HEADER_FILES
    include/simulator.h
)

SET(SRC_FILES
    simulator.cpp
)

ADD_LIBRARY(simulator STATIC ${SRC_FILES} ${HEADER_FILES})

TARGET_INCLUDE_DIRECTORIES(simulator
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
