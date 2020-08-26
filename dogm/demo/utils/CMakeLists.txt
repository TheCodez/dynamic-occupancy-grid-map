find_package(OpenCV)
if (NOT OpenCV_FOUND)
	set(OpenCV_INCLUDE_DIRS "" CACHE PATH "" )
	set(OpenCV_LIBS "" CACHE FILEPATH "" )
endif()

SET(HEADER_FILES
    include/color_wheel_adder.h
    include/clock.h
    include/dbscan.h
    include/iclock.h
    include/image_creation.h
    include/metrics.h
    include/precision_evaluator.h
    include/timer.h
    include/types.h
)

SET(SRC_FILES
    color_wheel_adder.cpp
    dbscan.cpp
    image_creation.cpp
    metrics.cpp
    precision_evaluator.cpp
    timer.cpp
)

ADD_LIBRARY(utils STATIC ${SRC_FILES} ${HEADER_FILES})

TARGET_COMPILE_FEATURES(utils PUBLIC cxx_std_17)

TARGET_INCLUDE_DIRECTORIES(utils
    PUBLIC
    simulator
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(utils PRIVATE simulator ${OpenCV_LIBS})

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
