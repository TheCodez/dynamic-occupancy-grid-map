find_package(OpenGL REQUIRED)

find_package(GLM)
if (NOT GLM_FOUND)
	set(GLM_INCLUDE_DIR "" CACHE PATH "")
endif()

find_package(GLFW3)
if (NOT GLFW3_FOUND)
	set(GLFW3_INCLUDE_DIR "" CACHE PATH "")
	set(GLFW3_LIBRARY "" CACHE FILEPATH "" )
endif()

if (NOT WIN32)
	find_package(GLEW)
endif()
if (NOT GLEW_FOUND)
	set(GLEW_INCLUDE_DIRS "" CACHE PATH "")
	set(GLEW_LIBRARIES "" CACHE FILEPATH "" )
endif()

SET(HEADER_FILES
    include/simulator.h
    include/mapping/laser_to_meas_grid.h
    include/mapping/opengl/texture.h
    include/mapping/opengl/framebuffer.h
    include/mapping/opengl/polygon.h
    include/mapping/opengl/renderer.h
    include/mapping/opengl/shader.h
    include/mapping/kernel/measurement_grid.h
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADER_FILES})

SET(SRC_FILES
    simulator.cpp
    mapping/laser_to_meas_grid.cu
    mapping/opengl/texture.cpp
    mapping/opengl/framebuffer.cpp
    mapping/opengl/polygon.cpp
    mapping/opengl/renderer.cpp
    mapping/opengl/shader.cpp
    mapping/kernel/measurement_grid.cu
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_FILES})

ADD_LIBRARY(simulator STATIC ${SRC_FILES} ${HEADER_FILES})

TARGET_INCLUDE_DIRECTORIES(simulator
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
