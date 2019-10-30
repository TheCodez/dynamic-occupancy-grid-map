cmake_minimum_required(VERSION 3.9)
project(occupancy_grid_map CXX CUDA)

find_package(OpenGL REQUIRED)
#find_package(glm REQUIRED)
#find_package(GLFW3 REQUIRED)
#find_package(GLEW REQUIRED)
#find_package(OpenCV REQUIRED)

set(GLFW_INCLUDE_DIR "" CACHE PATH "" )
set(GLFW_LIBRARY "" CACHE FILEPATH "" )

set(GLEW_INCLUDE_DIR "" CACHE PATH "" )
set(GLEW_LIBRARY "" CACHE FILEPATH "" )

set(OPENCV_INCLUDE_DIR "" CACHE PATH "" )
set(OPENCV_LIBRARY "" CACHE FILEPATH "" )

set(GLM_INCLUDE_DIR "" CACHE PATH "" )

include_directories(
	include
	${GLFW_INCLUDE_DIR}
	${GLEW_INCLUDE_DIR}
	${GLM_INCLUDE_DIR}
	${OPENCV_INCLUDE_DIR}
)

set(HEADERS_FILES
    include/opengl/texture.h
    include/opengl/framebuffer.h
    include/opengl/polygon.h
    include/opengl/renderer.h
    include/opengl/shader.h
	include/kernel/init.h
	include/kernel/init_new_particles.h
	include/kernel/mass_update.h
	include/kernel/measurement_grid.h
	include/kernel/particle_to_grid.h
	include/kernel/predict.h
	include/kernel/resampling.h
	include/kernel/statistical_moments.h
	include/kernel/update_persistent_particles.h
	include/common.h
	include/cuda_utils.h
	include/occupancy_grid_map.h
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADERS_FILES})

set(SRC_FILES
    src/opengl/texture.cpp
    src/opengl/framebuffer.cpp
    src/opengl/polygon.cpp
    src/opengl/renderer.cpp
    src/opengl/shader.cpp
	src/kernel/init.cu
	src/kernel/init_new_particles.cu
	src/kernel/mass_update.cu
	src/kernel/measurement_grid.cu
	src/kernel/particle_to_grid.cu
	src/kernel/predict.cu
	src/kernel/resampling.cu
	src/kernel/statistical_moments.cu
	src/kernel/update_persistent_particles.cu
	src/occupancy_grid_map.cu
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_FILES})

add_library(occupancy_grid_map STATIC
    ${HEADERS_FILES}
	${SRC_FILES}
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CUDA_PROPAGATE_HOST_FLAGS ON)

set_target_properties(occupancy_grid_map PROPERTIES PUBLIC_HEADER include/occupancy_grid_map.h)

target_link_libraries(occupancy_grid_map
	${OPENGL_LIBRARY}
	${GLFW_LIBRARY}
	${GLEW_LIBRARY}
)

add_executable(demo
	demo/main.cpp
	demo/measurements.h
)

target_link_libraries(demo
	occupancy_grid_map
	${OPENCV_LIBRARY}
)
