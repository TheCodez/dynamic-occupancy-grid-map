cmake_minimum_required(VERSION 3.9)
project(DOGM CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

option(BUILD_TESTS "Build tests" ON)

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

find_package(GLEW)
if (NOT GLEW_FOUND)
	set(GLEW_INCLUDE_DIRS "" CACHE PATH "")
	set(GLEW_LIBRARY "" CACHE FILEPATH "" )
endif()

find_package(OpenCV)
if (NOT OpenCV_FOUND)
	set(OpenCV_INCLUDE_DIRS "" CACHE PATH "" )
	set(OpenCV_LIBS "" CACHE FILEPATH "" )
endif()

#if(WIN32)
#    install(FILES "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}d.dll" DESTINATION ${CMAKE_INSTALL_BINARY}/debug CONFIGURATIONS Debug)
#    install(FILES "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}.dll" DESTINATION ${CMAKE_INSTALL_BINARY} CONFIGURATIONS RelWithDebInfo Release)
#endif(WIN32)

include_directories(
	include
	${CUDA_INCLUDE_DIRS}
	${GLFW3_INCLUDE_DIR}
	${GLEW_INCLUDE_DIRS}
	${GLM_INCLUDE_DIR}
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
	include/dogm.h
	include/dogm_types.h
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
	src/dogm.cu
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_FILES})

add_library(dogm STATIC
    ${HEADERS_FILES}
	${SRC_FILES}
)

set(CMAKE_CXX_STANDARD 11)

# support for gpu lambdas
string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda")
# supress glm warnings
#string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
set(CUDA_PROPAGATE_HOST_FLAGS ON)

set_target_properties(dogm PROPERTIES PUBLIC_HEADER "include/dogm.h;include/dogm_types.h")

target_link_libraries(dogm
	${OPENGL_LIBRARY}
	${GLFW3_LIBRARY}
	${GLEW_LIBRARY}
)

INSTALL(TARGETS dogm
		ARCHIVE DESTINATION lib/dogm
        LIBRARY DESTINATION lib/dogm
        PUBLIC_HEADER DESTINATION include/dogm
)

add_executable(demo
	demo/main.cpp
)

target_include_directories(demo PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(demo
	dogm
	${OpenCV_LIBS}
)

#if (BUILD_TESTS)
#    enable_testing()
#    add_subdirectory(tests)
#endif()
