cmake_minimum_required(VERSION 3.9)
project(DOGM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

option(BUILD_TESTS "Build unit tests" ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
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

LIST (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
INCLUDE(GoogleTestDownloadAndBuild)

#if(WIN32)
#    install(FILES "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}d.dll" DESTINATION ${CMAKE_INSTALL_BINARY}/debug CONFIGURATIONS Debug)
#    install(FILES "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}.dll" DESTINATION ${CMAKE_INSTALL_BINARY} CONFIGURATIONS RelWithDebInfo Release)
#endif(WIN32)

include_directories(
	SYSTEM
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${GLFW3_INCLUDE_DIR}
	${GLEW_INCLUDE_DIRS}
	${GLM_INCLUDE_DIR}
)

set(HEADER_FILES
	include/dogm/opengl/texture.h
	include/dogm/opengl/framebuffer.h
	include/dogm/opengl/polygon.h
	include/dogm/opengl/renderer.h
	include/dogm/opengl/shader.h
	include/dogm/kernel/init.h
	include/dogm/kernel/init_new_particles.h
	include/dogm/kernel/mass_update.h
	include/dogm/kernel/measurement_grid.h
	include/dogm/kernel/particle_to_grid.h
	include/dogm/kernel/predict.h
	include/dogm/kernel/resampling.h
	include/dogm/kernel/statistical_moments.h
	include/dogm/kernel/update_persistent_particles.h
	include/dogm/common.h
	include/dogm/cuda_utils.h
	include/dogm/dogm.h
	include/dogm/dogm_types.h
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADER_FILES})

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
    ${HEADER_FILES}
	${SRC_FILES}
)

target_include_directories(dogm PUBLIC include)

# support for gpu lambdas
string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda")
# supress glm warnings
#string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
set(CUDA_PROPAGATE_HOST_FLAGS ON)

set_target_properties(dogm PROPERTIES PUBLIC_HEADER "include/dogm.h;include/dogm_types.h")

target_link_libraries(dogm
	PRIVATE ${OPENGL_LIBRARY}
	PRIVATE ${GLFW3_LIBRARY}
	PRIVATE ${GLEW_LIBRARIES}
)

#INSTALL(TARGETS dogm
#		ARCHIVE DESTINATION lib/dogm
#        LIBRARY DESTINATION lib/dogm
#        PUBLIC_HEADER DESTINATION include/dogm
#)

add_subdirectory(demo)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
