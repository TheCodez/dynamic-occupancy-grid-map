cmake_minimum_required(VERSION 3.9)
project(dogm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build unit tests" OFF)

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

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

LIST (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
#INCLUDE(GoogleTestDownloadAndBuild)

include_directories(
	SYSTEM
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${GLFW3_INCLUDE_DIR}
	${GLEW_INCLUDE_DIRS}
	${GLM_INCLUDE_DIR}
	${OpenCV_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
)

set(HEADER_FILES
	include/dogm/kernel/init.h
	include/dogm/kernel/init_new_particles.h
	include/dogm/kernel/mass_update.h
	include/dogm/kernel/particle_to_grid.h
	include/dogm/kernel/predict.h
	include/dogm/kernel/resampling.h
	include/dogm/kernel/statistical_moments.h
	include/dogm/kernel/update_persistent_particles.h
	include/dogm/kernel/ego_motion_compensation.h
	include/dogm/common.h
	include/dogm/cuda_utils.h
	include/dogm/dogm.h
	include/dogm/dogm_types.h
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADER_FILES})

set(SRC_FILES
	src/kernel/init.cu
	src/kernel/init_new_particles.cu
	src/kernel/mass_update.cu
	src/kernel/particle_to_grid.cu
	src/kernel/predict.cu
	src/kernel/resampling.cu
	src/kernel/statistical_moments.cu
	src/kernel/update_persistent_particles.cu
	src/kernel/ego_motion_compensation.cu
	src/dogm.cu
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_FILES})

add_library(dogm SHARED
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

set_target_properties(dogm PROPERTIES PUBLIC_HEADER "include/dogm/dogm.h;include/dogm/dogm_types.h")

target_link_libraries(dogm
	${OPENGL_LIBRARY}
	${GLFW3_LIBRARY}
	${GLEW_LIBRARIES}
	${OpenCV_LIBS}
)

#INSTALL(TARGETS dogm
#		ARCHIVE DESTINATION lib/dogm
#        LIBRARY DESTINATION lib/dogm
#        PUBLIC_HEADER DESTINATION include/dogm
#)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

add_subdirectory(demo)

# Install package.xml for catkin
install(FILES package.xml DESTINATION share/${PROJECT_NAME}/)

set(CONF_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include")
set(DOGM_CMAKE_DIR share/${PROJECT_NAME}/cmake)
include(CMakePackageConfigHelpers)
configure_package_config_file(
	dogm-config.cmake.in
	${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}/dogm-config.cmake
	PATH_VARS DOGM_CMAKE_DIR
	INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}
)

install(
	FILES ${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}/dogm-config.cmake
	DESTINATION share/${PROJECT_NAME}/
)

install(TARGETS dogm
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
)

install(
	DIRECTORY include/
	DESTINATION include/
)
