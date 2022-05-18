find_package(OpenCV)
if (NOT OpenCV_FOUND)
	set(OpenCV_INCLUDE_DIRS "" CACHE PATH "" )
	set(OpenCV_LIBS "" CACHE FILEPATH "" )
endif()

find_package(CUDA REQUIRED)

add_definitions(-D_USE_MATH_DEFINES)  # Required to make M_PI from cmath available in MSVC

add_executable(demo
	main.cpp
)

target_include_directories(demo PUBLIC
	${OpenCV_INCLUDE_DIRS}
	${GLM_INCLUDE_DIR}
)

target_link_libraries(demo
	dogm
	dogm_utils
	dogm_simulator
	${OpenCV_LIBS}
	${CUDA_CUDART_LIBRARY}
)

add_subdirectory(utils)
add_subdirectory(simulator)
