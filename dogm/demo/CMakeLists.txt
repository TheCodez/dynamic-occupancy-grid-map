find_package(OpenCV)
if (NOT OpenCV_FOUND)
	set(OpenCV_INCLUDE_DIRS "" CACHE PATH "" )
	set(OpenCV_LIBS "" CACHE FILEPATH "" )
endif()

add_definitions(-D_USE_MATH_DEFINES)

add_executable(demo
	main.cpp
	dbscan.h
	image_creation.h
	simulator.h
)

target_include_directories(demo PUBLIC ${OpenCV_INCLUDE_DIRS} ${GLM_INCLUDE_DIR})

target_link_libraries(demo
	dogm
	utils
	${OpenCV_LIBS}
)

add_subdirectory(utils)
