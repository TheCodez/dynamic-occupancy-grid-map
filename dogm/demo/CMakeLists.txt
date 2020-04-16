find_package(OpenCV)
if (NOT OpenCV_FOUND)
	set(OpenCV_INCLUDE_DIRS "" CACHE PATH "" )
	set(OpenCV_LIBS "" CACHE FILEPATH "" )
endif()

add_definitions(-D_USE_MATH_DEFINES)  # Required to make M_PI from cmath available in MSVC
add_definitions(-DGLEW_STATIC)

add_executable(demo
	main.cpp
	image_creation.h
	simulator.h
)

target_include_directories(demo PUBLIC ${OpenCV_INCLUDE_DIRS} ${GLM_INCLUDE_DIR})

target_link_libraries(demo
	dogm
	utils
	${OpenCV_LIBS}
        lGLEW -lglfw3 -lGL -lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -pthread
)

add_subdirectory(utils)
