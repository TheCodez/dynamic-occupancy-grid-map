find_package(OpenCV)
if (NOT OpenCV_FOUND)
	set(OpenCV_INCLUDE_DIRS "" CACHE PATH "" )
	set(OpenCV_LIBS "" CACHE FILEPATH "" )
endif()

add_executable(demo
	main.cpp
)

target_include_directories(demo PUBLIC ${OpenCV_INCLUDE_DIRS} ${GLM_INCLUDE_DIR} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(demo
	dogm
	${OpenCV_LIBS}
)
