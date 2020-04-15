SET(HEADER_FILES include/dbscan.h include/precision_evaluator.h)

SET(SRC_FILES dbscan.cpp precision_evaluator.cpp)

ADD_LIBRARY(utils STATIC ${SRC_FILES} ${HEADER_FILES})

TARGET_INCLUDE_DIRECTORIES(utils
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/demo  # TODO remove once all utils headers are moved to the utils folder
    ${PROJECT_SOURCE_DIR}/include)
