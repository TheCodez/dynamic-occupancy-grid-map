# Download and unpack googletest at configure time using the generic
# donwload_project macro
LIST (APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
INCLUDE (DownloadProject)

DOWNLOAD_PROJECT (
  PROJ
  googletest
  GIT_REPOSITORY
  https://github.com/google/googletest.git
  GIT_TAG
  release-1.10.0
  SOURCE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/googletest-src"
  BINARY_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/googletest-build"
  ${UPDATE_DISCONNECTED_IF_AVAILABLE})
ADD_SUBDIRECTORY (${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
INCLUDE_DIRECTORIES (SYSTEM ${googletest_SOURCE_DIR})

# Target must already exist
MACRO (add_gtest TESTNAME)
  TARGET_LINK_LIBRARIES (
    ${TESTNAME}
    PUBLIC gtest
    PUBLIC gmock
    PUBLIC gtest_main)

  IF (GOOGLE_TEST_INDIVIDUAL)
    IF (CMAKE_VERSION VERSION_LESS 3.10)
      GTEST_ADD_TESTS (TARGET ${TESTNAME} TEST_PREFIX "${TESTNAME}." TEST_LIST
                       TmpTestList)
      SET_TESTS_PROPERTIES (${TmpTestList} PROPERTIES FOLDER "Tests")
    ELSE ()
      GTEST_DISCOVER_TESTS (${TESTNAME} TEST_PREFIX "${TESTNAME}." PROPERTIES
                            FOLDER "Tests")
    ENDIF ()
  ELSE ()
    ADD_TEST (${TESTNAME} ${TESTNAME})
    SET_TARGET_PROPERTIES (${TESTNAME} PROPERTIES FOLDER "Tests")
  ENDIF ()

  INSTALL (TARGETS ${executable_name} RUNTIME DESTINATION ${executable_name})
ENDMACRO ()
