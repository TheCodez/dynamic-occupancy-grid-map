# Locate the DOGM library
#
# This module defines the following variables:
#
# DOGM_LIBRARIES the name of the libraries;
# DOGM_INCLUDE_DIRS where to find dogm include files.
# DOGM_FOUND true if both the DOGM_LIBRARIES and DOGM_INCLUDE_DIR have been found.
#

#Find dogm library
FIND_LIBRARY(DOGM_LIBRARIES 
  NAMES dogm
  PATHS 
  "/usr/local/lib/dogm"
  "/usr/lib/dogm"
  )

#Find dogm header
FIND_PATH(DOGM_INCLUDE_DIRS
  PATHS
  "/usr/local/include/dogm"
  "/usr/include/dogm"
)

SET(DOGM_FOUND "NO")
IF(DOGM_LIBRARIES)
  IF(DOGM_INCLUDE_DIRS)
    SET(DOGM_FOUND "YES")
  ENDIF(DOGM_INCLUDE_DIRS)
ENDIF(DOGM_LIBRARIES)

#Mark options as advanced
MARK_AS_ADVANCED(
  DOGM_INCLUDE_DIRS
  DOGM_LIBRARIES
)