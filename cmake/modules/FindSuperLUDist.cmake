# Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-745557. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the ParElag library. For more information and source code
# availability see http://github.com/LLNL/parelag.
#
# ParElag is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Sets the following variables
#   - SuperLUDist_FOUND
#   - SuperLUDist_INCLUDE_DIRS
#   - SuperLUDist_LIBRARIES
#   - SuperLUDist_HAVE_VERSION_5
#
# Imported target(s): SLU::dist

find_package(ParMETIS QUIET)

# Look for slu_ddefs.h
find_path(SuperLUDist_INCLUDE_DIRS superlu_defs.h
  HINTS ${SuperLUDist_DIR} $ENV{SuperLUDist_DIR}
  PATH_SUFFIXES include SRC SuperLU
  DOC "Directory where SuperLUDist defs headers live."
  NO_DEFAULT_PATH)
find_path(SuperLUDist_INCLUDE_DIRS slu_ddefs.h)

# Now try to find the library
find_library(SuperLUDist_LIBRARY
  NAMES superludist superlu_dist superlu_dist_4.3 NAMES_PER_DIR
  HINTS ${SuperLUDist_DIR} $ENV{SuperLUDist_DIR} ${SuperLUDist_LIBRARY_DIRS}
  PATH_SUFFIXES lib SRC
  DOC "Distributed SuperLU library."
  NO_DEFAULT_PATH)
find_library(SuperLUDist_LIBRARY
  NAMES superludist superlu_dist superlu_dist_4.3 NAMES_PER_DIR)

# Test the SuperLUDist version
include(CheckCXXSourceCompiles)
function(check_superlu_dist_version_5 VAR)
  set(TEST_SOURCE
    "
#include <superlu_defs.h>

int main()
{
  superlu_dist_options_t opts;
  return 0;
}
")
  set(CMAKE_REQUIRED_INCLUDES ${MPI_C_INCLUDE_PATH} ${SuperLUDist_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARY ${MPI_C_LIBRARY})
  check_cxx_source_compiles("${TEST_SOURCE}" ${VAR})
endfunction()

check_superlu_dist_version_5(${PROJECT_NAME}_SuperLUDist_HAVE_VERSION_5)

# Add imported target
if (NOT TARGET SLU::dist)
  # Check if we have shared or static libraries
  include(ParELAGCMakeUtilities)
  parelag_determine_library_type(${SuperLUDist_LIBRARY} SLUDist_LIB_TYPE)
  
  add_library(SLU::dist ${SLUDist_LIB_TYPE} IMPORTED)
endif (NOT TARGET SLU::dist)

# Set library location
set_property(TARGET SLU::dist
  PROPERTY IMPORTED_LOCATION ${SuperLUDist_LIBRARY})

# Set library include dirs
set_property(TARGET SLU::dist APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${SuperLUDist_INCLUDE_DIRS})

# Optional ParMETIS dependency
if (ParMETIS_FOUND)
  set_property(TARGET SLU::dist APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES ${ParMETIS_LIBRARIES})
endif (ParMETIS_FOUND)

# MPI dependency
set_property(TARGET SLU::dist APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_C_LIBRARIES})

#
# Manage the cache
#

# Set the include directories
set(SuperLUDist_INCLUDE_DIRS ${SuperLUDist_INCLUDE_DIRS}
  CACHE PATH
  "Directories in which to find headers for SuperLUDist.")
mark_as_advanced(FORCE SuperLUDist_INCLUDE_DIRS)

# Set the libraries
set(SuperLUDist_LIBRARIES SLU::dist)
mark_as_advanced(FORCE SuperLUDist_LIBRARY)

#
# Exit using standard function call
#

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SuperLUDist
  DEFAULT_MSG
  SuperLUDist_LIBRARY SuperLUDist_INCLUDE_DIRS)
