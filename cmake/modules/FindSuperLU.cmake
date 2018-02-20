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
#   - SuperLU_FOUND
#   - SuperLU_INCLUDE_DIRS
#   - SuperLU_LIBRARIES
#   - HAVE_SuperLU_VERSION_5

# We'll require the "slu_*defs.h" first, since they are unique to
# sequential SuperLU.

find_package(BLAS QUIET REQUIRED)

# Look for slu_ddefs.h
find_path(SuperLU_INCLUDE_DIRS slu_ddefs.h
  HINTS ${SuperLU_DIR} $ENV{SuperLU_DIR}
  PATH_SUFFIXES include SRC SuperLU
  DOC "Directory where sequential SuperLU *def headers live."
  NO_DEFAULT_PATH)
find_path(SuperLU_INCLUDE_DIRS slu_ddefs.h)

# Now try to find the library
find_library(SuperLU_LIBRARY
  NAMES superlu superlu_4.0 superlu_4.1 superlu_4.2 superlu_4.3 superlu_5.0 superlu_5.1 superlu_5.1.1 NAMES_PER_DIR
  HINTS ${SuperLU_DIR} $ENV{SuperLU_DIR} ${SuperLU_LIBRARY_DIRS}
  PATH_SUFFIXES lib SRC
  DOC "Sequential SuperLU library."
  NO_DEFAULT_PATH)
find_library(SuperLU_LIBRARY
  NAMES superlu superlu_4.0 superlu_4.1 superlu_4.2 superlu_4.3 superlu_5.0 superlu_5.1 superlu_5.1.1 NAMES_PER_DIR)

# There are several interface changes that happen across the different
# versions of SuperLU. I need some handle on my version number. Since
# SuperLU doesn't make that easy, let's hack it out.
include(CheckCSourceCompiles)
function(check_superlu_version_5  VARNAME)
  set(SOURCE
    "
typedef int int_t;
#include <supermatrix.h>
#include <slu_ddefs.h>

int main()
{
  superlu_options_t opts;
  SuperMatrix A;
  int my_int;
  GlobalLU_t glu;
  SuperLUStat_t stat;

  dgstrf(&opts,&A,my_int,my_int,&my_int,&my_int,my_int,
         &my_int,&my_int,&A,&A,&glu,&stat,&my_int);
  return 0;
}
"
    )
  set(CMAKE_REQUIRED_FLAGS "-c")
  set(CMAKE_REQUIRED_INCLUDES ${SuperLU_INCLUDE_DIRS})
  check_c_source_compiles("${SOURCE}" ${VARNAME})
ENDFUNCTION()

check_superlu_version_5(${PROJECT_NAME}_SuperLU_HAVE_VERSION_5)

# Add imported target
if (NOT TARGET SLU::sequential)
  # Check if we have shared or static libraries
  include(ParELAGCMakeUtilities)
  parelag_determine_library_type(${SuperLU_LIBRARY} SuperLU_LIB_TYPE)

  add_library(SLU::sequential ${SuperLU_LIB_TYPE} IMPORTED)
endif (NOT TARGET SLU::sequential)

# Set library location
set_property(TARGET SLU::sequential
  PROPERTY IMPORTED_LOCATION ${SuperLU_LIBRARY})

# Set include directories
set_property(TARGET SLU::sequential APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${SuperLU_INCLUDE_DIRS})

# BLAS dependency
set_property(TARGET SLU::sequential APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES})

#
# Manage the cache
#

# Set the include directories
set(SuperLU_INCLUDE_DIRS ${SuperLU_INCLUDE_DIRS}
  CACHE PATH
  "Directories in which to find headers for SuperLU.")
mark_as_advanced(FORCE SuperLU_INCLUDE_DIRS)

# Set the libraries
set(SuperLU_LIBRARIES SLU::sequential)
mark_as_advanced(FORCE SuperLU_LIBRARY)

#
# Exit using standard function call
#

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SuperLU
  DEFAULT_MSG
  SuperLU_LIBRARY SuperLU_INCLUDE_DIRS)
