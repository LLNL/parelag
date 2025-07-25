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

# Defines the following variables
#   - MATRED_FOUND
#   - MATRED_LIBRARIES
#   - MATRED_INCLUDE_DIRS
#   - MATRED_LIBRARY_DIRS

# FIXME: Perhaps I should try to assert that we're building against
# the same hypre library MATRED built on?
# Find the header files
find_path(MATRED_INCLUDE_DIRS matred.hpp
  HINTS ${MATRED_DIR} $ENV{MATRED_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with MATRED header.")
find_path(MATRED_INCLUDE_DIRS matred.hpp)

# Find the library
find_library(MATRED_LIBRARY matred
  HINTS ${MATRED_DIR} $ENV{MATRED_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The MATRED library.")
find_library(MATRED_LIBRARY matred)

# Setup the imported target
if (NOT TARGET MATRED::matred)
  # Check if we have shared or static libraries
  include(ParELAGCMakeUtilities)
  parelag_determine_library_type(${MATRED_LIBRARY} MATRED_LIB_TYPE)

  add_library(MATRED::matred ${MATRED_LIB_TYPE} IMPORTED)
endif (NOT TARGET MATRED::matred)

# Set library
set_property(TARGET MATRED::matred
  PROPERTY IMPORTED_LOCATION ${MATRED_LIBRARY})

# Add include path
set_property(TARGET MATRED::matred APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${MATRED_INCLUDE_DIRS})

# Add MPI and BLAS/LAPACK dependencies
set_property(TARGET MATRED::matred APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES
  ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# Set the include directories
set(MATRED_INCLUDE_DIRS ${MATRED_INCLUDE_DIRS}
  CACHE PATH
  "Directories in which to find headers for MATRED.")
mark_as_advanced(FORCE MATRED_INCLUDE_DIRS)

# Set the libraries
set(MATRED_LIBRARIES MATRED::matred)
mark_as_advanced(FORCE MATRED_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MATRED
  DEFAULT_MSG
  MATRED_LIBRARY MATRED_INCLUDE_DIRS)
