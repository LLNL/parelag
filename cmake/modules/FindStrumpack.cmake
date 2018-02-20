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

# Defines the following variables:
#   - STRUMPACK_FOUND
#   - STRUMPACK_LIBRARIES
#   - STRUMPACK_INCLUDE_DIRS

# FIXME: Technically, this depends on bunches of other libs: 
#   - ScaLAPACK
#   - BLACS
#   - METIS
#   - ParMETIS
#   - Scotch
find_package(METIS REQUIRED)
find_package(ParMETIS REQUIRED)
find_package(Scotch REQUIRED)
find_package(ScaLAPACK REQUIRED)

# Find the inlcude directory
find_path(STRUMPACK_INCLUDE_DIRS StrumpackSparseSolver.hpp
  HINTS ${STRUMPACK_DIR} $ENV{STRUMPACK_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with STRUMPACK header.")
find_path(STRUMPACK_INCLUDE_DIRS StrumpackSparseSolver.hpp)

find_library(STRUMPACK_LIBRARY strumpack_sparse
  HINTS ${STRUMPACK_DIR} $ENV{STRUMPACK_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The STRUMPACK library.")
find_library(STRUMPACK_LIBRARY strumpack_sparse)

#
# Add imported target
#

if (NOT TARGET STRUMPACK::strumpack)
  # Check if we have shared or static libraries
  include(ParELAGCMakeUtilities)
  parelag_determine_library_type(${STRUMPACK_LIBRARY} STRUMPACK_LIB_TYPE)

  add_library(STRUMPACK::strumpack ${STRUMPACK_LIB_TYPE} IMPORTED)
endif (NOT TARGET STRUMPACK::strumpack)

# Set the location
set_property(TARGET STRUMPACK::strumpack
  PROPERTY IMPORTED_LOCATION ${STRUMPACK_LIBRARY})

# Set the includes
set_property(TARGET STRUMPACK::strumpack APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${STRUMPACK_INCLUDE_DIRS})

# Manage dependencies
set_property(TARGET STRUMPACK::strumpack APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES
  ${SCOTCH_LIBRARIES} ${ParMETIS_LIBRARIES} ${METIS_LIBRARIES}
  ${ScaLAPACK_LIBRARIES} ${MPI_CXX_LIBRARIES} ${STRUMPACK_EXTRA_LINKAGE})

#
# Cleanup
#

# Set the include directories
set(STRUMPACK_INCLUDE_DIRS ${STRUMPACK_INCLUDE_DIRS}
  CACHE PATH
  "Directories in which to find headers for STRUMPACK.")
mark_as_advanced(FORCE STRUMPACK_INCLUDE_DIRS)

# Set the libraries
set(STRUMPACK_LIBRARIES STRUMPACK::strumpack)
mark_as_advanced(FORCE STRUMPACK_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(STRUMPACK
  DEFAULT_MSG
  STRUMPACK_LIBRARIES STRUMPACK_INCLUDE_DIRS)
