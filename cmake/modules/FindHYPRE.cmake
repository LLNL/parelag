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

# Finding HYPRE is as simple as finding its include/ and lib/
# directories... It's version number appears in
# include/HYPRE_config.h.
#
# Defines the following variables:
#   - HYPRE_FOUND
#   - HYPRE_LIBRARIES
#   - HYPRE_INCLUDE_DIRS

# Find the header files
find_path(HYPRE_INCLUDE_DIRS HYPRE.h
  HINTS ${HYPRE_DIR} $ENV{HYPRE_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with HYPRE header.")
find_path(HYPRE_INCLUDE_DIRS HYPRE.h)

#
# TODO: Do we want to find HYPRE_config.h as well and parse it to
# glean information about how HYPRE was built? E.g. With OpenMP? With
# it's own BLAS? FORTRAN symbol mangling?
#

# Find the library
find_library(HYPRE_LIBRARY HYPRE
  HINTS ${HYPRE_DIR} $ENV{HYPRE_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The HYPRE library.")
find_library(HYPRE_LIBRARY HYPRE)

# Setup the imported target
if (NOT TARGET HYPRE::hypre)
  # Check if we have shared or static libraries
  include(ParELAGCMakeUtilities)
  parelag_determine_library_type(${HYPRE_LIBRARY} HYPRE_LIB_TYPE)

  add_library(HYPRE::hypre ${HYPRE_LIB_TYPE} IMPORTED)
endif (NOT TARGET HYPRE::hypre)

# Set library
set_property(TARGET HYPRE::hypre
  PROPERTY IMPORTED_LOCATION ${HYPRE_LIBRARY})

# Add include path
set_property(TARGET HYPRE::hypre APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HYPRE_INCLUDE_DIRS})

# Add MPI and BLAS/LAPACK dependencies
set_property(TARGET HYPRE::hypre APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES
  ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# Set the include directories
set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIRS}
  CACHE PATH
  "Directories in which to find headers for HYPRE.")
mark_as_advanced(FORCE HYPRE_INCLUDE_DIRS)

# Set the libraries
set(HYPRE_LIBRARIES HYPRE::hypre)
mark_as_advanced(FORCE HYPRE_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HYPRE
  DEFAULT_MSG
  HYPRE_LIBRARY HYPRE_INCLUDE_DIRS)
