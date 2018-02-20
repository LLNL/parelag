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
#   - METIS_FOUND
#   - METIS_LIBRARIES
#   - METIS_INCLUDE_DIRS
#

# Find the header
find_path(METIS_INCLUDE_DIRS metis.h
  HINTS ${METIS_DIR} $ENV{METIS_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with METIS header.")
find_path(METIS_INCLUDE_DIRS metis.h)

# Find the library
find_library(METIS_LIBRARY metis
  HINTS ${METIS_DIR} $ENV{METIS_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The METIS library.")
find_library(METIS_LIBRARY metis)

# Setup the imported target
if (NOT TARGET METIS::metis)
  # Check if we have shared or static libraries
  include(ParELAGCMakeUtilities)
  parelag_determine_library_type(${METIS_LIBRARY} METIS_LIB_TYPE)

  add_library(METIS::metis ${METIS_LIB_TYPE} IMPORTED)
endif (NOT TARGET METIS::metis)

# Set the location
set_property(TARGET METIS::metis
  PROPERTY IMPORTED_LOCATION ${METIS_LIBRARY})

# Set the include directories for the target
set_property(TARGET METIS::metis APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${METIS_INCLUDE_DIRS})

#
# Cleanup
#

# Set the include directories
set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIRS}
  CACHE PATH
  "Directories in which to find headers for METIS.")
mark_as_advanced(FORCE METIS_INCLUDE_DIRS)

# Set the libraries
set(METIS_LIBRARIES METIS::metis)
mark_as_advanced(FORCE METIS_LIBRARY)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS
  DEFAULT_MSG
  METIS_LIBRARY METIS_INCLUDE_DIRS)
