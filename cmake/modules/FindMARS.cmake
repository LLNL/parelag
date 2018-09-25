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
#   - MARS_FOUND
#   - MARS_LIBRARIES
#   - MARS_INCLUDE_DIRS
#

# Find the header
find_path(MARS_INCLUDE_DIRS mars.hpp
  HINTS ${MARS_DIR} $ENV{MARS_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with MARS header.")
find_path(MARS_INCLUDE_DIRS mars.hpp)

# Find the library
find_library(MARS_LIBRARY mars
  HINTS ${MARS_DIR} $ENV{MARS_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The MARS library.")
find_library(MARS_LIBRARY mars)

# Setup the imported target
if (NOT TARGET MARS::mars)
  # Check if we have shared or static libraries
  include(ParELAGCMakeUtilities)
  parelag_determine_library_type(${MARS_LIBRARY} MARS_LIB_TYPE)

  add_library(MARS::mars ${MARS_LIB_TYPE} IMPORTED)
endif (NOT TARGET MARS::mars)

# Set the location
set_property(TARGET MARS::mars
  PROPERTY IMPORTED_LOCATION ${MARS_LIBRARY})

# Set the include directories for the target
set_property(TARGET MARS::mars APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${MARS_INCLUDE_DIRS})

#
# Cleanup
#

# Set the include directories
set(MARS_INCLUDE_DIRS ${MARS_INCLUDE_DIRS}
  CACHE PATH
  "Directories in which to find headers for MARS.")
mark_as_advanced(FORCE MARS_INCLUDE_DIRS)

# Set the libraries
set(MARS_LIBRARIES MARS::mars)
mark_as_advanced(FORCE MARS_LIBRARY)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MARS
  DEFAULT_MSG
  MARS_LIBRARY MARS_INCLUDE_DIRS)
