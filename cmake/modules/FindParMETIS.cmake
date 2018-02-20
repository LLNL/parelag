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
#   - ParMETIS_FOUND
#   - ParMETIS_INCLUDE_DIRS
#   - ParMETIS_LIBRARIES
#
# We need the following libraries:
#   parmetis

find_package(METIS QUIET REQUIRED)

# Find the header
find_path(ParMETIS_INCLUDE_DIRS parmetis.h
  HINTS ${ParMETIS_DIR} $ENV{ParMETIS_DIR} ${METIS_DIR} $ENV{METIS_DIR}
  PATH_SUFFIXES include
    NO_DEFAULT_PATH
  DOC "Directory where ParMETIS headers live.")
find_path(ParMETIS_INCLUDE_DIRS parmetis.h
  HINTS ${METIS_INCLUDE_DIRS}
  NO_DEFAULT_PATH)
find_path(ParMETIS_INCLUDE_DIRS parmetis.h)

# Find the library
find_library(ParMETIS_LIBRARY parmetis
  HINTS ${ParMETIS_DIR} $ENV{ParMETIS_DIR} ${METIS_DIR} $ENV{METIS_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The ParMETIS Library.")
find_library(ParMETIS_LIBRARY parmetis
  HINTS ${METIS_LIBRARY_DIRS}
  NO_DEFAULT_PATH)
find_library(ParMETIS_LIBRARY parmetis)

# Setup the imported target
if (NOT TARGET ParMETIS::parmetis)
  # Check if we have shared or static libraries
  include(ParELAGCMakeUtilities)
  parelag_determine_library_type(${ParMETIS_LIBRARY} ParMETIS_LIB_TYPE)

  add_library(ParMETIS::parmetis ${ParMETIS_LIB_TYPE} IMPORTED)
endif (NOT TARGET ParMETIS::parmetis)

# Add the library file
set_property(TARGET ParMETIS::parmetis
  PROPERTY IMPORTED_LOCATION ${ParMETIS_LIBRARY})

# Add the include path
set_property(TARGET ParMETIS::parmetis APPEND
  PROPERTY  INTERFACE_INCLUDE_DIRECTORIES ${ParMETIS_INCLUDE_DIRS})

# Add the MPI and METIS dependencies
set_property(TARGET ParMETIS::parmetis APPEND
  PROPERTY  INTERFACE_LINK_LIBRARIES
  ${MPI_C_LIBRARIES} ${METIS_LIBRARIES})

# Set the include directories
set(ParMETIS_INCLUDE_DIRS ${ParMETIS_INCLUDE_DIRS}
  CACHE PATH
  "Directories in which to find headers for ParMETIS.")
mark_as_advanced(FORCE ParMETIS_INCLUDE_DIRS)

# Set the libraries
set(ParMETIS_LIBRARIES ParMETIS::parmetis)
mark_as_advanced(FORCE ParMETIS_LIBRARY)


# This handles "REQUIRED" etc keywords
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ParMETIS
  DEFAULT_MSG
  ParMETIS_LIBRARY ParMETIS_INCLUDE_DIRS)
