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
#   - MFEM_FOUND
#   - MFEM_LIBRARIES
#   - MFEM_INCLUDE_DIRS
#   - MFEM_LIBRARY_DIRS

# FIXME: Perhaps I should try to assert that we're building against
# the same hypre library MFEM built on?

# We require parallel MFEM, which has explicit dependency on METIS and HYPRE
find_package(METIS REQUIRED)
find_package(HYPRE REQUIRED)

# Start by finding the MFEM config.mk file
find_file(MFEM_CONFIG_FILE config.mk
  HINTS ${MFEM_DIR} $ENV{MFEM_DIR}
  PATH_SUFFIXES config
  NO_DEFAULT_PATH
  DOC "The MFEM configuration file")
find_file(MFEM_CONFIG_FILE config.mk)

if (MFEM_CONFIG_FILE)
  # Extract the directory name
  get_filename_component(MFEM_CONFIG_PATH ${MFEM_CONFIG_FILE} DIRECTORY)

  # Extract the relevant list of lines
  file(STRINGS ${MFEM_CONFIG_FILE} MFEM_CONFIG_STRINGS REGEX " = ")
  
  # Now I need to parse this file; it's not pretty. Mayber there's a better way?
  foreach(MY_LINE IN LISTS MFEM_CONFIG_STRINGS)
    # Make the equals sign the list separator
    string(REPLACE "=" ";" MY_LINE_NO_EQ ${MY_LINE})
    
    # Set the CMAKE variable name to the first thing; set the value to
    # the second thing.
    list(GET MY_LINE_NO_EQ 0 MY_VAR_NAME)
    list(GET MY_LINE_NO_EQ 1 MY_VAR_VALUE)
    
    string(STRIP ${MY_VAR_NAME} MY_VAR_NAME)
    string(STRIP ${MY_VAR_VALUE} MY_VAR_VALUE)
    
    set(${MY_VAR_NAME} ${MY_VAR_VALUE})
  endforeach(MY_LINE)
  
  # Verify MFEM was built with MPI support
  if(NOT MFEM_USE_MPI)
    message(FATAL_ERROR "MFEM must be built with MPI support.")
  endif(NOT MFEM_USE_MPI)
  
else(MFEM_CONFIG_FILE)
  message(FATAL_ERROR "MFEM configuration file not found!")
endif(MFEM_CONFIG_FILE)
  
# Find the header
find_path(MFEM_INCLUDE_DIRS mfem.hpp
  HINTS ${MFEM_DIR} ${MFEM_CONFIG_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "MFEM header location.")
find_path(MFEM_INCLUDE_DIRS mfem.hpp)

# Find the library
find_library(MFEM_LIBRARY mfem
  HINTS ${MFEM_DIR} ${MFEM_CONFIG_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The MFEM library.")
find_library(MFEM_LIBRARY mfem
  DOC "The MFEM library.")


# Setup the imported target
if (NOT TARGET MFEM::mfem)
  # Check if we have shared or static libraries
  include(ParELAGCMakeUtilities)
  parelag_determine_library_type(${MFEM_LIBRARY} MFEM_LIB_TYPE)

  add_library(MFEM::mfem ${MFEM_LIB_TYPE} IMPORTED)
endif (NOT TARGET MFEM::mfem)

# Set the library location
set_property(TARGET MFEM::mfem
  PROPERTY IMPORTED_LOCATION ${MFEM_LIBRARY})

# Add the include path
set_property(TARGET MFEM::mfem APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${MFEM_INCLUDE_DIRS})

# Add the guaranteed dependencies -- HYPRE, METIS, MPI
set_property(TARGET MFEM::mfem APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES
  ${HYPRE_LIBRARIES} ${METIS_LIBRARIES} ${MPI_CXX_LIBRARIES})

# Add SuiteSparse, if used
if (MFEM_USE_SUITESPARSE)
  set_property(TARGET MFEM::mfem APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES ${SuiteSparse_LIBRARIES})
endif (MFEM_USE_SUITESPARSE)

# Add LAPACK, if used
if (MFEM_USE_LAPACK)
  set_property(TARGET MFEM::mfem APPEND 
    PROPERTY INTERFACE_LINK_LIBRARIES ${LAPACK_LIBRARIES})
endif (MFEM_USE_LAPACK)

# Set the include directories
set(MFEM_INCLUDE_DIRS ${MFEM_INCLUDE_DIRS}
  CACHE PATH
  "Directories in which to find headers for MFEM.")
mark_as_advanced(FORCE MFEM_INCLUDE_DIRS)

# Set the libraries
set(MFEM_LIBRARIES MFEM::mfem)
set(MFEM_LIBRARY ${MFEM_LIBRARY} CACHE PATH
  "The primary library file for MFEM")
mark_as_advanced(FORCE MFEM_LIBRARY)

mark_as_advanced(FORCE MFEM_CONFIG_FILE)


# This handles "REQUIRED" etc keywords
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MFEM
  DEFAULT_MSG
  MFEM_LIBRARY MFEM_CONFIG_FILE MFEM_USE_MPI MFEM_INCLUDE_DIRS)
