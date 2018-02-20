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
#   - ScaLAPACK_FOUND
#   - ScaLAPACK_LIBRARIES
#
# WARNING: I am not a regular user of this library; all I know is that
# this gets things to work for me

find_package(BLAS QUIET REQUIRED)

# BLACS
find_library(BLACS_LIBRARY
  NAMES mpiblacs blacs blacs-pvm blacs-mpi blacs-mpich blacs-mpich2 blacs-openmpi blacs-lam NAMES_PER_DIR
  HINTS ${ScaLAPACK_DIR}/lib $ENV{ScaLAPACK_DIR}/lib
  NO_DEFAULT_PATH
  DOC "The BLACS library.")
find_library(BLACS_LIBRARY
  NAMES mpiblacs blacs blacs-pvm blacs-mpi blacs-mpich blacs-mpich2 blacs-openmpi blacs-lam NAMES_PER_DIR
  DOC "The BLACS library.")

# ScaLAPACK
find_library(ScaLAPACK_LIBRARY
  NAMES scalapack scalapack-pvm scalapack-mpi scalapack-mpich scalapack-mpich2 scalapack-openmpi scalapack-lam NAMES_PER_DIR
  HINTS ${ScaLAPACK_DIR}/lib $ENV{ScaLAPACK_DIR}/lib
  NO_DEFAULT_PATH
  DOC "The ScaLAPACK library.")
find_library(ScaLAPACK_LIBRARY
  NAMES scalapack scalapack-pvm scalapack-mpi scalapack-mpich scalapack-mpich2 scalapack-openmpi scalapack-lam NAMES_PER_DIR
  DOC "The ScaLAPACK library.")

# Imported targets

#
# Add BLACS imported target
#
if (BLACS_LIBRARY)
  if (NOT TARGET ScaLAPACK::blacs)
    # Check if we have shared or static libraries
    include(ParELAGCMakeUtilities)
    parelag_determine_library_type(${BLACS_LIBRARY} BLACS_LIB_TYPE)
    
    add_library(ScaLAPACK::blacs ${BLACS_LIB_TYPE} IMPORTED)
  endif (NOT TARGET ScaLAPACK::blacs)
  
  # Set library location
  set_property(TARGET ScaLAPACK::blacs
    PROPERTY IMPORTED_LOCATION ${BLACS_LIBRARY})
  
  # MPI dependency
  set_property(TARGET ScaLAPACK::blacs APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_C_LIBRARIES})
else()
  message(STATUS "WARNING: No separate BLACS library found!")
endif (BLACS_LIBRARY)

#
# Add ScaLAPACK imported target
#
if (NOT TARGET ScaLAPACK::scalapack)
  # Check if we have shared or static libraries
  include(ParELAGCMakeUtilities)
  parelag_determine_library_type(${ScaLAPACK_LIBRARY} ScaLAPACK_LIB_TYPE)
  
  add_library(ScaLAPACK::scalapack ${ScaLAPACK_LIB_TYPE} IMPORTED)
endif (NOT TARGET ScaLAPACK::scalapack)

# Set library location
set_property(TARGET ScaLAPACK::scalapack
  PROPERTY IMPORTED_LOCATION ${ScaLAPACK_LIBRARY})

# BLACS and MPI dependencies
if (TARGET ScaLAPACK::blacs)
  set_property(TARGET ScaLAPACK::scalapack APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES
    ScaLAPACK::blacs ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES})
else()
  set_property(TARGET ScaLAPACK::scalapack APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES
    ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES})
endif (TARGET ScaLAPACK::blacs)

# Set the libraries
mark_as_advanced(FORCE BLACS_LIBRARY)
mark_as_advanced(FORCE ScaLAPACK_LIBRARY)

set(ScaLAPACK_LIBRARIES ScaLAPACK::scalapack)

# Report the found libraries, quit with fatal error if any required library has not been found.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ScaLAPACK DEFAULT_MSG ScaLAPACK_LIBRARY)
