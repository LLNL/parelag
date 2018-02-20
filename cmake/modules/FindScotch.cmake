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
#   - Scotch_FOUND
#   - Scotch_INCLUDE_DIRS
#   - Scotch_LIBRARIES

find_path(SCOTCH_INCLUDE_DIRS scotch.h
  HINTS ${SCOTCH_DIR} $ENV{SCOTCH_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with SCOTCH header.")
find_path(SCOTCH_INCLUDE_DIRS scotch.h
  DOC "Directory with SCOTCH header.")

set(REQUIRED_SCOTCH_LIBRARIES scotch ptscotch scotcherr ptscotcherr)
foreach(lib ${REQUIRED_SCOTCH_LIBRARIES})

  string(TOUPPER ${lib} lib_uc)
  
  # Find the library
  find_library(${lib_uc}_LIBRARY ${lib}
    HINTS ${SCOTCH_DIR} $ENV{SCOTCH_DIR}
    ${SCOTCH_LIBRARY_DIRS} $ENV{SCOTCH_LIBRARY_DIRS}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
    DOC "The SCOTCH library.")
  find_library(${lib_uc}_LIBRARY ${lib}
    DOC "The ${lib_uc} library.")

  #
  # Imported targets
  #
  if (${lib_uc}_LIBRARY)

    if (NOT TARGET SCOTCH::${lib})
      # Check if we have shared or static libraries
      include(ParELAGCMakeUtilities)
      parelag_determine_library_type(${${lib_uc}_LIBRARY} ${lib_uc}_LIB_TYPE)

      add_library(SCOTCH::${lib} ${${lib_uc}_LIB_TYPE} IMPORTED)
    endif (NOT TARGET SCOTCH::${lib})

    # Set library location
    set_property(TARGET SCOTCH::${lib}
      PROPERTY IMPORTED_LOCATION ${${lib_uc}_LIBRARY})

    # Set include directories
    set_property(TARGET SCOTCH::${lib} APPEND
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${SCOTCH_INCLUDE_DIRS})

    # Determine and set MPI dependency
    string(FIND ${lib} "pt" _is_parallel_lib)
    if (${_is_parallel_lib} GREATER -1)
      set_property(TARGET SCOTCH::${lib} APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_C_LIBRARIES})
    endif (${_is_parallel_lib} GREATER -1)

    list(APPEND SCOTCH_LIBRARIES "SCOTCH::${lib}")

    # Cleanup
    mark_as_advanced(FORCE ${lib_uc}_LIBRARY)

  endif (${lib_uc}_LIBRARY)

endforeach(lib ${REQUIRED_SCOTCH_LIBRARIES})

set(SCOTCH_INCLUDE_DIRS ${SCOTCH_INCLUDE_DIRS}
  CACHE PATH
  "Directories in which to find headers for SCOTCH.")
mark_as_advanced(FORCE SCOTCH_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SCOTCH
  DEFAULT_MSG SCOTCH_INCLUDE_DIRS SCOTCH_LIBRARIES SCOTCH_LIBRARY_DIRS)
