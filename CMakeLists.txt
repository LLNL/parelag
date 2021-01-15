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

# TODO: Cleanup the cache
# TODO: Cleanup "make help"
message("")
message("Welcome to ParElag! Best of luck... You'll need it.")
message("")

# Some features here won't work if version < 3.1. If you find a
# machine with cmake < 3.1, do let me know.
#
# Internal: The LLNL RHEL repo is way behind. Please manually install
# CMake version >= 3.1 and use that instead.
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# Project name
set(PROJECT_NAME ParELAG)
project(${PROJECT_NAME})

# RPATH on MACOSX
if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()

# Set version
set(${PROJECT_NAME}_VERSION_MAJOR 0)
set(${PROJECT_NAME}_VERSION_MINOR 1)

# Just in case we need it
string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_UC)

#
# Important paths/variables
#

set(HOME_DIR $ENV{HOME})
set(${PROJECT_NAME}_CMAKE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${${PROJECT_NAME}_CMAKE_PATH}/modules)

# Enforce C11
if ((NOT CMAKE_C_STANDARD) OR (CMAKE_C_STANDARD LESS 11))
  set(CMAKE_C_STANDARD 11)
endif()
set(CMAKE_C_STANDARD_REQUIRED ON)

message(STATUS "Using C standard: c${CMAKE_C_STANDARD}")

# Enforce C++11
if ((NOT CMAKE_CXX_STANDARD) OR (CMAKE_CXX_STANDARD LESS 11))
  set(CMAKE_CXX_STANDARD 11)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Using CXX standard: c++${CMAKE_CXX_STANDARD}")

include(ParELAGTestForPrettyFunction)
check_for_pretty_function(${PROJECT_NAME}_HAVE_PRETTY_FUNCTION)

# This is for backwards compatibility with the former build
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DELAG_DEBUG -O0")

#
# Options that control what is built and how
#
option(${PROJECT_NAME}_MG_ENABLE_UNITTESTS
  "Enable Tom's unit tests for MG"
  ON)

#
# Development flags
#
# Currently just -Wall, because clean code is warning-free code
option(${PROJECT_NAME}_DEVELOPER_BUILD
  "Enable extra compiler flags for development."
  OFF)
if (${PROJECT_NAME}_DEVELOPER_BUILD)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR
      ${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang" OR
      ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
  endif ()
endif()

# Whether to build the examples
option(${PROJECT_NAME}_BUILD_EXAMPLES
  "Build the examples/ directory"
  OFF)

# Whether to use MARS
option(${PROJECT_NAME}_USE_MARS
  "Build the MARS example"
  OFF)

# Whether to build the testsuite
option(${PROJECT_NAME}_BUILD_TESTSUITE
  "Build the testsuite/ directory"
  OFF)

# Whether to enable the testing framework
option(${PROJECT_NAME}_ENABLE_TESTING
  "Enable the Ctest framework"
  ON)
option(${PROJECT_NAME}_ENABLE_VALGRIND_TESTS
  "Enable the tests that use valgrind (very time-consuming)"
  OFF)
option(${PROJECT_NAME}_ENABLE_FAILING_TOPOLOGY_TESTS
  "Enable tests revealing bug in topology/partitioning correction."
  OFF)
if (${PROJECT_NAME}_ENABLE_TESTING)
  # Need the examples/testsuite if tests are enabled
  set(${PROJECT_NAME}_BUILD_EXAMPLES ON)
  set(${PROJECT_NAME}_BUILD_TESTSUITE ON)
endif()

#
# Options for third-party libraries
#
option(${PROJECT_NAME}_ENABLE_OPENMP
  "Enable OpenMP support for the library"
  OFF)

option(${PROJECT_NAME}_ENABLE_PARMETIS
  "Should ParMETIS content be enabled?"
  OFF)

option(${PROJECT_NAME}_ENABLE_SUPERLU
  "Enable experimental code for the library"
  OFF)

option(${PROJECT_NAME}_ENABLE_SUPERLUDIST
  "Enable experimental code for the library"
  OFF)

option(${PROJECT_NAME}_ENABLE_STRUMPACK
  "Enable support for the strumpack library"
  OFF)

option(${PROJECT_NAME}_ENABLE_PETSC
  "Should Petsc content be enabled?"
  OFF)

# On massively distributed systems on which it may be best/required to
# use static executables, this option should be enabled to "force"
# compilers to link everything as statically as possible.
option(${PROJECT_NAME}_FORCE_STATIC_LINKAGE
  "Whether to (attempt to) build static executables for examples and tests"
  OFF)

if (${PROJECT_NAME}_FORCE_STATIC_LINKAGE)
  # Find .a before .so. This matters A LOT for Vulcan
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a .so)

  # Disables trying to build any shared library
  set(BUILD_SHARED_LIBS OFF)

  # At least strongly suggest to a compiler to use static linkage by
  # default. This works, for example, for getting GNU to link libc
  # statically.
  set(CMAKE_LINK_SEARCH_START_STATIC ON)
endif (${PROJECT_NAME}_FORCE_STATIC_LINKAGE)

#
# Initial configuration of linked libraries and include directories
#

# This is a list of TPLs that are used by all targets
set(TPL_LIBRARIES "")

# This is a list of linker flags to be used with TPLs for all targets
set(TPL_LINKER_FLAGS "")

# Currently ParELAG is setup to include things rooted off of
# src/. I.e., includes of are of the form
#
#    #include "linalg/MultiVector.hpp"
#
include_directories("${PROJECT_SOURCE_DIR}/src")

# Configure file gets written to the binary directory
include_directories("${PROJECT_BINARY_DIR}/src")

#
# THIRD-PARTY LIBRARIES (TPLs)
#
# Required third-party libraries:
#   0. MPI
#   1. BLAS/LAPACK
#   2. Hypre
#   3. MFEM
#      3.1. METIS
#      3.2. SUITESPARSE
#
# Optional third-party libraries:
#   0. ParMETIS
#   1. Petsc
# In Tom's MG branch:
#   1. SuperLU
#   2. SuperLU_DIST
#   3. STRUMPack
#      3.1. ScaLAPACK
#      3.2. Scotch

#
# REQUIRED LIBRARIES
#

# We need to have MPI enabled for ParElag
find_package(MPI REQUIRED)
if (MPI_CXX_FOUND)
  if (NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX INTERFACE IMPORTED)

    if (MPI_CXX_COMPILE_FLAGS)
      separate_arguments(_MPI_CXX_COMPILE_OPTIONS UNIX_COMMAND
        "${MPI_CXX_COMPILE_FLAGS}")
      set_property(TARGET MPI::MPI_CXX PROPERTY
        INTERFACE_COMPILE_OPTIONS "${_MPI_CXX_COMPILE_OPTIONS}")
    endif()

    if(MPI_CXX_LINK_FLAGS)
      separate_arguments(_MPI_CXX_LINK_LINE UNIX_COMMAND
        "${MPI_CXX_LINK_FLAGS}")
    endif()
    list(APPEND _MPI_CXX_LINK_LINE "${MPI_CXX_LIBRARIES}")

    set_property(TARGET MPI::MPI_CXX PROPERTY
      INTERFACE_LINK_LIBRARIES "${_MPI_CXX_LINK_LINE}")

    set_property(TARGET MPI::MPI_CXX PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")

  endif (NOT TARGET MPI::MPI_CXX)
else (MPI_CXX_FOUND)

  message(FATAL_ERROR "MPI not found for CXX language.")
  
endif (MPI_CXX_FOUND)

# I don't know why these don't get hidden by default, but we can
# safely hide them.
mark_as_advanced(FORCE MPI_LIBRARY)
mark_as_advanced(FORCE MPI_EXTRA_LIBRARY)
# The FindMPI module now exports MPI_<LANG>_WHATEVER variables, so
# this is all target-specific. Thus, we don't add the MPI libraries to
# the TPL_LIBRARIES list or the MPI include directories to the global
# include directories.

# Get system BLAS and LAPACK. These can, of course, be overriden, but
# the libraries are required to build ParElag.
find_package(BLAS REQUIRED)

find_package(LAPACK REQUIRED)

# We need to find MFEM (root, lib, and inc dirs) and assert that it
# was built with MPI support.
#
# NOTE: ParElag relies ONLY on the fact that MFEM must be built with
# MPI support. All other choices are viewed to be independent
# options. That is, whether or not MFEM was built with OpenMP does not
# affect whether ParElag will be built with OpenMP support,
# etc. This should be fixed. :/
find_package(MFEM REQUIRED)
# Check if any executables need to be linked, while the MFEM library was not
# discovered.
if ((${PROJECT_NAME}_BUILD_EXAMPLES OR ${PROJECT_NAME}_BUILD_TESTSUITE) AND (NOT MFEM_LIBRARY))
message(FATAL_ERROR "Failed to find the MFEM library file, while Examples and Testsuite linking is ON!")
endif ()

# Look for METIS; this should be the same metis used to build MFEM,
# but if it links, it links.
find_package(METIS REQUIRED)
include_directories(${METIS_INCLUDE_DIRS})

# We require a small collection of the SuiteSparse library. We need
# the following libraries: umfpack, amd, colamd, cholmod, spqr
# suitesparseconfig, klu, and btf.
#
# If this collection of libraries cannot be found, SuiteSparse will be
# incomplete and a fatal error will have occured
find_package(SuiteSparse REQUIRED)
include_directories(${SuiteSparse_INCLUDE_DIRS})

#
# OPTIONAL LIBRARIES
#

# Look for MARS support.
if(${PROJECT_NAME}_USE_MARS)
  message(STATUS "Searching for MARS...")
  find_package(MARS)
  if (MARS_FOUND)
    include_directories(${MARS_INCLUDE_DIRS})
  else()
    set(${PROJECT_NAME}_USE_MARS OFF)
    message(STATUS "** MARS NOT FOUND; CONTENT DISABLED **")
  endif()
endif()

# Look for OpenMP support. This is completely independent of whether
# Hypre/MFEM are built with OpenMP support.
if(${PROJECT_NAME}_ENABLE_OPENMP)
  message(STATUS "Searching for OpenMP...")
  find_package(OpenMP)
  if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
    set(${PROJECT_NAME}_ENABLE_OPENMP OFF)
    message(STATUS "** OpenMP NOT FOUND; CONTENT DISABLED **")
  endif()
endif()

# Look for ParMETIS support.
if (${PROJECT_NAME}_ENABLE_PARMETIS)
  message(STATUS "Searching for ParMETIS...")
  find_package(ParMETIS)
  if (PARMETIS_FOUND)
    include_directories(${ParMETIS_INCLUDE_DIRS})
  else()
    set(${PROJECT_NAME}_ENABLE_PARMETIS OFF)
    message(STATUS "** PARMETIS NOT FOUND; CONTENT DISABLED **")
  endif()
endif()

# FIXME: Should ParMETIS really be added to the global include dirs
#        and/or TPL_LIBRARIES list?

# Check for SuperLU (sequential)
if (${PROJECT_NAME}_ENABLE_SUPERLU)
  find_package(SuperLU)
  if (SUPERLU_FOUND)
    include_directories(${SuperLU_INCLUDE_DIRS})
  else()
    set(${PROJECT_NAME}_ENABLE_SUPERLU OFF)
    message(STATUS "** SuperLU NOT FOUND; SuperLU will be disabled **")
  endif()
endif()

    
# PETSc cannot be used with MFEM_USE_PETSC=NO
if (${PROJECT_NAME}_ENABLE_PETSC)
  if ("${MFEM_USE_PETSC}" STREQUAL "YES")
    message(STATUS "Enabling PETSc support")
    find_package(PETSc)
    include_directories(${PETSC_INCLUDES})
  else()
    message(STATUS "Cannot enable PETSc support. Reconfigure MFEM with MFEM_USE_PETSC=YES")
  endif()
endif()

if (${PROJECT_NAME}_ENABLE_SUPERLUDIST)
  find_package(SuperLUDist)
  if (SUPERLUDIST_FOUND)
    include_directories(${SuperLUDist_INCLUDE_DIRS})
  else()
    set(${PROJECT_NAME}_ENABLE_SUPERLUDIST OFF)
    message(STATUS "** SuperLUDist NOT FOUND; SuperLUDist will be disabled **")
  endif()
endif()


if (${PROJECT_NAME}_ENABLE_STRUMPACK)
  find_package(Strumpack)
  if (STRUMPACK_FOUND)
    include_directories(${SCALAPACK_INCLUDE_DIRS})
    include_directories(${SCOTCH_INCLUDE_DIRS})
    include_directories(${STRUMPACK_INCLUDE_DIRS})
  else()
    set(${PROJECT_NAME}_ENABLE_STRUMPACK OFF)
    message(STATUS "** Strumpack NOT FOUND; Strumpack will be disabled **")
  endif()
endif()

#
# Build the TPL_LIBRARIES LIST
#
list(APPEND TPL_LIBRARIES ${MFEM_LIBRARIES})
list(APPEND TPL_LIBRARIES ${PETSC_LIBRARIES})
list(APPEND TPL_LIBRARIES ${SuiteSparse_LIBRARIES})
list(APPEND TPL_LIBRARIES ${SuperLU_LIBRARIES})
list(APPEND TPL_LIBRARIES ${SuperLUDist_LIBRARIES})
list(APPEND TPL_LIBRARIES ${STRUMPACK_LIBRARIES})
list(APPEND TPL_LIBRARIES ${SCALAPACK_LIBRARIES})
list(APPEND TPL_LIBRARIES ${SCOTCH_LIBRARIES})
list(APPEND TPL_LIBRARIES ${ParMETIS_LIBRARIES})
list(APPEND TPL_LIBRARIES ${METIS_LIBRARIES})
list(APPEND TPL_LIBRARIES ${LAPACK_LIBRARIES})
list(APPEND TPL_LIBRARIES ${BLAS_LIBRARIES})

list(APPEND TPL_LINKER_FLAGS ${LAPACK_LINKER_FLAGS})
list(APPEND TPL_LINKER_FLAGS ${BLAS_LINKER_FLAGS})

# Cleanup the TPL list
list(REMOVE_DUPLICATES TPL_LIBRARIES)

#
# Add the subdirectories
#
include(ParELAGCMakeUtilities)
if (${PROJECT_NAME}_ENABLE_TESTING OR ${PROJECT_NAME}_MG_ENABLE_UNITTESTS)
  enable_testing()

  # look for meshes, if we can find them we run different tests
  find_package(ElagMeshes)

  # Logic to enable valgrind tests
  if(${PROJECT_NAME}_ENABLE_VALGRIND_TESTS)
    find_program(MEMORYCHECK_COMMAND valgrind)
    set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full")
    function(add_valgrind_test name exe_target)
      add_test(NAME ${name}
        COMMAND ${MEMORYCHECK_COMMAND} ${MEMORYCHECK_COMMAND_OPTIONS}
        $<TARGET_FILE:${exe_target}> ${ARGN})
      set_tests_properties(${name}
        PROPERTIES PASS_REGULAR_EXPRESSION "ERROR SUMMARY: 0 errors")
    endfunction(add_valgrind_test)
  endif()

  function(add_mpi_test name procs exe_target)
    add_test(
      NAME ${name}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${procs} ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:${exe_target}> ${MPIEXEC_POSTFLAGS} ${ARGN})    
  endfunction(add_mpi_test)
  
  # style test - no tabs
  add_test(notabs
    grep -rP --include={*.cpp,*.hpp,*.c,*.h} '\t' ${PROJECT_SOURCE_DIR})
  set_tests_properties(notabs PROPERTIES WILL_FAIL "TRUE")

endif()

# The primary source directory for the library.
add_subdirectory(src)

# Build the examples directory, if requested
if (${PROJECT_NAME}_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Build the regression testsuite, if requested
if (${PROJECT_NAME}_BUILD_TESTSUITE)
  add_subdirectory(testsuite)
endif()

#
# Documentation target
#
add_subdirectory(doc)

#
# CONFIGURATION FILE
#
configure_file(
  "${PROJECT_SOURCE_DIR}/src/${PROJECT_NAME}_Config.h.in"
  "${PROJECT_BINARY_DIR}/src/${PROJECT_NAME}_Config.h"
  )

#
# INSTALL TARGET
#

# Library install command in src/CMakeLists.txt
# All the export stuff is either in src/CMakeLists.txt or cmake/CMakeLists.txt

# Build directory export
export(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE parelag::
  FILE ${PROJECT_NAME}Targets.cmake)

# Add the build tree to package registry
export(PACKAGE ${PROJECT_NAME})

# Handle the rest of the export stuff
add_subdirectory(cmake)

# Install the headers
install(DIRECTORY src/ DESTINATION include/parelag
  FILES_MATCHING PATTERN "*.hpp")
install(FILES "${PROJECT_BINARY_DIR}/src/${PROJECT_NAME}_Config.h"
  DESTINATION include/parelag)
