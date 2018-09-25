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

# Build the CMake configuration files
include(CMakePackageConfigHelpers)

set(MODULES_TO_INSTALL
  modules/CorrectWindowsPaths.cmake
  modules/FindElagMeshes.cmake
  modules/FindHYPRE.cmake
  modules/FindMETIS.cmake
  modules/FindMFEM.cmake
  modules/FindPackageMultipass.cmake
  modules/FindParMETIS.cmake
  modules/FindPETSc.cmake
  modules/FindScaLAPACK.cmake
  modules/FindScotch.cmake
  modules/FindStrumpack.cmake
  modules/FindSuiteSparse.cmake
  modules/FindSuperLU.cmake
  modules/FindSuperLUDist.cmake
  modules/FindMARS.cmake
  modules/ParELAGCMakeUtilities.cmake
  modules/ParELAGTestForPrettyFunction.cmake
  modules/ResolveCompilerPaths.cmake
  )

# Create the version for the build tree
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}"
  COMPATIBILITY SameMajorVersion)

# NOTE: The build directory here is not relocatable with respect to
# the build directory SquirrelConfig.cmake file!

# Create the build-tree configuration file
set(INCLUDE_INSTALL_DIRS "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/src")
set(LIB_INSTALL_DIR "${PROJECT_BINARY_DIR}/src")
set(MODULE_INSTALL_DIR "${PROJECT_SOURCE_DIR}/cmake/modules")

configure_package_config_file(ParELAGConfig.cmake.in
  ${CMAKE_BINARY_DIR}/ParELAGConfig.cmake
  INSTALL_DESTINATION ${CMAKE_BINARY_DIR}
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR MODULE_INSTALL_DIR)

#
# Install tree stuff
#

# Create the install-tree configuration file
set(INCLUDE_INSTALL_DIRS include/parelag)
set(LIB_INSTALL_DIR lib)
set(CMAKE_INSTALL_DIR lib/cmake/parelag)
set(MODULE_INSTALL_DIR ${CMAKE_INSTALL_DIR}/modules)

configure_package_config_file(
  ParELAGConfig.cmake.in
  ParELAGConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_DIR}
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR MODULE_INSTALL_DIR)

# Install the version file
install(
  FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_DIR})

# Install the modules
install(FILES ${MODULES_TO_INSTALL} DESTINATION ${MODULE_INSTALL_DIR})
