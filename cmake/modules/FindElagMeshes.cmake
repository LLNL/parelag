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

# Sets the following variables:
#   - ElagMeshes_FOUND
#   - ElagMeshes_DIR
#
# This refers to meshes that we use for some Parelag tests,
# they live in a repo at
#
#   git clone https://barker29@lc.llnl.gov/stash/scm/elag/meshes.git
#
# There are more meshes that ship with MFEM.

find_path(ElagMeshes_DIR cyl_in_cube.mesh3d
  HINTS ${MESH_DIR} $ENV{HOME}/meshes $ENV{HOME}/Meshes
  ${PROJECT_SOURCE_DIR}/../meshes)

# Set the include directories
set(ElagMeshes_DIR ${ElagMeshes_DIR}
  CACHE PATH
  "Directories in which to find extra meshes for testing.")
mark_as_advanced(FORCE ElagMeshes_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ElagMeshes
  DEFAULT_MSG
  ElagMeshes_DIR)
