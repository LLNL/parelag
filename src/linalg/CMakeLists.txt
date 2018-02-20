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

list(APPEND SRCS
  )

list(APPEND HDRS
  elag_linalg.hpp
  )

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

add_subdirectory(dense)
add_subdirectory(factories)
add_subdirectory(legacy)
add_subdirectory(solver_core)
add_subdirectory(solver_ops)
add_subdirectory(utilities)

if (${PROJECT_NAME}_ENABLE_SUPERLU OR ${PROJECT_NAME}_ENABLE_SUPERLUDIST OR
    ${PROJECT_NAME}_ENABLE_STRUMPACK)
  add_subdirectory(sparse_direct)
endif()

if (${PROJECT_NAME}_ENABLE_TESTING)
  add_subdirectory(unit_test)
endif (${PROJECT_NAME}_ENABLE_TESTING)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
