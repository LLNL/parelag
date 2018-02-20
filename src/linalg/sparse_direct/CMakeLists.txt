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

# SuperLU, SuperLU_DIST, and Strumpack require special treatment
if (${PROJECT_NAME}_ENABLE_SUPERLU)
  list(APPEND SPARSE_DIRECT_SRCS
    ParELAG_SuperLUSolver.cpp
    )

  list(APPEND SPARSE_DIRECT_HDRS
    ParELAG_SuperLU_Helpers.hpp
    ParELAG_SuperLUSolver.hpp
    ParELAG_SuperLUSolver_impl.hpp
    )
endif()

if (${PROJECT_NAME}_ENABLE_SUPERLUDIST)
  list(APPEND SPARSE_DIRECT_SRCS
    ParELAG_SuperLUDistSolver.cpp
    )

  list(APPEND SPARSE_DIRECT_HDRS
    ParELAG_SuperLUDist_Helpers.hpp
    ParELAG_SuperLUDistSolver.hpp
    ParELAG_SuperLUDistSolver_impl.hpp
    )
endif()

if (${PROJECT_NAME}_ENABLE_STRUMPACK)
  list(APPEND SPARSE_DIRECT_SRCS
    ParELAG_StrumpackSolver.cpp
    )

  list(APPEND SPARSE_DIRECT_HDRS
    ParELAG_StrumpackSolver.hpp
    ParELAG_StrumpackSolver_impl.hpp
    )
endif()

convert_filenames_to_full_paths(SPARSE_DIRECT_SRCS)
convert_filenames_to_full_paths(SPARSE_DIRECT_HDRS)

set(SRCS ${SRCS} ${SPARSE_DIRECT_SRCS} PARENT_SCOPE)
set(HDRS ${HDRS} ${SPARSE_DIRECT_HDRS} PARENT_SCOPE)
