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

list(APPEND SOLVER_OPS_SRCS
  ParELAG_ADSSolverWrapper.cpp
  ParELAG_AMSSolverWrapper.cpp
  ParELAG_Block2x2LDUInverseOperator.cpp
  ParELAG_BlockDiagonalSolver.cpp
  ParELAG_BlockTriangularSolver.cpp
  ParELAG_BoomerAMGSolverWrapper.cpp
  ParELAG_BramblePasciakSolver.cpp
  ParELAG_BramblePasciakTransformation.cpp
  ParELAG_Hierarchy.cpp
  ParELAG_HiptmairSmoother.cpp
  ParELAG_HybridizationSolver.cpp
  ParELAG_HypreSmootherWrapper.cpp
  ParELAG_KrylovSolver.cpp
  ParELAG_StationarySolver.cpp
  )

list(APPEND SOLVER_OPS_HDRS
  ParELAG_ADSSolverWrapper.hpp
  ParELAG_AMSSolverWrapper.hpp
  ParELAG_Block2x2LDUInverseOperator.hpp
  ParELAG_BlockDiagonalSolver.hpp
  ParELAG_BlockTriangularSolver.hpp
  ParELAG_BoomerAMGSolverWrapper.hpp
  ParELAG_BramblePasciakSolver.hpp
  ParELAG_BramblePasciakTransformation.hpp
  ParELAG_Hierarchy.hpp
  ParELAG_HiptmairSmoother.hpp
  ParELAG_HybridizationSolver.hpp
  ParELAG_HypreSmootherWrapper.hpp
  ParELAG_KrylovSolver.hpp
  ParELAG_StationarySolver.hpp
  )

convert_filenames_to_full_paths(SOLVER_OPS_SRCS)
convert_filenames_to_full_paths(SOLVER_OPS_HDRS)

set(SRCS ${SRCS} ${SOLVER_OPS_SRCS} PARENT_SCOPE)
set(HDRS ${HDRS} ${SOLVER_OPS_HDRS} PARENT_SCOPE)
