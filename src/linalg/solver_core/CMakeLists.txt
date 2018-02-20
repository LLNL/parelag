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

list(APPEND SOLVER_CORE_SRCS
  ParELAG_SaddlePointSolver.cpp
  ParELAG_SolverLibrary.cpp
  ParELAG_SymmetrizedUmfpack.cpp
  )

list(APPEND SOLVER_CORE_HDRS
  ParELAG_DirectSolver.hpp
  ParELAG_SaddlePointSolver.hpp
  ParELAG_Solver.hpp
  ParELAG_SolverFactory.hpp
  ParELAG_SolverFactoryCreator.hpp
  ParELAG_SolverLibrary.hpp
  ParELAG_SolverState.hpp
  ParELAG_SymmetrizedUmfpack.hpp
  )

convert_filenames_to_full_paths(SOLVER_CORE_SRCS)
convert_filenames_to_full_paths(SOLVER_CORE_HDRS)

set(SRCS ${SRCS} ${SOLVER_CORE_SRCS} PARENT_SCOPE)
set(HDRS ${HDRS} ${SOLVER_CORE_HDRS} PARENT_SCOPE)
