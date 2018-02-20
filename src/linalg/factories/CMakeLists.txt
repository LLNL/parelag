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

list(APPEND FACTORIES_SRCS
  ParELAG_ADSSolverFactory.cpp
  ParELAG_AMGeSolverFactory.cpp
  ParELAG_AMSSolverFactory.cpp
  ParELAG_Block2x2GaussSeidelSolverFactory.cpp
  ParELAG_Block2x2JacobiSolverFactory.cpp
  ParELAG_Block2x2LDUSolverFactory.cpp
  ParELAG_BoomerAMGSolverFactory.cpp
  ParELAG_BramblePasciakFactory.cpp
  ParELAG_DirectSolverFactory.cpp
  ParELAG_HiptmairSmootherFactory.cpp
  ParELAG_HybridizationSolverFactory.cpp
  ParELAG_HypreSmootherFactory.cpp
  ParELAG_KrylovSolverFactory.cpp
  ParELAG_SchurComplementFactory.cpp
  )

list(APPEND FACTORIES_HDRS
  ParELAG_ADSSolverFactory.hpp
  ParELAG_AMGeSolverFactory.hpp
  ParELAG_AMSSolverFactory.hpp
  ParELAG_Block2x2GaussSeidelSolverFactory.hpp
  ParELAG_Block2x2JacobiSolverFactory.hpp
  ParELAG_Block2x2LDUSolverFactory.hpp
  ParELAG_BoomerAMGSolverFactory.hpp
  ParELAG_BramblePasciakFactory.hpp
  ParELAG_DirectSolverFactory.hpp
  ParELAG_HiptmairSmootherFactory.hpp
  ParELAG_HybridizationSolverFactory.hpp
  ParELAG_HypreSmootherFactory.hpp
  ParELAG_KrylovSolverFactory.hpp
  ParELAG_SchurComplementFactory.hpp
  ParELAG_StationarySolverFactory.hpp
  )

convert_filenames_to_full_paths(FACTORIES_SRCS)
convert_filenames_to_full_paths(FACTORIES_HDRS)

set(SRCS ${SRCS} ${FACTORIES_SRCS} PARENT_SCOPE)
set(HDRS ${HDRS} ${FACTORIES_HDRS} PARENT_SCOPE)
