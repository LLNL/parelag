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

list(APPEND LEGACY_SRCS
  ParELAG_AuxHypreSmoother.cpp
  ParELAG_HypreExtension.cpp
  ParELAG_MLDivFree.cpp
  ParELAG_MLHiptmairSolver.cpp
)

list(APPEND LEGACY_HDRS
  ParELAG_AuxHypreSmoother.hpp
  ParELAG_HypreExtension.hpp
  ParELAG_MLDivFree.hpp
  ParELAG_MLHiptmairSolver.hpp
  )

convert_filenames_to_full_paths(LEGACY_SRCS)
convert_filenames_to_full_paths(LEGACY_HDRS)

set(SRCS ${SRCS} ${LEGACY_SRCS} PARENT_SCOPE)
set(HDRS ${HDRS} ${LEGACY_HDRS} PARENT_SCOPE)
