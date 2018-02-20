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

list(APPEND UTILITIES_SRCS
  ParELAG_ImplicitProductOperator.cpp
  ParELAG_MG_Utils.cpp
  ParELAG_MatrixUtils.cpp
  ParELAG_MfemBlockOperator.cpp
  ParELAG_MonolithicBlockedOperatorFactory.cpp
  ParELAG_SubMatrixExtraction.cpp
  )

list(APPEND UTILITIES_HDRS
  ParELAG_ImplicitProductOperator.hpp
  ParELAG_MG_Utils.hpp
  ParELAG_MG_Utils_impl.hpp
  ParELAG_MatrixUtils.hpp
  ParELAG_MfemBlockOperator.hpp
  ParELAG_MonolithicBlockedOperatorFactory.hpp
  ParELAG_SubMatrixExtraction.hpp
  )

convert_filenames_to_full_paths(UTILITIES_SRCS)
convert_filenames_to_full_paths(UTILITIES_HDRS)

set(SRCS ${SRCS} ${UTILITIES_SRCS} PARENT_SCOPE)
set(HDRS ${HDRS} ${UTILITIES_HDRS} PARENT_SCOPE)
