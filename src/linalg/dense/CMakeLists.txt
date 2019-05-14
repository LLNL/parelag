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

list(APPEND DENSE_SRCS
  ParELAG_Eigensolver.cpp
  ParELAG_InnerProduct.cpp
  ParELAG_LDLCalculator.cpp
  ParELAG_QDQCalculator.cpp
  ParELAG_MultiVector.cpp
  ParELAG_SVDCalculator.cpp
  )

list(APPEND DENSE_HDRS
  ParELAG_Eigensolver.hpp
  ParELAG_InnerProduct.hpp
  ParELAG_LAPACK.hpp
  ParELAG_LAPACK_Helpers.hpp
  ParELAG_DenseInverseCalculator.hpp
  ParELAG_LDLCalculator.hpp
  ParELAG_QDQCalculator.hpp
  ParELAG_MultiVector.hpp
  ParELAG_SVDCalculator.hpp
  )

convert_filenames_to_full_paths(DENSE_SRCS)
convert_filenames_to_full_paths(DENSE_HDRS)

set(SRCS ${SRCS} ${DENSE_SRCS} PARENT_SCOPE)
set(HDRS ${HDRS} ${DENSE_HDRS} PARENT_SCOPE)
