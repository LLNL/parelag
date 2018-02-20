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
  deleteZeros.c
  hypre_CSRFactory.c
  hypre_ParCSRDataTransformationSign.c
  hypre_ParCSRMatrixCompare.c
  hypre_ParCSRMatrixMatvecBoolInt.c
  hypre_ParCSRMatrixNorms.c
  par_Tmatmul.c
  parcsr-add.c
  )

list(APPEND HDRS
  hypreExtension.hpp
  )

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
