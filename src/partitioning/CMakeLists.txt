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
  CartesianPartitioner.cpp
  MFEMRefinedMeshPartitioner.cpp
  MetisGraphPartitioner.cpp
  GeometricBoxPartitioner.cpp
  )

list(APPEND HDRS
  CartesianPartitioner.hpp
  LogicalPartitioner.hpp
  MFEMRefinedMeshPartitioner.hpp
  MetisGraphPartitioner.hpp
  GeometricBoxPartitioner.hpp
  CoarsenMetisMaterialId.hpp
  elag_partitioning.hpp
  )

if (${PROJECT_NAME}_ENABLE_PARMETIS)
  list(APPEND SRCS ParmetisGraphPartitioner.cpp)
  list(APPEND HDRS ParmetisGraphPartitioner.hpp)
endif()

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)


