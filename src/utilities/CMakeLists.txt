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
  ParELAG_ParameterList.cpp
  ParELAG_SimpleXMLParameterListReader.cpp
  ParELAG_TimeManager.cpp
  Trace.cpp
  UpscalingPieces.cpp
  elagError.cpp
  mpiUtils.cpp
  )

list(APPEND HDRS
  HypreTraits.hpp
  MemoryUtils.hpp
  ParELAG_DataTraits.hpp
  ParELAG_Factory.hpp
  ParELAG_ParameterList.hpp
  ParELAG_SimpleXMLParameterListReader.hpp
  ParELAG_TimeManager.hpp
  ParELAG_Timer.hpp
  ParELAG_VariableContainer.hpp
  ParELAG_VectorTraits.hpp
  ParELAG_Watch.hpp
  Trace.hpp
  UpscalingPieces.hpp
  elagError.hpp
  elag_utilities.hpp
  mpiUtils.hpp
  )

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)

if (${PROJECT_NAME}_ENABLE_TESTING)
  add_subdirectory(unit_test)
endif (${PROJECT_NAME}_ENABLE_TESTING)
