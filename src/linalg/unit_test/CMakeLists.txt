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

list(APPEND UNITTEST_SRCS solver_lib_test.cpp)

if (${PROJECT_NAME}_ENABLE_SUPERLU)
  list(APPEND UNITTEST_SRCS slu_test.cpp)
endif()

if (${PROJECT_NAME}_ENABLE_SUPERLUDIST)
  list(APPEND UNITTEST_SRCS slud_test.cpp)

  # The strumpack test compares against SLUD, so this goes inside the "if SLUD"
  if (${PROJECT_NAME}_ENABLE_STRUMPACK)
    list(APPEND UNITTEST_SRCS strum_test.cpp)
  endif()
endif()

add_parelag_executables(UNITTEST_SRCS)
add_parelag_unittests(UNITTEST_SRCS)
