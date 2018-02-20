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

# this file borrowed from Jed Brown: https://github.com/jedbrown/cmake-modules/
#
# CorrectWindowsPaths - this module defines one macro
#
# CONVERT_CYGWIN_PATH( PATH )
#  This uses the command cygpath (provided by cygwin) to convert
#  unix-style paths into paths useable by cmake on windows

macro (CONVERT_CYGWIN_PATH _path)
  if (WIN32)
    EXECUTE_PROCESS(COMMAND cygpath.exe -m ${${_path}}
      OUTPUT_VARIABLE ${_path})
    string (STRIP ${${_path}} ${_path})
  endif (WIN32)
endmacro (CONVERT_CYGWIN_PATH)

