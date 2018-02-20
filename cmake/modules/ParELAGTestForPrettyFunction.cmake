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

include(CheckCXXSourceCompiles)
function(check_for_pretty_function VAR)
  set(TEST_SOURCE
    "
#include <iostream>

int main()
{
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  return 0;
}
")
  check_cxx_source_compiles("${TEST_SOURCE}" ${VAR})
endfunction()
