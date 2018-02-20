/*
  Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-745557. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#include <string>

#include "elagError.hpp"

namespace
{
    size_t throw_count = 0;
}

namespace parelag
{
    void Debug_BreakOnMe(const std::string& errorMessage)
    {
        const int break_here = errorMessage.length();
        (void)break_here;
        // Useful for GDB: 'p errorMessage' will print the error that
        // tripped this exception. Examining your stack will hopefully
        // help solve your problem.
    }

    size_t Debug_GetThrowCount()
    {
        return throw_count;
    }

    void Debug_IncreaseThrowCount()
    {
        ++throw_count;
    }
}// namespace parelag

/* Process the error with code ierr raised in the given line of the
   given source file. */
void elag_error_handler(const char *filename,
                        HYPRE_Int line,
                        HYPRE_Int ierr,
                        const char *msg)
{
    if (msg)
    {
        hypre_fprintf(
            stderr, "elag error in file \"%s\", line %d, error code = %d - %s\n",
            filename, line, ierr, msg);
    }
   else
   {
      hypre_fprintf(
         stderr, "elag error in file \"%s\", line %d, error code = %d\n",
         filename, line, ierr);
   }

   throw(ierr);
}
