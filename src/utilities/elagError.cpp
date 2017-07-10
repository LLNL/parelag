/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#include "_hypre_utilities.h"

/* Process the error with code ierr raised in the given line of the
   given source file. */
void elag_error_handler(const char *filename, HYPRE_Int line, HYPRE_Int ierr, const char *msg)
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
