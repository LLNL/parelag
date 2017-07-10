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

#ifndef ELAGERROR_HPP
#define ELAGERROR_HPP

/*--------------------------------------------------------------------------
 * Global variable used in hypre error checking
 *--------------------------------------------------------------------------*/

extern HYPRE_Int hypre__global_error;
#define hypre_error_flag  hypre__global_error

#define NOT_IMPLEMENTED_YET 999

/*--------------------------------------------------------------------------
 * HYPRE error macros
 *--------------------------------------------------------------------------*/

void elag_error_handler(const char *filename, HYPRE_Int line, HYPRE_Int ierr, const char *msg);
#define elag_error(IERR)  elag_error_handler(__FILE__, __LINE__, IERR, NULL)
#define elag_error_msg(IERR, MSG)  elag_error_handler(__FILE__, __LINE__, IERR, MSG)
#ifndef ELAG_DEBUG
#define elag_assert(EX)
#else
#define elag_assert(EX) if (!(EX)) {hypre_fprintf(stderr,"elag_assert failed: %s\n", #EX); elag_error(1);}
#endif
#ifndef ELAG_DEBUG_PEDANTIC
#define elag_assert_pendantic(EX)
#else
#define elag_assert_pedantic(EX) if (!(EX)) {hypre_fprintf(stderr,"elag_assert failed: %s\n", #EX); elag_error(1);}
#endif

#endif
