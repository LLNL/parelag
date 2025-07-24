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

#include "hypreExtension.hpp"


HYPRE_Int
hypre_RDP(hypre_ParCSRMatrix  *RT,
        double * d,
        hypre_ParCSRMatrix  *P,
        hypre_ParCSRMatrix **RDP_ptr)
{
    hypre_assert(hypre_ParCSRMatrixGlobalNumRows(RT) == hypre_ParCSRMatrixGlobalNumRows(P) );
    MPI_Comm comm = hypre_ParCSRMatrixComm(RT);
    HYPRE_Int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(RT);
    HYPRE_Int * row_starts = hypre_ParCSRMatrixRowStarts(RT);
    hypre_ParCSRMatrix * D = hypre_DiagonalParCSRMatrix( comm, global_num_rows, row_starts, d);

    hypre_ParCSRMatrix * RD = hypre_ParTMatmul( RT, D );
    *RDP_ptr = hypre_ParMatmul(RD, P);

    hypre_ParCSRMatrixSetRowStartsOwner(*RDP_ptr, 0);
    hypre_ParCSRMatrixSetColStartsOwner(*RDP_ptr, 0);

    hypre_ParCSRMatrixDestroy(RD);
    hypre_ParCSRMatrixDestroy(D);

    return 0;
}

HYPRE_Int
hypre_ParCSRMatrixTranspose2(hypre_ParCSRMatrix  *A,
        hypre_ParCSRMatrix **At)
{
#if 0
    //THERE IS A BUG :(
    MPI_Comm comm = hypre_ParCSRMatrixComm(A);
    HYPRE_Int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
    HYPRE_Int * row_starts = hypre_ParCSRMatrixRowStarts(A);
    hypre_ParCSRMatrix * I = hypre_IdentityParCSRMatrix( comm, global_num_rows, row_starts );
    *At = hypre_ParTMatmul(A, I );

    hypre_ParCSRMatrixSetRowStartsOwner(*At, 0);
    hypre_ParCSRMatrixSetColStartsOwner(*At, 0);


    hypre_ParCSRMatrixDestroy(I);

    return 0;
#else
    hypre_ParCSRMatrixTranspose(A,At,1);

#if MFEM_HYPRE_VERSION <= 22200
    if(hypre_ParCSRMatrixOwnsRowStarts(*At))
        parelag_hypre_TFree(hypre_ParCSRMatrixRowStarts(*At));

    if(hypre_ParCSRMatrixOwnsColStarts(*At))
        parelag_hypre_TFree(hypre_ParCSRMatrixColStarts(*At));

    hypre_ParCSRMatrixRowStarts(*At) = hypre_ParCSRMatrixColStarts(A);
    hypre_ParCSRMatrixColStarts(*At) = hypre_ParCSRMatrixRowStarts(A);

    hypre_ParCSRMatrixSetRowStartsOwner(*At, 0);
    hypre_ParCSRMatrixSetColStartsOwner(*At, 0);
#endif

    return 0;
#endif
}
