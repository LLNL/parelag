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

#include <limits.h>
#include <assert.h>
#include "hypreExtension.hpp"

hypre_CSRMatrix *hypre_CSRMatrixSubtract ( hypre_CSRMatrix *A , hypre_CSRMatrix *B )
{
       double           *A_data   = hypre_CSRMatrixData(A);
       HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
#if MFEM_HYPRE_VERSION >= 21600
       HYPRE_BigInt     *A_j      = hypre_CSRMatrixBigJ(A);
#else
       HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
#endif
       HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
       HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
       double           *B_data   = hypre_CSRMatrixData(B);
       HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
#if MFEM_HYPRE_VERSION >= 21600
       HYPRE_BigInt     *B_j      = hypre_CSRMatrixBigJ(B);
#else
       HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
#endif
       HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
       HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
       hypre_CSRMatrix  *C;
       double           *C_data;
       HYPRE_Int	    *C_i;
       HYPRE_Int        *C_j;

       HYPRE_Int         ia, ib, ic, jcol, num_nonzeros;
       HYPRE_Int	     pos;
       HYPRE_Int        *marker;

       assert(ncols_A <= INT_MAX);

       if (nrows_A != nrows_B || ncols_A != ncols_B)
       {
                  hypre_printf("Warning! incompatible matrix dimensions!\n");
              return NULL;
       }


       marker = parelag_hypre_CTAlloc(HYPRE_Int, ncols_A);
       C_i = parelag_hypre_CTAlloc(HYPRE_Int, nrows_A+1);

       for (ia = 0; ia < ncols_A; ia++)
        marker[ia] = -1;

       num_nonzeros = 0;
       C_i[0] = 0;
       for (ic = 0; ic < nrows_A; ic++)
       {
        for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
        {
            jcol = A_j[ia];
            marker[jcol] = ic;
            num_nonzeros++;
        }
        for (ib = B_i[ic]; ib < B_i[ic+1]; ib++)
        {
            jcol = B_j[ib];
            if (marker[jcol] != ic)
            {
                marker[jcol] = ic;
                num_nonzeros++;
            }
        }
        C_i[ic+1] = num_nonzeros;
       }

       C = hypre_CSRMatrixCreate(nrows_A, ncols_A, num_nonzeros);
       hypre_CSRMatrixI(C) = C_i;
       hypre_CSRMatrixInitialize(C);
       C_j = hypre_CSRMatrixJ(C);
       C_data = hypre_CSRMatrixData(C);

       for (ia = 0; ia < ncols_A; ia++)
        marker[ia] = -1;

       pos = 0;
       for (ic = 0; ic < nrows_A; ic++)
       {
        for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
        {
            jcol = A_j[ia];
            C_j[pos] = jcol;
            C_data[pos] = A_data[ia];
            marker[jcol] = pos;
            pos++;
        }
        for (ib = B_i[ic]; ib < B_i[ic+1]; ib++)
        {
            jcol = B_j[ib];
            if (marker[jcol] < C_i[ic])
            {
                C_j[pos] = jcol;
                C_data[pos] = -B_data[ib];
                marker[jcol] = pos;
                pos++;
            }
            else
            {
                C_data[marker[jcol]] -= B_data[ib];
            }
        }
       }

       parelag_hypre_TFree(marker);
       return C;
}

HYPRE_Int hypre_ParCSRMatrixCompare(hypre_ParCSRMatrix * A, hypre_ParCSRMatrix * B, double tol, int verbose)
{

    HYPRE_Int cmp = 0;
    if( hypre_ParCSRMatrixGlobalNumRows(A) != hypre_ParCSRMatrixGlobalNumRows(B))
    {
        if(verbose)
            printf("A: global number of rows %d \n B: global number of rows %d \n", hypre_ParCSRMatrixGlobalNumRows(A), hypre_ParCSRMatrixGlobalNumRows(B));
        cmp |= 1;
    }

    if( hypre_ParCSRMatrixGlobalNumCols(A) != hypre_ParCSRMatrixGlobalNumCols(B) )
    {
        if(verbose)
            printf("A: global number of cols %d \n B: global number of cols %d \n", hypre_ParCSRMatrixGlobalNumCols(A), hypre_ParCSRMatrixGlobalNumCols(B));
        cmp |= 2;
    }

    if( hypre_ParCSRMatrixFirstRowIndex(A) != hypre_ParCSRMatrixFirstRowIndex(B) )
    {
        if(verbose)
            printf("A: first row index %d \n B: first row index %d \n", hypre_ParCSRMatrixFirstRowIndex(A), hypre_ParCSRMatrixFirstRowIndex(B));
        cmp |= 4;
    }

    if( hypre_ParCSRMatrixLastRowIndex(A) != hypre_ParCSRMatrixLastRowIndex(B) )
    {
        if(verbose)
            printf("A: last row index %d \n B: last row index %d \n", hypre_ParCSRMatrixLastRowIndex(A), hypre_ParCSRMatrixLastRowIndex(B));
        cmp |= 8;
    }

    if( hypre_ParCSRMatrixFirstColDiag(A) != hypre_ParCSRMatrixFirstColDiag(B) )
    {
        if(verbose)
            printf("A: first col diag %d \n B: first col diag %d \n", hypre_ParCSRMatrixFirstColDiag(A), hypre_ParCSRMatrixFirstColDiag(B));
        cmp |= 16;
    }

    if( hypre_ParCSRMatrixLastColDiag(A) != hypre_ParCSRMatrixLastColDiag(B) )
    {
        if(verbose)
            printf("A: last col diag %d \n B: last col diag %d \n", hypre_ParCSRMatrixLastColDiag(A), hypre_ParCSRMatrixLastColDiag(B));
        cmp |= 32;
    }


    hypre_CSRMatrix * a = hypre_MergeDiagAndOffd(A);
    hypre_CSRMatrix * b = hypre_MergeDiagAndOffd(B);

    hypre_CSRMatrix * c = hypre_CSRMatrixSubtract(a,b);

    double diff_loc = hypre_CSRMatrixMaxNorm(c);
    double diff = 0;

    MPI_Allreduce(&diff_loc, &diff, 1, MPI_DOUBLE, MPI_SUM, hypre_ParCSRMatrixComm(A));

    if( diff > tol )
    {
        if(verbose)
            printf("||A - B||_max %10g %10g\n", diff, diff_loc);
        cmp |= 64;
    }

    hypre_CSRMatrixDestroy(c);
    hypre_CSRMatrixDestroy(a);
    hypre_CSRMatrixDestroy(b);

    return cmp;

}
