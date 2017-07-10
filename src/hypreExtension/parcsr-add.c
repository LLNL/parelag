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

#include <mpi.h>
#include "seq_mv.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixAdd
 *
 * Add two ParCSR matrices: C = A + B.
 *--------------------------------------------------------------------------*/
int hypre_ParCSRMatrixAdd(hypre_ParCSRMatrix *A,
                          hypre_ParCSRMatrix *B,
                          hypre_ParCSRMatrix **C_ptr)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   int global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
   int *row_starts = hypre_ParCSRMatrixRowStarts(A);
   int *col_starts = hypre_ParCSRMatrixColStarts(A);
   int A_num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
   int A_num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(A));
   int A_num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(A));
   int B_num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(B));
   int B_num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(B));
   int B_num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(B));

   hypre_ParCSRMatrix *C;
   hypre_CSRMatrix *A_local, *B_local, *C_local;

   A_local = hypre_MergeDiagAndOffd(A);
   B_local = hypre_MergeDiagAndOffd(B);
   C_local = hypre_CSRMatrixAdd(A_local, B_local);

   C = hypre_ParCSRMatrixCreate (comm,
                                 global_num_rows,
                                 global_num_cols,
                                 row_starts,
                                 col_starts,
                                 A_num_cols_offd + B_num_cols_offd,
                                 A_num_nonzeros_diag + B_num_nonzeros_diag,
                                 A_num_nonzeros_offd + B_num_nonzeros_offd);
   GenerateDiagAndOffd(C_local, C,
                       hypre_ParCSRMatrixFirstColDiag(A),
                       hypre_ParCSRMatrixLastColDiag(A));
   hypre_ParCSRMatrixOwnsRowStarts(C) = 0;
   hypre_ParCSRMatrixOwnsColStarts(C) = 0;

   hypre_CSRMatrixDestroy(A_local);
   hypre_CSRMatrixDestroy(B_local);
   hypre_CSRMatrixDestroy(C_local);

   *C_ptr = C;

   return hypre_error_flag;
}

hypre_CSRMatrix *
hypre_CSRMatrixAdd2( double a, hypre_CSRMatrix *A,
              double b, hypre_CSRMatrix *B)
{
   double     *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   double     *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   hypre_CSRMatrix *C;
   double     *C_data;
   HYPRE_Int	      *C_i;
   HYPRE_Int        *C_j;

   HYPRE_Int         ia, ib, ic, jcol, num_nonzeros;
   HYPRE_Int	       pos;
   HYPRE_Int         *marker;

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
              hypre_printf("Warning! incompatible matrix dimensions!\n");
	      return NULL;
   }


   marker = hypre_CTAlloc(HYPRE_Int, ncols_A);
   C_i = hypre_CTAlloc(HYPRE_Int, nrows_A+1);

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
		C_data[pos] = a*A_data[ia];
		marker[jcol] = pos;
		pos++;
	}
	for (ib = B_i[ic]; ib < B_i[ic+1]; ib++)
	{
		jcol = B_j[ib];
		if (marker[jcol] < C_i[ic])
		{
			C_j[pos] = jcol;
			C_data[pos] = b*B_data[ib];
			marker[jcol] = pos;
			pos++;
		}
		else
		{
			C_data[marker[jcol]] += b*B_data[ib];
		}
   	}
   }

   hypre_TFree(marker);
   return C;
}


int hypre_ParCSRMatrixAdd2(double a, hypre_ParCSRMatrix *A,
                          double b, hypre_ParCSRMatrix *B,
                          hypre_ParCSRMatrix **C_ptr)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   int global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
   int *row_starts = hypre_ParCSRMatrixRowStarts(A);
   int *col_starts = hypre_ParCSRMatrixColStarts(A);
   int A_num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
   int A_num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(A));
   int A_num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(A));
   int B_num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(B));
   int B_num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(B));
   int B_num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(B));

   hypre_ParCSRMatrix *C;
   hypre_CSRMatrix *A_local, *B_local, *C_local;

   A_local = hypre_MergeDiagAndOffd(A);
   B_local = hypre_MergeDiagAndOffd(B);
   C_local = hypre_CSRMatrixAdd2(a, A_local, b, B_local);

   C = hypre_ParCSRMatrixCreate (comm,
                                 global_num_rows,
                                 global_num_cols,
                                 row_starts,
                                 col_starts,
                                 A_num_cols_offd + B_num_cols_offd,
                                 A_num_nonzeros_diag + B_num_nonzeros_diag,
                                 A_num_nonzeros_offd + B_num_nonzeros_offd);
   GenerateDiagAndOffd(C_local, C,
                       hypre_ParCSRMatrixFirstColDiag(A),
                       hypre_ParCSRMatrixLastColDiag(A));
   hypre_ParCSRMatrixOwnsRowStarts(C) = 0;
   hypre_ParCSRMatrixOwnsColStarts(C) = 0;

   hypre_CSRMatrixDestroy(A_local);
   hypre_CSRMatrixDestroy(B_local);
   hypre_CSRMatrixDestroy(C_local);

   *C_ptr = C;

   return hypre_error_flag;
}
