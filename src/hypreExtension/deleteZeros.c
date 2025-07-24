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

HYPRE_Int hypre_ParCSRMatrixDeleteZeros( hypre_ParCSRMatrix *A , double tol )
{
    hypre_CSRMatrix * diag = hypre_CSRMatrixDeleteZeros( hypre_ParCSRMatrixDiag(A), tol );
    hypre_CSRMatrix * offd = hypre_CSRMatrixDeleteZeros( hypre_ParCSRMatrixOffd(A), tol );

    if(diag)
    {
        hypre_CSRMatrixDestroy( hypre_ParCSRMatrixDiag(A) );
        hypre_ParCSRMatrixDiag(A) = diag;
    }

    if(offd)
    {
        hypre_CSRMatrixDestroy( hypre_ParCSRMatrixOffd(A) );
        hypre_ParCSRMatrixOffd(A) = offd;
    }

    if( hypre_ParCSRMatrixCommPkg(A) )
    {
        hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkg(A));
        hypre_MatvecCommPkgCreate(A);
    }

    if( hypre_ParCSRMatrixCommPkgT(A) )
    {
        hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkgT(A));
    }

    hypre_ParCSRMatrixSetNumNonzeros(A);

    return 0;
}


/**
   This is orphan code, is never called, and should probably be removed
   unless we have a specific reason to keep it.

   The !FIXME BUGGY!! comment also does not inspire much confidence.
*/
hypre_ParCSRMatrix * hypre_ParCSRMatrixDeleteZeros2 ( hypre_ParCSRMatrix *A , double tol )
{
    MPI_Comm comm = hypre_ParCSRMatrixComm(A);
    HYPRE_Int global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
    HYPRE_Int global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
    HYPRE_Int * row_starts = hypre_ParCSRMatrixRowStarts(A);
    HYPRE_Int * col_starts = hypre_ParCSRMatrixColStarts(A);
    hypre_CSRMatrix * diag = hypre_CSRMatrixDeleteZeros( hypre_ParCSRMatrixDiag(A), tol );
    //! FIXME BUGGY!!
    hypre_CSRMatrix * offd = hypre_CSRMatrixDeleteZeros( hypre_ParCSRMatrixOffd(A), tol );
    HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);

    hypre_ParCSRMatrix * out = hypre_ParCSRMatrixCreate (comm,
                                     global_num_rows,
                                     global_num_cols,
                                     row_starts,
                                     col_starts,
                                     num_cols_offd,
                                     hypre_CSRMatrixNumNonzeros(diag),
                                     hypre_CSRMatrixNumNonzeros(offd));

    hypre_ParCSRMatrixSetDataOwner(out,1);
    hypre_ParCSRMatrixSetRowStartsOwner(out,0);
    hypre_ParCSRMatrixSetColStartsOwner(out,0);

    hypre_CSRMatrixSetDataOwner(out->diag,1);
    hypre_CSRMatrixI(out->diag)    = hypre_CSRMatrixI(diag);
    hypre_CSRMatrixJ(out->diag)    = hypre_CSRMatrixJ(diag);
    hypre_CSRMatrixData(out->diag) = hypre_CSRMatrixData(diag);
    hypre_CSRMatrixSetRownnz(out->diag);

    hypre_CSRMatrixI(diag) = NULL;
    hypre_CSRMatrixJ(diag) = NULL;
    hypre_CSRMatrixData(diag) = NULL;
    hypre_CSRMatrixDestroy(diag);


    hypre_CSRMatrixSetDataOwner(A->offd,1);
    hypre_CSRMatrixI(A->offd)    = hypre_CSRMatrixI(offd);
    hypre_CSRMatrixJ(A->offd)    = hypre_CSRMatrixJ(offd);
    hypre_CSRMatrixData(A->offd) = hypre_CSRMatrixData(offd);
    hypre_CSRMatrixSetRownnz(A->offd);

    hypre_CSRMatrixI(offd) = NULL;
    hypre_CSRMatrixJ(offd) = NULL;
    hypre_CSRMatrixData(offd) = NULL;
    hypre_CSRMatrixDestroy(offd);


    hypre_ParCSRMatrixColMapOffd(out) = parelag_hypre_CTAlloc(HYPRE_Int,num_cols_offd);
    memcpy(hypre_ParCSRMatrixColMapOffd(out), hypre_ParCSRMatrixColMapOffd(A), num_cols_offd*sizeof(HYPRE_Int));
    hypre_ParCSRMatrixSetNumNonzeros(out);

    return out;
}

//Assume A and B have the same sparsity pattern. This routine will overwrite A
HYPRE_Int hypre_ParCSRMatrixKeepEqualEntries(hypre_ParCSRMatrix * A, hypre_ParCSRMatrix * B)
{
    hypre_CSRMatrix * diag_A = hypre_ParCSRMatrixDiag(A);
    hypre_CSRMatrix * diag_B = hypre_ParCSRMatrixDiag(B);

#ifndef NDEBUG
    {
        HYPRE_Int * i_diag_A = hypre_CSRMatrixI(diag_A);
        HYPRE_Int * i_diag_B = hypre_CSRMatrixI(diag_B);
        HYPRE_Int * j_diag_A = hypre_CSRMatrixJ(diag_A);
        HYPRE_Int * j_diag_B = hypre_CSRMatrixJ(diag_B);
        HYPRE_Int nrows_A = hypre_CSRMatrixNumRows(diag_A);
        HYPRE_Int nrows_B = hypre_CSRMatrixNumRows(diag_B);
        HYPRE_Int nnz_B = hypre_CSRMatrixNumNonzeros(diag_B);

        hypre_assert(nrows_A == nrows_B);

        HYPRE_Int i;
        for(i = 0; i < nrows_A+1; ++i)
            hypre_assert(i_diag_A[i] == i_diag_B[i] );

        HYPRE_Int nnz_A = hypre_CSRMatrixNumNonzeros(diag_A);

        hypre_assert(nnz_A == nnz_B);

        for(i = 0; i < nnz_A; ++i)
            hypre_assert(j_diag_A[i] == j_diag_B[i] );
    }
#endif

    double * data_A = hypre_CSRMatrixData(diag_A);
    double * data_B = hypre_CSRMatrixData(diag_B);
    HYPRE_Int nnz_A = hypre_CSRMatrixNumNonzeros(diag_A);

    HYPRE_Int i;
    for(i = 0; i < nnz_A; ++i)
    {
        hypre_assert( fabs( data_A[i] + data_B[i] ) > 1e-9);
        if( fabs( data_A[i] - data_B[i] ) < 1e-9 )
            data_A[i] = 1.; //(data_A[i] > 0) ? 1.: -1.;
        else
            data_A[i] = 0.;
    }

    hypre_CSRMatrix * offd_A = hypre_ParCSRMatrixOffd(A);
    hypre_CSRMatrix * offd_B = hypre_ParCSRMatrixOffd(B);

#ifndef NDEBUG
    {
        HYPRE_Int * i_offd_A = hypre_CSRMatrixI(offd_A);
        HYPRE_Int * i_offd_B = hypre_CSRMatrixI(offd_B);
        HYPRE_Int * j_offd_A = hypre_CSRMatrixJ(offd_A);
        HYPRE_Int * j_offd_B = hypre_CSRMatrixJ(offd_B);
        HYPRE_Int nrows_A = hypre_CSRMatrixNumRows(offd_A);
        HYPRE_Int nrows_B = hypre_CSRMatrixNumRows(offd_B);
        HYPRE_Int nnz_B = hypre_CSRMatrixNumNonzeros(offd_B);

        hypre_assert(nrows_A == nrows_B);

        HYPRE_Int i;
        for(i = 0; i < nrows_A+1; ++i)
            hypre_assert(i_offd_A[i] == i_offd_B[i] );

        HYPRE_Int nnz_A = hypre_CSRMatrixNumNonzeros(offd_A);

        hypre_assert(nnz_A == nnz_B);

        for(i = 0; i < nnz_A; ++i)
            hypre_assert(j_offd_A[i] == j_offd_B[i] );
    }
#endif

    data_A = hypre_CSRMatrixData(offd_A);
    data_B = hypre_CSRMatrixData(offd_B);
    nnz_A = hypre_CSRMatrixNumNonzeros(offd_A);

    for(i = 0; i < nnz_A; ++i)
    {
        hypre_assert( fabs( data_A[i] + data_B[i] ) > 1e-9);
        if( fabs( data_A[i] - data_B[i] ) < 1e-9 )
            data_A[i] = 1.; //(data_A[i] > 0) ? 1.: -1.;
        else
            data_A[i] = 0.;
    }

    return hypre_ParCSRMatrixDeleteZeros(A , 1e-9 );

}
