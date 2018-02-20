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

hypre_CSRMatrix * hypre_ZerosCSRMatrix( HYPRE_Int nrows, HYPRE_Int ncols)
{
    hypre_CSRMatrix * A = hypre_CSRMatrixCreate( nrows, ncols, 0);
    hypre_CSRMatrixInitialize( A );

    HYPRE_Int * i_A = hypre_CSRMatrixI(A);

    HYPRE_Int i;
    for( i = 0; i < nrows+1; ++i)
        i_A[i] = 0;

    return A;
}

hypre_CSRMatrix * hypre_IdentityCSRMatrix( HYPRE_Int nrows)
{
    hypre_CSRMatrix * A = hypre_CSRMatrixCreate( nrows, nrows, nrows);
    hypre_CSRMatrixInitialize( A );

    HYPRE_Int * i_A = hypre_CSRMatrixI(A);
    HYPRE_Int * j_A = hypre_CSRMatrixJ(A);
    double    * a_A = hypre_CSRMatrixData(A);

    HYPRE_Int i;
    for( i = 0; i < nrows+1; ++i)
        i_A[i] = i;

    for( i = 0; i < nrows; ++i)
    {
        j_A[i] = i;
        a_A[i] = 1.;
    }

    return A;
}

hypre_CSRMatrix * hypre_DiagonalCSRMatrix( HYPRE_Int nrows, double * d)
{
    hypre_CSRMatrix * A = hypre_CSRMatrixCreate( nrows, nrows, nrows);
    hypre_CSRMatrixInitialize( A );

    HYPRE_Int * i_A = hypre_CSRMatrixI(A);
    HYPRE_Int * j_A = hypre_CSRMatrixJ(A);
    double    * a_A = hypre_CSRMatrixData(A);

    HYPRE_Int i;
    for( i = 0; i < nrows+1; ++i)
        i_A[i] = i;

    for( i = 0; i < nrows; ++i)
    {
        j_A[i] = i;
        a_A[i] = d[i];
    }

    return A;
}

hypre_ParCSRMatrix * hypre_IdentityParCSRMatrix( MPI_Comm comm, HYPRE_Int global_num_rows, HYPRE_Int * row_starts)
{
    HYPRE_Int num_nonzeros_diag;
    if(HYPRE_AssumedPartitionCheck())
    {
        num_nonzeros_diag = row_starts[1] - row_starts[0];
        hypre_assert(row_starts[2] == global_num_rows);
    }
    else
    {
        HYPRE_Int pid;
        HYPRE_Int np;

        MPI_Comm_rank(comm, &pid);
        MPI_Comm_size(comm, &np);

        num_nonzeros_diag = row_starts[pid+1] - row_starts[pid];
        hypre_assert(row_starts[np] == global_num_rows);

    }
    hypre_ParCSRMatrix * mat = hypre_ParCSRMatrixCreate ( comm , global_num_rows , global_num_rows ,
            row_starts , row_starts , 0 , num_nonzeros_diag, 0 );

    hypre_ParCSRMatrixSetRowStartsOwner(mat, 0);
    hypre_ParCSRMatrixSetColStartsOwner(mat, 0);

    hypre_ParCSRMatrixColMapOffd(mat) = NULL;

    hypre_CSRMatrixDestroy( hypre_ParCSRMatrixDiag(mat) );
    hypre_CSRMatrixDestroy( hypre_ParCSRMatrixOffd(mat) );

    hypre_ParCSRMatrixDiag(mat) = hypre_IdentityCSRMatrix(num_nonzeros_diag);
    hypre_ParCSRMatrixOffd(mat) = hypre_ZerosCSRMatrix(num_nonzeros_diag, 0);

    hypre_ParCSRMatrixSetDataOwner(mat, 1);

    return mat;
}

hypre_ParCSRMatrix * hypre_DiagonalParCSRMatrix( MPI_Comm comm, HYPRE_Int global_num_rows,
                                                                 HYPRE_Int * row_starts, double * d)
{
    HYPRE_Int num_nonzeros_diag;
    if(HYPRE_AssumedPartitionCheck())
    {
        num_nonzeros_diag = row_starts[1] - row_starts[0];
        hypre_assert(row_starts[2] == global_num_rows);
    }
    else
    {
        HYPRE_Int pid;
        HYPRE_Int np;

        MPI_Comm_rank(comm, &pid);
        MPI_Comm_size(comm, &np);

        num_nonzeros_diag = row_starts[pid+1] - row_starts[pid];
        hypre_assert(row_starts[np] == global_num_rows);

    }
    hypre_ParCSRMatrix * mat = hypre_ParCSRMatrixCreate ( comm , global_num_rows , global_num_rows ,
            row_starts , row_starts , 0 , num_nonzeros_diag, 0 );
    hypre_ParCSRMatrixSetRowStartsOwner(mat, 0);
    hypre_ParCSRMatrixSetColStartsOwner(mat, 0);
    hypre_ParCSRMatrixColMapOffd(mat) = NULL;
    hypre_ParCSRMatrixDiag(mat) = hypre_DiagonalCSRMatrix(num_nonzeros_diag, d);
    hypre_ParCSRMatrixOffd(mat) = hypre_ZerosCSRMatrix(num_nonzeros_diag, 0);
    hypre_ParCSRMatrixSetRowStartsOwner(mat, 0);
    hypre_ParCSRMatrixSetDataOwner(mat, 1);

    return mat;
}

hypre_ParCSRMatrix * hypre_IdentityParCSRMatrixOffsets( MPI_Comm comm, HYPRE_Int global_num_rows, 
                                                                          HYPRE_Int * row_starts, HYPRE_Int * col_starts)
{
    HYPRE_Int pid;
    HYPRE_Int np;

    MPI_Comm_rank(comm, &pid);
    MPI_Comm_size(comm, &np);

    HYPRE_Int num_rows, num_nonzeros_diag, num_cols_diag, num_nonzeros_offd,
        begin_row, end_row, begin_col, end_col;

    if (HYPRE_AssumedPartitionCheck())
    {
        begin_row = row_starts[0];
        end_row = row_starts[1];
        begin_col = col_starts[0];
        end_col = col_starts[1];
        hypre_assert(row_starts[2] == global_num_rows);
    }
    else
    {
        begin_row = row_starts[pid];
        end_row = row_starts[pid+1];
        begin_col = col_starts[pid];
        end_col = col_starts[pid+1];
        hypre_assert(row_starts[np] == global_num_rows);
    }
    num_rows = end_row - begin_row;
    num_cols_diag = end_col - begin_col;

    num_nonzeros_diag = num_rows;
    if (num_cols_diag < num_nonzeros_diag) num_nonzeros_diag = num_cols_diag;
    if (end_row - begin_col < num_nonzeros_diag) num_nonzeros_diag = end_row - begin_col;
    if (end_col - begin_row < num_nonzeros_diag) num_nonzeros_diag = end_col - begin_row;
    if (num_nonzeros_diag < 0) num_nonzeros_diag = 0;
    num_nonzeros_offd = num_rows - num_nonzeros_diag;
    hypre_assert(num_nonzeros_offd >= 0);

    hypre_ParCSRMatrix * mat = hypre_ParCSRMatrixCreate ( comm , global_num_rows , global_num_rows ,
                                                                            row_starts , col_starts , 
                                                                            num_nonzeros_offd , num_nonzeros_diag, num_nonzeros_offd );

    // row_starts, col_starts still owned by caller
    hypre_ParCSRMatrixSetRowStartsOwner(mat, 0);
    hypre_ParCSRMatrixSetColStartsOwner(mat, 0);
    
    hypre_ParCSRMatrixInitialize(mat);

    hypre_CSRMatrix * diag = hypre_CSRMatrixCreate(num_rows, num_cols_diag, num_nonzeros_diag);
    hypre_CSRMatrix * offd = hypre_CSRMatrixCreate(num_rows, num_nonzeros_offd, num_nonzeros_offd);
    hypre_CSRMatrixInitialize(diag);
    hypre_CSRMatrixInitialize(offd);

    HYPRE_Int * i_diag = hypre_CSRMatrixI(diag);
    HYPRE_Int * j_diag = hypre_CSRMatrixJ(diag);
    double    * a_diag = hypre_CSRMatrixData(diag);

    HYPRE_Int * i_offd = hypre_CSRMatrixI(offd);
    HYPRE_Int * j_offd = hypre_CSRMatrixJ(offd);
    double    * a_offd = hypre_CSRMatrixData(offd);

    HYPRE_Int * col_map_offd = mat->col_map_offd;
    int offd_j=0;
    int diag_j=0;
    int i;
    for (i=0; i<num_rows; ++i)
    {
        // if i in local column range
        if ((begin_col <= begin_row + i) && (begin_row + i < end_col))
        {
            i_diag[i] = diag_j;
            j_diag[diag_j] = begin_row - begin_col + i; // local numbering
            a_diag[diag_j] = 1.0;
            diag_j++;

            i_offd[i] = offd_j;
        }
        else
        {
            i_offd[i] = offd_j;
            j_offd[offd_j] = offd_j; // local (mapped) numbering
            a_offd[offd_j] = 1.0;
            col_map_offd[offd_j] = begin_row + i; // global numbering
            offd_j++;

            i_diag[i] = diag_j;
        }
    }
    i_diag[num_rows] = diag_j;
    i_offd[num_rows] = offd_j;
    hypre_assert(diag_j == num_nonzeros_diag);
    hypre_assert(offd_j == num_nonzeros_offd);

    hypre_CSRMatrixDestroy( hypre_ParCSRMatrixDiag(mat) );
    hypre_CSRMatrixDestroy( hypre_ParCSRMatrixOffd(mat) );

    hypre_ParCSRMatrixDiag(mat) = diag;
    hypre_ParCSRMatrixOffd(mat) = offd;

    hypre_ParCSRMatrixSetDataOwner(mat, 1);

    return mat;
}
