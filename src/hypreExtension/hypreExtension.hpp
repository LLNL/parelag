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

#ifndef HYPREEXTENSION_H_
#define HYPREEXTENSION_H_

#include "ParELAG_Config.h"
#include PARELAG_MFEM_CONFIG_HEADER

#include "seq_mv.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"

#if MFEM_HYPRE_VERSION >= 21400
#define parelag_hypre_CTAlloc(type, size) \
    hypre_CTAlloc(type, size, HYPRE_MEMORY_HOST)
#define parelag_hypre_TFree(ptr) hypre_TFree(ptr, HYPRE_MEMORY_HOST)
#else // MFEM_HYPRE_VERSION < 21400
#define parelag_hypre_CTAlloc(type, size) hypre_CTAlloc(type, size)
#define parelag_hypre_TFree(ptr) hypre_TFree(ptr)
#endif

#if MFEM_HYPRE_VERSION > 22200
#define hypre_ParCSRMatrixSetRowStartsOwner(A, own)
#define hypre_ParCSRMatrixSetColStartsOwner(A, own)
#endif

#ifdef __cplusplus
extern "C"
{
#endif

hypre_CSRMatrix * hypre_ZerosCSRMatrix( HYPRE_Int nrows, HYPRE_Int ncols);
hypre_CSRMatrix * hypre_IdentityCSRMatrix( HYPRE_Int nrows);

hypre_CSRMatrix * hypre_DiagonalCSRMatrix( HYPRE_Int nrows, double * d);
hypre_ParCSRMatrix * hypre_IdentityParCSRMatrix( MPI_Comm comm, HYPRE_Int global_num_rows,
                                                                 HYPRE_Int * row_starts);
hypre_ParCSRMatrix * hypre_IdentityParCSRMatrixOffsets( MPI_Comm comm, HYPRE_Int global_num_rows,
                                                                          HYPRE_Int * row_starts, HYPRE_Int * col_starts);
hypre_ParCSRMatrix * hypre_DiagonalParCSRMatrix( MPI_Comm comm, HYPRE_Int global_num_rows,
                                                                 HYPRE_Int * row_starts, double * d);

HYPRE_Int
hypre_RDP(hypre_ParCSRMatrix  *RT,
        double * d,
        hypre_ParCSRMatrix  *P,
        hypre_ParCSRMatrix **RDP_ptr);

HYPRE_Int
hypre_ParCSRMatrixTranspose2(hypre_ParCSRMatrix  *RT,
        hypre_ParCSRMatrix **out);


// l1 norm: maximum absolute column sum of the matrix
// || A ||_linf = max_j ( sum_i |a_ij| )
double
hypre_ParCSRMatrixNorml1(hypre_ParCSRMatrix * A);
// linf norm: maximum absolute row sum of the matrix:
// || A ||_linf = max_i ( sum_j |a_ij| )
double
hypre_ParCSRMatrixNormlinf(hypre_ParCSRMatrix * A);
// Max norm of A = max_{ij} | a_ij |
double
hypre_ParCSRMatrixMaxNorm(hypre_ParCSRMatrix * A);
// Frobenius norm A = sqrt( sum_i sum_j |a_ij|^2 )
double
hypre_ParCSRMatrixFrobeniusNorm(hypre_ParCSRMatrix * A);
double
hypre_CSRMatrixMaxNorm(hypre_CSRMatrix * A);
double
hypre_CSRMatrixFrobeniusNorm(hypre_CSRMatrix * A);

void hypre_CSRDataTransformationSign(hypre_CSRMatrix * mat);
void hypre_ParCSRDataTransformationSign(hypre_ParCSRMatrix * mat);

HYPRE_Int hypre_ParCSRMatrixCompare(hypre_ParCSRMatrix * A, hypre_ParCSRMatrix * B, double tol, int verbose);


int parelag_ParCSRMatrixAdd(hypre_ParCSRMatrix *A,
                          hypre_ParCSRMatrix *B,
                          hypre_ParCSRMatrix **C_ptr);

hypre_CSRMatrix *
hypre_CSRMatrixAdd2( double a, hypre_CSRMatrix *A,
              double b, hypre_CSRMatrix *B);

int hypre_ParCSRMatrixAdd2(double a, hypre_ParCSRMatrix *A,
                          double b, hypre_ParCSRMatrix *B,
                          hypre_ParCSRMatrix **C_ptr);


HYPRE_Int hypre_ParCSRMatrixDeleteZeros ( hypre_ParCSRMatrix *A , double tol );
HYPRE_Int hypre_ParCSRMatrixKeepEqualEntries(hypre_ParCSRMatrix * A, hypre_ParCSRMatrix * B);

HYPRE_Int hypre_ParCSRMatrixMatvecBoolInt(int alpha, hypre_ParCSRMatrix *A, int *x_local, int beta, int *y_local );
HYPRE_Int hypre_CSRMatrixMatvecBoolInt( int alpha, hypre_CSRMatrix *A, int    *x, int beta, int *y);

#ifdef __cplusplus
}
#endif

#endif /* HYPREEXTENSION_H_ */
