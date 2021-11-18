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
#include <mpi.h>

// l1 norm: maximum absolute column sum of the matrix
// || A ||_linf = max_j ( sum_i |a_ij| )
double
hypre_ParCSRMatrixNorml1(hypre_ParCSRMatrix * A)
{
    hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(A);
    hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd(A);

    HYPRE_Int nnz_diag = hypre_CSRMatrixNumNonzeros(diag);
    HYPRE_Int nnz_offd = hypre_CSRMatrixNumNonzeros(offd);

    double * abs_data_diag = parelag_hypre_CTAlloc(double, nnz_diag);
    double * abs_data_offd = parelag_hypre_CTAlloc(double, nnz_offd);
    double * orig_data_diag = hypre_CSRMatrixData(diag);
    double * orig_data_offd = hypre_CSRMatrixData(offd);
    hypre_CSRMatrixData(diag) = abs_data_diag;
    hypre_CSRMatrixData(offd) = abs_data_offd;

    int i;
    for(i = 0; i < nnz_diag; ++i)
        abs_data_diag[i] = fabs(orig_data_diag[i]);
    for(i = 0; i < nnz_offd; ++i)
        abs_data_offd[i] = fabs(orig_data_offd[i]);

    hypre_ParVector * x = hypre_ParVectorInRangeOf(A);
    hypre_ParVector * y = hypre_ParVectorInDomainOf(A);

    double * ones = hypre_VectorData( hypre_ParVectorLocalVector(x) );
    int local_range_size = hypre_VectorSize( hypre_ParVectorLocalVector(x) );

    for(i = 0; i < local_range_size; ++i)
        ones[i] = 1;

    hypre_ParCSRMatrixMatvecT(1., A, x, 0., y);

    double * data_y = hypre_VectorData( hypre_ParVectorLocalVector(y) );
    int local_domain_size = hypre_VectorSize( hypre_ParVectorLocalVector(y) );

    double locl1norm = 0;
    for(i = 0; i < local_domain_size; ++i)
        if( data_y[i] > locl1norm)
            locl1norm = data_y[i];

    double globall1norm = 0;
    MPI_Allreduce(&locl1norm, &globall1norm, 1, MPI_DOUBLE, MPI_MAX, hypre_ParCSRMatrixComm(A));

    hypre_CSRMatrixData(diag) = orig_data_diag;
    hypre_CSRMatrixData(offd) = orig_data_offd;
    parelag_hypre_TFree(abs_data_diag);
    parelag_hypre_TFree(abs_data_offd);
    hypre_ParVectorDestroy(x);
    hypre_ParVectorDestroy(y);

    return globall1norm;

}

// linf norm: maximum absolute row sum of the matrix:
// || A ||_linf = max_i ( sum_j |a_ij| )
double
hypre_ParCSRMatrixNormlinf(hypre_ParCSRMatrix * A)
{
    hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(A);
    hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd(A);

    HYPRE_Int nnz_diag = hypre_CSRMatrixNumNonzeros(diag);
    HYPRE_Int nnz_offd = hypre_CSRMatrixNumNonzeros(offd);

    double * abs_data_diag = parelag_hypre_CTAlloc(double, nnz_diag);
    double * abs_data_offd = parelag_hypre_CTAlloc(double, nnz_offd);
    double * orig_data_diag = hypre_CSRMatrixData(diag);
    double * orig_data_offd = hypre_CSRMatrixData(offd);
    hypre_CSRMatrixData(diag) = abs_data_diag;
    hypre_CSRMatrixData(offd) = abs_data_offd;

    int i;
    for(i = 0; i < nnz_diag; ++i)
        abs_data_diag[i] = fabs(orig_data_diag[i]);
    for(i = 0; i < nnz_offd; ++i)
        abs_data_offd[i] = fabs(orig_data_offd[i]);

    hypre_ParVector * x = hypre_ParVectorInDomainOf(A);
    hypre_ParVector * y = hypre_ParVectorInRangeOf(A);

    double * ones = hypre_VectorData( hypre_ParVectorLocalVector(x) );
    int local_domain_size = hypre_VectorSize( hypre_ParVectorLocalVector(x) );

    for(i = 0; i < local_domain_size; ++i)
        ones[i] = 1;

    hypre_ParCSRMatrixMatvec(1., A, x, 0., y);

    double * data_y = hypre_VectorData( hypre_ParVectorLocalVector(y) );
    int local_range_size = hypre_VectorSize( hypre_ParVectorLocalVector(y) );

    double loclinfnorm = 0;
    for(i = 0; i < local_range_size; ++i)
        if( data_y[i] > loclinfnorm)
            loclinfnorm = data_y[i];

    double globallinfnorm = 0;
    MPI_Allreduce(&loclinfnorm, &globallinfnorm, 1, MPI_DOUBLE, MPI_MAX, hypre_ParCSRMatrixComm(A));

    hypre_CSRMatrixData(diag) = orig_data_diag;
    hypre_CSRMatrixData(offd) = orig_data_offd;
    parelag_hypre_TFree(abs_data_diag);
    parelag_hypre_TFree(abs_data_offd);
    hypre_ParVectorDestroy(x);
    hypre_ParVectorDestroy(y);

    return globallinfnorm;
}

// Max norm of A = max_{ij} | a_ij |
double
hypre_ParCSRMatrixMaxNorm(hypre_ParCSRMatrix * A)
{
    hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(A);
    hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd(A);

    double locMaxNormDiag = hypre_CSRMatrixMaxNorm(diag);
    double locMaxNormOffd = hypre_CSRMatrixMaxNorm(offd);
    double locMaxNorm = (locMaxNormDiag > locMaxNormOffd) ? locMaxNormDiag : locMaxNormOffd;

    double globalMaxNorm = 0;
    MPI_Allreduce(&locMaxNorm, &globalMaxNorm, 1, MPI_DOUBLE, MPI_MAX, hypre_ParCSRMatrixComm(A));

    return globalMaxNorm;
}

// Frobenius norm A = sqrt( sum_i sum_j |a_ij|^2 )
double
hypre_ParCSRMatrixFrobeniusNorm(hypre_ParCSRMatrix * A)
{
    hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(A);
    hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd(A);

    double locFrobNormDiag = hypre_CSRMatrixFrobeniusNorm(diag);
    double locFrobNormOffd = hypre_CSRMatrixFrobeniusNorm(offd);
    double locFrobNorm2 = (locFrobNormDiag*locFrobNormDiag) + (locFrobNormOffd*locFrobNormOffd);

    double globalFrobNorm2 = 0;
    MPI_Allreduce(&locFrobNorm2, &globalFrobNorm2, 1, MPI_DOUBLE, MPI_SUM, hypre_ParCSRMatrixComm(A));

    return sqrt( globalFrobNorm2 );
}

double
hypre_CSRMatrixMaxNorm(hypre_CSRMatrix * A)
{
    double * data = hypre_CSRMatrixData(A);
    int nnz = hypre_CSRMatrixNumNonzeros(A);

    double maxNorm = 0;
    double help = 0;

    int i;

    for(i = 0; i < nnz; ++i)
        if( (help = fabs(data[i])) > maxNorm )
            maxNorm = help;

    return maxNorm;
}

double
hypre_CSRMatrixFrobeniusNorm(hypre_CSRMatrix * A)
{
    double * data = hypre_CSRMatrixData(A);
    int nnz = hypre_CSRMatrixNumNonzeros(A);

    double frobNorm2 = 0;

    int i;

    for(i = 0; i < nnz; ++i)
        frobNorm2 += data[i] * data[i];

    return sqrt(frobNorm2);
}
