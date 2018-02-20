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

#ifndef PARELAG_LAPACK_HELPERS_HPP_
#define PARELAG_LAPACK_HELPERS_HPP_

namespace parelag
{

extern "C"
{
    //*** Single precision real ***//
    void sgesvd_(char* JOBU, char* JOBVT, int* M, int* N, float* A, int* LDA,
                 float* S, float* U, int* LDU,
                 float* VT, int* LDVT, float* WORK, int* LWORK,
                 int* INFO);

    void sgetrf_(int* M, int* N, float* A, int* LDA, int* IPIV, int* INFO);

    void sgetrs_(char* TRANS, int* N, int* NRHS, float* A, int* LDA, int* IPIV,
                 float* B, int* LDB, int* INFO);

    void ssyevx_(char* JOBZ, char* RANGE, char* UPLO,
                 int* N, float* A, int* LDA, float* VL, float* VU,
                 int* IL, int* IU, float* ABSTOL, int* M, float* W,
                 float* Z, int* LDZ, float* WORK, int* LWORK, int* IWORK,
                 int* IFAIL, int* INFO);

    void ssygvx_(int* ITYPE, char* JOBZ, char* RANGE, char* UPLO,
                 int* N, float* A, int* LDA, float* B, int* LDB,
                 float* VL, float* VU, int* IL, int* IU, float* ABSTOL,
                 int* M, float* W, float* Z__, int* LDZ,
                 float* WORK, int* LWORK, int* IWORK, int* IFAIL, int* INFO);

    void ssytrf_(char* UPLO, int* N, float* A, int* LDA, int* IPIV,
                 float* WORK, int* LWORK, int* INFO);

    void ssytrs_(char* UPLO, int* N, int* NRHS, float* A, int* LDA, int* IPIV,
                 float* B, int* LDB, int* INFO);

    //*** Double precision real ***//
    void dgesvd_(char* JOBU, char* JOBVT, int* M, int* N, double* A, int* LDA,
                 double* S, double* U, int* LDU, double* VT,
                 int* LDVT, double* WORK, int* LWORK, int* INFO);

    void dgetrf_(int* M, int* N, double* A, int* LDA, int* IPIV, int* INFO);

    void dgetrs_(char* TRANS, int* N, int* NRHS, double* A, int* LDA, int* IPIV,
                 double* B, int* LDB, int* INFO);

    void dsyevx_(char* JOBZ, char* RANGE, char* UPLO,
                 int* N, double* A, int* LDA,
                 double* VL, double* VU, int* IL, int* IU, double* ABSTOL,
                 int* M, double* W, double* Z, int* LDZ,
                 double* WORK, int* LWORK, int* IWORK, int* IFAIL, int* INFO);

    void dsygvx_(int* ITYPE, char* JOBZ, char* RANGE, char* UPLO,
                 int* N, double* A, int* LDA, double* B, int* LDB,
                 double* VL, double* VU, int* IL, int* IU, double* ABSTOL,
                 int* M, double* W, double* Z__, int* LDZ,
                 double* WORK, int* LWORK, int* IWORK, int* IFAIL, int* INFO);

    void dsytrf_(char* UPLO, int* N, double* A, int* LDA, int* IPIV,
                 double* WORK, int* LWORK, int* INFO);

    void dsytrs_(char* UPLO, int* N, int* NRHS, double* A, int* LDA, int* IPIV,
                 double* B, int* LDB, int* INFO);

    //*** SINGLE PRECISION COMPLEX ***//
    void cgesvd_(char* JOBU, char* JOBVT, int* M, int* N,
                 std::complex<float>* A, int* LDA,
                 float* S, std::complex<float>* U, int* LDU,
                 std::complex<float>* VT, int* LDVT,
                 std::complex<float>* WORK, int* LWORK, float* RWORK,
                 int* INFO);

    void cgetrf_(int* M, int* N, std::complex<float>* A, int* LDA,
                 int* IPIV, int* INFO);

    void cgetrs_(char* TRANS, int* N, int* NRHS,
                 std::complex<float>* A, int* LDA, int* IPIV,
                 std::complex<float>* B, int* LDB, int* INFO);

    //void csyevx_(char* JOBZ, char* RANGE, char* UPLO, int* N, std::complex<float>* A, int* LDA, std::complex<float>* VL, std::complex<float>* VU, int* IL, int* IU, std::complex<float>* ABSTOL, int* M, std::complex<float>* W, std::complex<float>* Z, int* LDZ, std::complex<float>* WORK, int* LWORK, int* IWORK, int* IFAIL, int* INFO);

    //void csygvx_(int* ITYPE, char* JOBZ, char* RANGE, char* UPLO, int* N, std::complex<float>* A, int* LDA, std::complex<float>* B, int* LDB, std::complex<float>* VL, std::complex<float>* VU, int* IL, int* IU, std::complex<float>* ABSTOL, int* M, std::complex<float>* W, std::complex<float>* Z__, int* LDZ, std::complex<float>* WORK, int* LWORK, int* IWORK, int* IFAIL, int* INFO);

    void csytrf_(char* UPLO, int* N, std::complex<float>* A, int* LDA,
                  int* IPIV, std::complex<float>* WORK, int* LWORK, int* INFO);

    void csytrs_(char* UPLO, int* N, int* NRHS,
                 std::complex<float>* A, int* LDA, int* IPIV,
                 std::complex<float>* B, int* LDB, int* INFO);

    //*** double PRECISION COMPLEX ***//
    void zgesvd_(char* JOBU, char* JOBVT, int* M, int* N,
                 std::complex<double>* A, int* LDA, double* S,
                 std::complex<double>* U, int* LDU,
                 std::complex<double>* VT, int* LDVT,
                 std::complex<double>* WORK, int* LWORK, double* RWORK,
                 int* INFO);

    void zgetrf_(int* M, int* N, std::complex<double>* A, int* LDA, int* IPIV,
                 int* INFO);

    void zgetrs_(char* TRANS, int* N, int* NRHS,
                 std::complex<double>* A, int* LDA, int* IPIV,
                 std::complex<double>* B, int* LDB, int* INFO);

    //void zsyevx_(char* JOBZ, char* RANGE, char* UPLO, int* N, std::complex<double>* A, int* LDA, std::complex<double>* VL, std::complex<double>* VU, int* IL, int* IU, std::complex<double>* ABSTOL, int* M, std::complex<double>* W, std::complex<double>* Z, int* LDZ, std::complex<double>* WORK, int* LWORK, int* IWORK, int* IFAIL, int* INFO);

    //void zsygvx_(int* ITYPE, char* JOBZ, char* RANGE, char* UPLO, int* N, std::complex<double>* A, int* LDA, std::complex<double>* B, int* LDB, std::complex<double>* VL, std::complex<double>* VU, int* IL, int* IU, std::complex<double>* ABSTOL, int* M, std::complex<double>* W, std::complex<double>* Z__, int* LDZ, std::complex<double>* WORK, int* LWORK, int* IWORK, int* IFAIL, int* INFO);

    void zsytrf_(char* UPLO, int* N, std::complex<double>* A, int* LDA,
                 int* IPIV, std::complex<double>* WORK, int* LWORK, int* INFO);

    void zsytrs_(char* UPLO, int* N, int* NRHS,
                 std::complex<double>* A, int* LDA, int* IPIV,
                 std::complex<double>* B, int* LDB, int* INFO);


} // extern "C"

}// namespace parelag
#endif /* PARELAG_LAPACK_HELPERS_HPP_ */
