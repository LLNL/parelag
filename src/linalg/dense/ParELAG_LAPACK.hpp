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

#ifndef PARELAG_LAPACK_HPP_
#define PARELAG_LAPACK_HPP_

#include <complex>

#include "ParELAG_LAPACK_Helpers.hpp"
#include "utilities/ParELAG_DataTraits.hpp"

namespace parelag
{

/** \class ReturnErrorCode
 *  \brief Handle errors by returning the error code.
 */
struct ReturnErrorCode
{
    template <typename ErrorCodeT>
    static ErrorCodeT HandleErrorCode(ErrorCodeT&& ErrCode)
    {
        return std::forward<ErrorCodeT>(ErrCode);
    }
};

/** \class ExceptionOnNonzeroError
 *  \brief Throw a special exeception if error code is not zero.
 */
struct ExceptionOnNonzeroError
{
    /** \class ErrorCodeIsZero
     *  \brief Exception that is thrown when a nonzero error code is
     *         encountered.
     */
    struct ErrorCodeIsZero : std::runtime_error
    {
        template <typename... Ts>
        ErrorCodeIsZero(Ts&&... args)
            : std::runtime_error(std::forward<Ts>(args)...)
        {}
    };

    /** \brief Test return code and throw if zero.
     *
     *  \throws ErrorCodeIsZero Input error code is numerically zero.
     */
    template <typename ErrorCodeT>
    static ErrorCodeT HandleErrorCode(ErrorCodeT&& ErrCode)
    {
        using ErrorCodeNoRefT =
            typename std::remove_reference<ErrorCodeT>::type;
        PARELAG_TEST_FOR_EXCEPTION(
            ErrCode != static_cast<ErrorCodeNoRefT>(0),
            ErrorCodeIsZero,
            "Encountered nonzero error! ErrCode = " << ErrCode << ".");

        return ErrCode;
    }
};


/** \class Lapack
 *  \brief A templated wrapper for LAPACK.
 *
 *  As opposed to the standard LAPACK functions, these handle the INFO
 *  variable according to the error policy.
 *
 *  This is a class with only static functions and no state (generally
 *  a dumb idea) because C++ does not allow templated namespaces.
 *
 *  Variables passed by pointer are arrays; by value are scalars.
 */
template <typename ValueT, typename ErrorPolicy = ReturnErrorCode>
struct Lapack;


/** \class Lapack<float,ErrorPolicy>
 *  \brief A templated wrapper for LAPACK, single-precision specialization.
 *
 *  As opposed to the standard LAPACK functions, these handle the INFO
 *  variable according to the error policy.
 *
 *  This is a class with only static functions and no state (generally
 *  a dumb idea) because C++ does not allow templated namespaces.
 *
 *  Variables passed by pointer are arrays; by value are scalars.
 */
template <typename ErrorPolicy>
struct Lapack<float,ErrorPolicy>
{
    using value_type = float;
    using real_type = typename DataTraits<value_type>::real_type;
    using index_type = int;
    using size_type = int;

    /** \brief Computes the SVD of a real MxN matrix A and,
     *         optionally, the left and/or right singular vectors.
     */
    static int GESVD(char JOBU, char JOBVT, size_type M, size_type N,
                     value_type* A, size_type LDA, real_type* S,
                     value_type* U, size_type LDU, value_type* VT,
                     size_type LDVT, value_type* WORK, size_type LWORK)
    {
        int INFO;
        sgesvd_(&JOBU,&JOBVT,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,WORK,&LWORK,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Compute LU factorization of a general MxN matrix. */
    static int GETRF(size_type M, size_type N, value_type* A, size_type LDA,
                     index_type* IPIV)
    {
        int INFO;
        sgetrf_(&M,&N,A,&LDA,IPIV,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Solve the system of linear equations Ax = B, A^Tx = B,
     *         or A^Hx = B.
     */
    static int GETRS(char TRANS, size_type N, size_type NRHS,
                     value_type* A, size_type LDA,
                     index_type* IPIV, value_type* B, size_type LDB)
    {
        int INFO;
        sgetrs_(&TRANS,&N,&NRHS,A,&LDA,IPIV,B,&LDB,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Computes seleced eigenvalues and, optionally, the left
     *         and/or right eigenvectors for SY matrices.
     */
    static std::pair<int,size_type> SYEVX(
        char JOBZ, char RANGE, char UPLO, size_type N,
        value_type* A, size_type LDA,
        value_type VL, value_type VU, index_type IL, index_type IU,
        value_type ABSTOL, value_type* W,
        value_type* Z, size_type LDZ, value_type* WORK, size_type LWORK,
        index_type* IWORK, index_type* IFAIL)
    {
        int INFO;
        size_type M;
        ssyevx_(&JOBZ,&RANGE,&UPLO,&N,A,&LDA,&VL,&VU,&IL,&IU,&ABSTOL,&M,
                W,Z,&LDZ,WORK,&LWORK,IWORK,IFAIL,&INFO);
        return std::make_pair(ErrorPolicy::HandleErrorCode(INFO),M);
    }

    /** \brief Computes selected eigenvalues and, optionally,
     *         eigenvectors of a real generalized symmetric-definite
     *         eigenproblem, of the form Ax = (lambda)Bx, ABx =
     *         (lambda)x, or BAx = (lambda)x.
     */
    static std::pair<int,size_type> SYGVX(
        int ITYPE, char JOBZ, char RANGE, char UPLO, size_type N,
        value_type* A, size_type LDA, value_type* B, size_type LDB,
        value_type VL, value_type VU, index_type IL, index_type IU,
        real_type ABSTOL, value_type* W,
        value_type* Z, size_type LDZ, value_type* WORK, size_type LWORK,
        index_type* IWORK, index_type* IFAIL)
    {
        int INFO;
        size_type M;
        ssygvx_(&ITYPE,&JOBZ,&RANGE,&UPLO,&N,A,&LDA,B,&LDB,&VL,&VU,&IL,&IU,
                &ABSTOL,&M,W,Z,&LDZ,WORK,&LWORK,IWORK,IFAIL,&INFO);
        return std::make_pair(ErrorPolicy::HandleErrorCode(INFO),M);
    }

    /** \brief Computes the factorization of a real symmetric matrix A
     *         using the Bunch-Kaufman diagonal pivoting method.
     */
    static int SYTRF(char UPLO, size_type N, value_type* A, size_type LDA,
                     index_type* IPIV, value_type* WORK, size_type LWORK)
    {
        size_type INFO;
        ssytrf_(&UPLO,&N,A,&LDA,IPIV,WORK,&LWORK,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Solves a system of linear equations AX = B with a real
     *         symmetric matrix A using the factorization A = UDU^T or
     *         A = LDL^T computed by DSYTRF.
     */
    static int SYTRS(char UPLO, size_type N, size_type NRHS,
                     value_type* A, size_type LDA,
                     index_type* IPIV, value_type* B, size_type LDB)
    {
        size_type INFO;
        ssytrs_(&UPLO,&N,&NRHS,A,&LDA,IPIV,B,&LDB,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }
};// struct Lapack<float,ErrorPolicy>


/** \class Lapack<double,ErrorPolicy>
 *  \brief A templated wrapper for LAPACK, double-precision specialization.
 *
 *  As opposed to the standard LAPACK functions, these handle the INFO
 *  variable according to the error policy.
 *
 *  This is a class with only static functions and no state (generally
 *  a dumb idea) because C++ does not allow templated namespaces.
 *
 *  Variables passed by pointer are arrays; by value are scalars.
 */
template <typename ErrorPolicy>
struct Lapack<double,ErrorPolicy>
{
    using value_type = double;
    using real_type = typename DataTraits<value_type>::real_type;
    using index_type = int;
    using size_type = int;

    /** \brief Computes the SVD of a real MxN matrix A and,
     *         optionally, the left and/or right singular vectors.
     */
    static int GESVD(char JOBU, char JOBVT, size_type M, size_type N,
                     value_type* A, size_type LDA, real_type* S,
                     value_type* U, size_type LDU, value_type* VT,
                     size_type LDVT, value_type* WORK, size_type LWORK)
    {
        int INFO;
        dgesvd_(&JOBU,&JOBVT,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,WORK,&LWORK,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Compute LU factorization of a general MxN matrix. */
    static int GETRF(size_type M, size_type N, value_type* A, size_type LDA,
                     index_type* IPIV)
    {
        int INFO;
        dgetrf_(&M,&N,A,&LDA,IPIV,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Solve the system of linear equations Ax = B, A^Tx = B,
     *         or A^Hx = B.
     */
    static int GETRS(char TRANS, size_type N, size_type NRHS,
                     value_type* A, size_type LDA,
                     index_type* IPIV, value_type* B, size_type LDB)
    {
        int INFO;
        dgetrs_(&TRANS,&N,&NRHS,A,&LDA,IPIV,B,&LDB,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Computes seleced eigenvalues and, optionally, the left
     *         and/or right eigenvectors for SY matrices.
     */
    static std::pair<int,size_type> SYEVX(
        char JOBZ, char RANGE, char UPLO, size_type N,
        value_type* A, size_type LDA,
        value_type VL, value_type VU, index_type IL, index_type IU,
        value_type ABSTOL, value_type* W,
        value_type* Z, size_type LDZ, value_type* WORK, size_type LWORK,
        index_type* IWORK, index_type* IFAIL)
    {
        int INFO;
        size_type M;
        dsyevx_(&JOBZ,&RANGE,&UPLO,&N,A,&LDA,&VL,&VU,&IL,&IU,&ABSTOL,&M,
                W,Z,&LDZ,WORK,&LWORK,IWORK,IFAIL,&INFO);
        return std::make_pair(ErrorPolicy::HandleErrorCode(INFO),M);
    }

    /** \brief Computes selected eigenvalues and, optionally,
     *         eigenvectors of a real generalized symmetric-definite
     *         eigenproblem, of the form Ax = (lambda)Bx, ABx =
     *         (lambda)x, or BAx = (lambda)x.
     */
    static std::pair<int,size_type> SYGVX(
        int ITYPE, char JOBZ, char RANGE, char UPLO, size_type N,
        value_type* A, size_type LDA, value_type* B, size_type LDB,
        value_type VL, value_type VU, index_type IL, index_type IU,
        real_type ABSTOL, value_type* W,
        value_type* Z, size_type LDZ, value_type* WORK, size_type LWORK,
        index_type* IWORK, index_type* IFAIL)
    {
        int INFO;
        size_type M;
        dsygvx_(&ITYPE,&JOBZ,&RANGE,&UPLO,&N,A,&LDA,B,&LDB,&VL,&VU,&IL,&IU,
                &ABSTOL,&M,W,Z,&LDZ,WORK,&LWORK,IWORK,IFAIL,&INFO);
        return std::make_pair(ErrorPolicy::HandleErrorCode(INFO),M);
    }

    /** \brief Computes the factorization of a real symmetric matrix A
     *         using the Bunch-Kaufman diagonal pivoting method.
     */
    static int SYTRF(
        char UPLO, size_type N, value_type* A, size_type LDA,
        index_type* IPIV, value_type* WORK, size_type LWORK)
    {
        int INFO;
        dsytrf_(&UPLO,&N,A,&LDA,IPIV,WORK,&LWORK,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Solves a system of linear equations AX = B with a real
     *         symmetric matrix A using the factorization A = UDU^T or
     *         A = LDL^T computed by DSYTRF.
     */
    static int SYTRS(char UPLO, size_type N, size_type NRHS,
                     value_type* A, size_type LDA,
                     index_type* IPIV, value_type* B, size_type LDB)
    {
        int INFO;
        dsytrs_(&UPLO,&N,&NRHS,A,&LDA,IPIV,B,&LDB,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }
};// struct Lapack<double,ErrorPolicy>


/** \class Lapack<double,ErrorPolicy>
 *  \brief A templated wrapper for LAPACK, single-precision complex
 *      specialization.
 *
 *  As opposed to the standard LAPACK functions, these handle the INFO
 *  variable according to the error policy.
 *
 *  This is a class with only static functions and no state (generally
 *  a dumb idea) because C++ does not allow templated namespaces.
 *
 *  Variables passed by pointer are arrays; by value are scalars.
 */
template <typename ErrorPolicy>
struct Lapack<std::complex<float>,ErrorPolicy>
{
    using value_type = std::complex<float>;
    using real_type = typename DataTraits<value_type>::real_type;
    using index_type = int;
    using size_type = int;

    /** \brief Computes the SVD of a real MxN matrix A and,
     *         optionally, the left and/or right singular vectors.
     */
    static int GESVD(char JOBU, char JOBVT, size_type M, size_type N,
                     value_type* A, size_type LDA, real_type* S,
                     value_type* U, size_type LDU, value_type* VT,
                     size_type LDVT, value_type* WORK, size_type LWORK,
                     real_type* RWORK)
    {
        int INFO;
        cgesvd_(&JOBU,&JOBVT,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,
                WORK,&LWORK,RWORK,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Compute LU factorization of a general MxN matrix. */
    static int GETRF(size_type M, size_type N, value_type* A,
                     size_type LDA, index_type* IPIV)
    {
        int INFO;
        cgetrf_(&M,&N,A,&LDA,IPIV,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Solve the system of linear equations Ax = B, A^Tx = B,
     *         or A^Hx = B.
     */
    static int GETRS(
        char TRANS, size_type N, size_type NRHS, value_type* A, size_type LDA,
        index_type* IPIV, value_type* B, size_type LDB)
    {
        int INFO;
        cgetrs_(&TRANS,&N,&NRHS,A,&LDA,IPIV,B,&LDB,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Computes the factorization of a real symmetric matrix A
     *         using the Bunch-Kaufman diagonal pivoting method.
     */
    static int SYTRF(
        char UPLO, size_type N, value_type* A, size_type LDA, index_type* IPIV,
        value_type* WORK, size_type LWORK)
    {
        int INFO;
        csytrf_(&UPLO,&N,A,&LDA,IPIV,WORK,&LWORK,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Solves a system of linear equations AX = B with a real
     *         symmetric matrix A using the factorization A = UDU^T or
     *         A = LDL^T computed by DSYTRF.
     */
    static int SYTRS(
        char UPLO, size_type N, size_type NRHS, value_type* A, size_type LDA,
        index_type* IPIV, value_type* B, size_type LDB)
    {
        int INFO;
        csytrs_(&UPLO,&N,&NRHS,A,&LDA,IPIV,B,&LDB,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }
};// class Lapack<std::complex<float>,ErrorPolicy>


/** \class Lapack<std::complex<double>,ErrorPolicy>
 *  \brief A templated wrapper for LAPACK, double-precision complex
 *      specialization.
 *
 *  As opposed to the standard LAPACK functions, these handle the INFO
 *  variable according to the error policy.
 *
 *  This is a class with only static functions and no state (generally
 *  a dumb idea) because C++ does not allow templated namespaces.
 *
 *  Variables passed by pointer are arrays; by value are scalars.
 */
template <typename ErrorPolicy>
struct Lapack<std::complex<double>,ErrorPolicy>
{
    using value_type = std::complex<double>;
    using real_type = typename DataTraits<value_type>::real_type;
    using index_type = int;
    using size_type = int;

    /** \brief Computes the SVD of a real MxN matrix A and,
     *         optionally, the left and/or right singular vectors.
     */
    static int GESVD(
        char JOBU, char JOBVT, size_type M, size_type N,
        value_type* A, size_type LDA, real_type* S,
        value_type* U, size_type LDU, value_type* VT, size_type LDVT,
        value_type* WORK, size_type LWORK, real_type* RWORK)
    {
        int INFO;
        zgesvd_(&JOBU,&JOBVT,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,
                WORK,&LWORK,RWORK,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Compute LU factorization of a general MxN matrix. */
    static int GETRF(size_type M, size_type N, value_type* A, size_type LDA,
                     index_type* IPIV)
    {
        int INFO;
        zgetrf_(&M,&N,A,&LDA,IPIV,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Solve the system of linear equations Ax = B, A^Tx = B,
     *         or A^Hx = B.
     */
    static int GETRS(
        char TRANS, size_type N, size_type NRHS, value_type* A, size_type LDA,
        index_type* IPIV, value_type* B, size_type LDB)
    {
        int INFO;
        zgetrs_(&TRANS,&N,&NRHS,A,&LDA,IPIV,B,&LDB,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Computes the factorization of a real symmetric matrix A
     *         using the Bunch-Kaufman diagonal pivoting method.
     */
    static int SYTRF(
        char UPLO, size_type N, value_type* A, size_type LDA,
        index_type* IPIV, value_type* WORK, size_type LWORK)
    {
        int INFO;
        zsytrf_(&UPLO,&N,A,&LDA,IPIV,WORK,&LWORK,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }

    /** \brief Solves a system of linear equations AX = B with a real
     *         symmetric matrix A using the factorization A = UDU^T or
     *         A = LDL^T computed by DSYTRF.
     */
    static int SYTRS(
        char UPLO, size_type N, size_type NRHS, value_type* A, size_type LDA,
        index_type* IPIV, value_type* B, size_type LDB)
    {
        int INFO;
        zsytrs_(&UPLO,&N,&NRHS,A,&LDA,IPIV,B,&LDB,&INFO);
        return ErrorPolicy::HandleErrorCode(INFO);
    }
};// class Lapack<std::complex<double>,ErrorPolicy>

}// namespace parelag
#endif /* PARELAG_LAPACK_HPP_ */
