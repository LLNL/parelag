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

#ifndef EIGENSOLVER_HPP_
#define EIGENSOLVER_HPP_

//!Wrapper for dsyvx / dsygvx
/**
 *  DSYEVX computes selected eigenvalues and, optionally, eigenvectors
 *  of a real symmetric matrix A.
 *
 *  DSYGVX computes selected eigenvalues, and optionally, eigenvectors
 *  of a real generalized symmetric-definite eigenproblem, of the form
 *  A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A
 *  and B are assumed to be symmetric and B is also positive definite.
 *
 *  Eigenvalues and eigenvectors can be selected by specifying either a
 *  range of values or a range of indices for the desired eigenvalues.
 */

class SymEigensolver
{
public:
	SymEigensolver();

	//! Compute all eigenvalues A x = lambda x
	int Compute(DenseMatrix & A, Vector & evals);
	//! Compute all eigenvectors and eigenvalues A x = lambda x
	int Compute(DenseMatrix & A, Vector & evals, DenseMatrix & evects);
	//! Compute all eigenvalues of the generalized eigenvalue problem A x = lambda B x
	int Compute(DenseMatrix & A, DenseMatrix & B, Vector & evals);
	//! Compute all eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda B x
	int Compute(DenseMatrix & A, DenseMatrix & B, Vector & evals, DenseMatrix & evects);
	//! Compute all eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d).
	int Compute(DenseMatrix & A, Vector & d, Vector & evals);
	//! Compute all eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d)
	int Compute(DenseMatrix & A, Vector & d, Vector & evals, DenseMatrix & evects);

	//! Compute the il-th to up-th eigenvalues of A x = lambda x
	int Compute(DenseMatrix & A, Vector & evals, int il, int iu);
	//! Compute the il-th to up-th eigenvectors and eigenvalues A x = lambda x
	int Compute(DenseMatrix & A, Vector & evals, DenseMatrix & evects, int il, int iu);
	//! Compute the il-th to up-th eigenvalues of the generalized eigenvalue problem A x = lambda B x
	int Compute(DenseMatrix & A, DenseMatrix & B, Vector & evals, int il, int iu);
	//! Compute the il-th to up-th eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda B x
	int Compute(DenseMatrix & A, DenseMatrix & B, Vector & evals, DenseMatrix & evects, int il, int iu);
	//! Compute the il-th to up-th eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d).
	int Compute(DenseMatrix & A, Vector & d, Vector & evals, int il, int iu);
	//! Compute the il-th to up-th eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d)
	int Compute(DenseMatrix & A, Vector & d, Vector & evals, DenseMatrix & evects, int il, int iu);

	//! Compute all eigenvalues A x = lambda x in the interval (vl, vu].
	int Compute(DenseMatrix & A, Vector & evals, double vl, double vu);
	//! Compute all eigenvectors and eigenvalues A x = lambda x in the interval (vl, vu].
	int Compute(DenseMatrix & A, Vector & evals, DenseMatrix & evects, double vl, double vu);
	//! Compute all eigenvalues of the generalized eigenvalue problem A x = lambda B x in the interval (vl, vu].
	int Compute(DenseMatrix & A, DenseMatrix & B, Vector & evals, double vl, double vu);
	//! Compute all eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda B x in the interval (vl, vu].
	int Compute(DenseMatrix & A, DenseMatrix & B, Vector & evals, DenseMatrix & evects, double vl, double vu);
	//! Compute all eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d) in the interval (vl, vu].
	int Compute(DenseMatrix & A, Vector & d, Vector & evals, double vl, double vu);
	//! Compute all eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d) in the interval (vl, vu].
	int Compute(DenseMatrix & A, Vector & d, Vector & evals, DenseMatrix & evects, double vl, double vu);

	//! Compute all eigenvalues A x = lambda x such that | lambda_i | \leq rel_tol |\lambda_\max|
	int ComputeSmallerMagnitude(DenseMatrix & A, Vector & evals, double rel_tol);
	//! Compute all eigenvectors and eigenvalues A x = lambda x such that | lambda_i | \leq rel_tol |\lambda_\max|
	int ComputeSmallerMagnitude(DenseMatrix & A, Vector & evals, DenseMatrix & evects, double rel_tol);
	//! Compute all eigenvalues of the generalized eigenvalue problem A x = lambda B x such that | lambda_i | \leq rel_tol |\lambda_\max|
	int ComputeSmallerMagnitude(DenseMatrix & A, DenseMatrix & B, Vector & evals, double rel_tol);
	//! Compute all eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda B x such that | lambda_i | \leq rel_tol |\lambda_\max|
	int ComputeSmallerMagnitude(DenseMatrix & A, DenseMatrix & B, Vector & evals, DenseMatrix & evects, double rel_tol);
	//! Compute all eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d) such that | lambda_i | \leq rel_tol |\lambda_\max|
	int ComputeSmallerMagnitude(DenseMatrix & A, Vector & d, Vector & evals, double rel_tol);
	//! Compute all eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d) such that | lambda_i | \leq rel_tol |\lambda_\max|
	int ComputeSmallerMagnitude(DenseMatrix & A, Vector & d, Vector & evals, DenseMatrix & evects, double rel_tol);

	void AllocateWorkspace(int n);

	void SetGeneralizedEigenProblem(int itype_){ itype = itype_;}

	virtual ~SymEigensolver();

	int doNotOverwrite;
private:
	/**
	 * Specifies the problem type to be solved (dsygvx):
	 * = 1:  A*x = (lambda)*B*x
	 * = 2:  A*B*x = (lambda)*x
	 * = 3:  B*A*x = (lambda)*x
	 */
	int	itype;
	/**
	 *  = 'N':  Compute eigenvalues only;
	 *  = 'V':  Compute eigenvalues and eigenvectors.
	 */
	char jobz;
	/**
	 * = 'A': all eigenvalues will be found.
	 * = 'V': all eigenvalues in the half-open interval (VL,VU]
	 *        will be found.
	 * = 'I': the IL-th through IU-th eigenvalues will be found.
	 */
	char range;
	/**
	 * 	= 'U':  Upper triangle of A and B are stored;
	 *  = 'L':  Lower triangle of A and B are stored.
	 */
	char uplo;
	/**
	 * 	  A is DOUBLE PRECISION array, dimension (LDA, N)
	 *    On entry, the symmetric matrix A.  If UPLO = 'U', the
	 *    leading N-by-N upper triangular part of A contains the
	 *    upper triangular part of the matrix A.  If UPLO = 'L',
	 *    the leading N-by-N lower triangular part of A contains
	 *    the lower triangular part of the matrix A.
	 *
	 *    On exit, the lower triangle (if UPLO='L') or the upper
	 *    triangle (if UPLO='U') of A, including the diagonal, is
	 *    destroyed.
	 */
	Array<double> A;
	/**
	 *  B is DOUBLE PRECISION array, dimension (LDB, N)
	 *  On entry, the symmetric matrix B.  If UPLO = 'U', the
	 *  leading N-by-N upper triangular part of B contains the
	 *  upper triangular part of the matrix B.  If UPLO = 'L',
	 *  the leading N-by-N lower triangular part of B contains
	 *  the lower triangular part of the matrix B.
	 *
	 *  On exit, if INFO <= N, the part of B containing the matrix is
	 *  overwritten by the triangular factor U or L from the Cholesky
	 *  factorization B = U**T*U or B = L*L**T.
	 */
	Array<double> B;
    /**
     *  The absolute error tolerance for the eigenvalues.
	 *  An approximate eigenvalue is accepted as converged
	 *  when it is determined to lie in an interval [a,b]
	 *  of width less than or equal to
	 *
	 *         ABSTOL + EPS *   max( |a|,|b| ) ,
	 *
	 *  where EPS is the machine precision.  If ABSTOL is less than
	 *  or equal to zero, then  EPS*|T|  will be used in its place,
	 *  where |T| is the 1-norm of the tridiagonal matrix obtained
	 *  by reducing C to tridiagonal form, where C is the symmetric
	 *  matrix of the standard symmetric problem to which the
	 *  generalized problem is transformed.
	 *
	 *  Eigenvalues will be computed most accurately when ABSTOL is
	 *  set to twice the underflow threshold 2*DLAMCH('S'), not zero.
	 *  If this routine returns with INFO>0, indicating that some
	 *  eigenvectors did not converge, try setting ABSTOL to
	 *  2*DLAMCH('S').
     */
	double abstol;
	/**
	 * The total number of eigenvalues found.  0 <= M <= N.
	 * If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
	 */
	int m;
	/**
	 * 	w is DOUBLE PRECISION array, dimension (N)
	 *    On normal exit, the first M elements contain the selected
	 *   eigenvalues in ascending order.
	 */
	Vector w;
	/*
	 * Z is DOUBLE PRECISION array, dimension (LDZ, max(1,M))
	 * If JOBZ = 'N', then Z is not referenced.
	 * If JOBZ = 'V', then if INFO = 0, the first M columns of Z
	 * contain the orthonormal eigenvectors of the matrix A
	 * corresponding to the selected eigenvalues, with the i-th
	 * column of Z holding the eigenvector associated with W(i).
	 * The eigenvectors are normalized as follows:
	 * if ITYPE = 1 or 2, Z**T*B*Z = I;
	 * if ITYPE = 3, Z**T*inv(B)*Z = I.
     *
	 * If an eigenvector fails to converge, then that column of Z
	 * contains the latest approximation to the eigenvector, and the
	 * index of the eigenvector is returned in IFAIL.
	 * Note: the user must ensure that at least max(1,M) columns are
	 * supplied in the array Z; if RANGE = 'V', the exact value of M
	 * is not known in advance and an upper bound must be used.
	 */
	Array<double> Z;
	/**
	 * The leading dimension of the array Z.  LDZ >= 1, and if
	 *  JOBZ = 'V', LDZ >= max(1,N).
	 */
	int ldz;
	/**
	 * 	WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
	 *  On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
	 */
	Array<double> work;
	/**
     * The length of the array WORK.  LWORK >= max(1,8*N).
     * For optimal efficiency, LWORK >= (NB+3)*N,
     * where NB is the blocksize for DSYTRD returned by ILAENV.
     *
     * If LWORK = -1, then a workspace query is assumed; the routine
     * only calculates the optimal size of the WORK array, returns
     * this value as the first entry of the WORK array, and no error
     * message related to LWORK is issued by XERBLA.
     */
	int lwork;
	//! IWORK is INTEGER array, dimension (5*N)
	Array<int> iwork;
	/**
	 * 	IFAIL is INTEGER array, dimension (N)
	 *   If JOBZ = 'V', then if INFO = 0, the first M elements of
	 *   IFAIL are zero.  If INFO > 0, then IFAIL contains the
	 *   indices of the eigenvectors that failed to converge.
	 *   If JOBZ = 'N', then IFAIL is not referenced.
	 */
	Array<int> ifail;
	/**
	 * = 0:  successful exit
	 * < 0:  if INFO = -i, the i-th argument had an illegal value
	 * > 0:  DPOTRF or DSYEVX returned an error code:
	 *       <= N:  if INFO = i, DSYEVX failed to converge;
	 *              i eigenvectors failed to converge.  Their indices
	 *              are stored in array IFAIL.
	 * > N:   if INFO = N + i, for 1 <= i <= N, then the leading
	 *        minor of order i of B is not positive definite.
	 *        The factorization of B could not be completed and
	 *        no eigenvalues or eigenvectors were computed.
	 */
	int info;

	int n_max;

};


template<class FUNC>
void matrixFunction(DenseMatrix & A, DenseMatrix & fA, FUNC & f)
{
	elag_assert(A.Height() == A.Width() );
	SymEigensolver eigs;
	eigs.doNotOverwrite = 1;
	int n = A.Height();
	Vector evals(n);
	DenseMatrix evects(n);
	eigs.Compute(A, evals, evects);

	std::transform(evals.GetData(), evals.GetData()+n, evals.GetData(), f);
	fA.SetSize(n);

	MultADAt(evects, evals, fA);

}

#endif /* EIGENSOLVER_HPP_ */
