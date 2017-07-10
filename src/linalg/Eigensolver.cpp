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

#include "elag_linalg.hpp"


extern "C"
{
double dlamch_(char *cmach);

void dsyevx_( const char* jobz, const char* range, const char* uplo, int* n, double* a, int* lda, double* vl,
		double* vu, int* il, int* iu, double* abstol, int* m, double* w, double* z, int* ldz, double* work, int* lwork,
		int* iwork, int* ifail, int* info );

int dsygvx_(int *itype, char *jobz, char *range, char *
	uplo, int *n, double *a, int *lda, double *b, int
	*ldb, double *vl, double *vu, int *il, int *iu,
	double *abstol, int *m, double *w, double *z__,
	int *ldz, double *work, int *lwork, int *iwork,
	int *ifail, int *info);
}


SymEigensolver::SymEigensolver():
	doNotOverwrite(0),
	itype(1),
	jobz('V'),
	range('A'),
	uplo('U'),
	A(0),
    B(0),
	abstol( 0 ),
	m(0),
	w(0),
    Z(0),
	ldz(0),
    work(0),
    lwork(0),
	iwork(0),
    ifail(0),
    info(0),
    n_max(-1)
{
	char c = 'S';
	abstol = 2.*dlamch_(&c);
}

SymEigensolver::~SymEigensolver()
{

}

//! Compute all eigenvalues A x = lambda x
int SymEigensolver::Compute(DenseMatrix & A_, Vector & evals)
{
	int n = A_.Size();

	if(n_max < n)
		AllocateWorkspace(n);

	jobz  = 'N';
	range = 'A';
	uplo  = 'U';

	double * a = NULL;

	if(doNotOverwrite)
	{
		a = A.GetData();
		int nnz = n*n;
		elag_assert(nnz <= A.Size());
		std::copy(A_.Data(), A_.Data()+nnz, a);
	}
	else
	{
		a = A_.Data();
	}

	evals.SetSize(n);
	dsyevx_( &jobz, &range, &uplo, &n, a, &n, NULL, NULL, NULL, NULL,
			&abstol, &m, evals.GetData(), NULL, &n, work.GetData(), &lwork,
			iwork.GetData(), ifail.GetData(), &info );

	return info;
}

// Compute all eigenvectors and eigenvalues A x = lambda x
int SymEigensolver::Compute(DenseMatrix & A_, Vector & evals, DenseMatrix & evects)
{
	int n = A_.Size();

	if(n_max < n)
		AllocateWorkspace(n);

	jobz  = 'V';
	range = 'A';
	uplo  = 'U';

	double * a = NULL;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
	}
	else
	{
		a = A_.Data();
	}

	evals.SetSize(n);
	evects.SetSize(n,n);

	dsyevx_( &jobz, &range, &uplo, &n, a, &n, NULL, NULL, NULL, NULL,
			&abstol, &m, evals.GetData(), evects.Data(), &n, work.GetData(), &lwork,
			iwork.GetData(), ifail.GetData(), &info );

	return info;
}
// Compute all eigenvalues of the generalized eigenvalue problem A x = lambda B x
int SymEigensolver::Compute(DenseMatrix & A_, DenseMatrix & B_, Vector & evals)
{
	int n = A_.Size();

	if(n_max < n)
		AllocateWorkspace(n);

	jobz  = 'N';
	range = 'A';
	uplo  = 'U';

	double * a = NULL, * b = NULL;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
		b = B.GetData();
		std::copy(B_.Data(), B_.Data()+n*n, b);
	}
	else
	{
		a = A_.Data();
		b = B_.Data();
	}

	evals.SetSize(n);

	dsygvx_( &itype, &jobz, &range, &uplo, &n, a, &n, b, &n, NULL, NULL, NULL, NULL,
			&abstol, &m, evals.GetData(), NULL, &n, work.GetData(), &lwork,
			iwork.GetData(), ifail.GetData(), &info );

	return info;
}

// Compute all eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda B x
int SymEigensolver::Compute(DenseMatrix & A_, DenseMatrix & B_, Vector & evals, DenseMatrix & evects)
{
	int n = A_.Size();

	if(n_max < n)
		AllocateWorkspace(n);

	jobz  = 'V';
	range = 'A';
	uplo  = 'U';

	double * a = NULL, * b = NULL;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
		b = B.GetData();
		std::copy(B_.Data(), B_.Data()+n*n, b);
	}
	else
	{
		a = A_.Data();
		b = B_.Data();
	}

	evals.SetSize(n);
	evects.SetSize(n,n);

	dsygvx_( &itype, &jobz, &range, &uplo, &n, a, &n, b, &n, NULL, NULL, NULL, NULL,
			&abstol, &m, evals.GetData(), evects.Data(), &n, work.GetData(), &lwork,
			iwork.GetData(), ifail.GetData(), &info );

	return info;
}
// Compute all eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d).
int SymEigensolver::Compute(DenseMatrix & A_, Vector & d, Vector & evals)
{
	if(itype != 1)
		mfem_error("Only A x = lambda D x is supported \n");

	int n = A_.Size();
	double * a = NULL;

	if(n_max < n)
		AllocateWorkspace(n);

	int oldOpt = doNotOverwrite;

	if(doNotOverwrite)
	{
		a = A.GetData();
		int nnz = n*n;
		std::copy(A_.Data(), A_.Data()+nnz, a);
		doNotOverwrite = 0;
	}
	else
	{
		a = A_.Data();
	}

	DenseMatrix sA(a,n,n);
	sA.InvSymmetricScaling(d);


	int r;
	r = Compute(sA, evals);

	sA.ClearExternalData();
	doNotOverwrite = oldOpt;

	return r;
}
//Compute all eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d)
int SymEigensolver::Compute(DenseMatrix & A_, Vector & d, Vector & evals, DenseMatrix & evects)
{
	if(itype != 1)
		mfem_error("Only A x = lambda D x is supported \n");

	int n = A_.Size();
	double * a = NULL;

	if(n_max < n)
		AllocateWorkspace(n);

	int oldOpt = doNotOverwrite;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
		doNotOverwrite = 0;
	}
	else
	{
		a = A_.Data();
	}

	DenseMatrix sA(a,n,n);
	sA.InvSymmetricScaling(d);


	int r;
	r = Compute(sA, evals, evects);

	Vector sqrtd(n);
	for(double * out = sqrtd.GetData(), * in = d.GetData(); in != d.GetData()+n; ++out, ++in)
		*out = sqrt(*in);

	evects.InvLeftScaling(sqrtd);

	sA.ClearExternalData();
	doNotOverwrite = oldOpt;

	return r;
}

// Compute the il-th to up-th eigenvalues of A x = lambda x
int SymEigensolver::Compute(DenseMatrix & A_, Vector & evals, int il, int iu)
{
	if(il > iu)
	{
		evals.SetSize(0);
		return 0;
	}

	int n = A_.Size();

	if(n_max < n)
		AllocateWorkspace(n);

	jobz  = 'N';
	range = 'I';
	uplo  = 'U';

	double * a = NULL;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
	}
	else
	{
		a = A_.Data();
	}


	dsyevx_( &jobz, &range, &uplo, &n, a, &n, NULL, NULL, &il, &iu,
			&abstol, &m, w.GetData(), NULL, &n, work.GetData(), &lwork,
			iwork.GetData(), ifail.GetData(), &info );

	if(m != iu- il + 1)
		mfem_error("SymEigensolver::Compute(DenseMatrix & A_, Vector & evals, int il, int iu)");

	evals.SetSize(iu-il+1);
	std::copy(w.GetData(), w.GetData()+m, evals.GetData());

	return info;
}
// Compute the il-th to up-th eigenvectors and eigenvalues A x = lambda x
int SymEigensolver::Compute(DenseMatrix & A_, Vector & evals, DenseMatrix & evects, int il, int iu)
{
	if(il > iu)
	{
		evals.SetSize(0);
		return 0;
	}

	int n = A_.Size();

	if(n_max < n)
		AllocateWorkspace(n);

	jobz  = 'V';
	range = 'I';
	uplo  = 'U';

	double * a = NULL;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
	}
	else
	{
		a = A_.Data();
	}

	evects.SetSize(n, iu-il+1);
	dsyevx_( &jobz, &range, &uplo, &n, a, &n, NULL, NULL, &il, &iu,
			&abstol, &m, w.GetData(), evects.Data(), &n, work.GetData(), &lwork,
			iwork.GetData(), ifail.GetData(), &info );

	if(m != iu- il + 1)
		mfem_error("SymEigensolver::Compute(DenseMatrix & A_, Vector & evals, int il, int iu)");

	evals.SetSize(iu-il+1);
	std::copy(w.GetData(), w.GetData()+m, evals.GetData());

	return info;
}
// Compute the il-th to up-th eigenvalues of the generalized eigenvalue problem A x = lambda B x
int SymEigensolver::Compute(DenseMatrix & A_, DenseMatrix & B_, Vector & evals, int il, int iu)
{
	if(il > iu)
	{
		evals.SetSize(0);
		return 0;
	}

	int n = A_.Size();

	if(n_max < n)
		AllocateWorkspace(n);

	jobz  = 'N';
	range = 'I';
	uplo  = 'U';

	double * a = NULL, * b = NULL;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
		b = B.GetData();
		std::copy(B_.Data(), B_.Data()+n*n, b);
	}
	else
	{
		a = A_.Data();
		b = B_.Data();
	}

	dsygvx_( &itype, &jobz, &range, &uplo, &n, a, &n, b, &n, NULL, NULL, &il, &iu,
			&abstol, &m, w.GetData(), NULL, &n, work.GetData(), &lwork,
			iwork.GetData(), ifail.GetData(), &info );

	if(m != iu- il + 1)
		mfem_error("SymEigensolver::Compute(DenseMatrix & A_, Vector & evals, int il, int iu)");

	evals.SetSize(iu-il+1);
	std::copy(w.GetData(), w.GetData()+m, evals.GetData());

	return info;
}
// Compute the il-th to up-th eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda B x
int SymEigensolver::Compute(DenseMatrix & A_, DenseMatrix & B_, Vector & evals, DenseMatrix & evects, int il, int iu)
{
	if(il > iu)
	{
		evals.SetSize(0);
		return 0;
	}

	int n = A_.Size();

	if(n_max < n)
		AllocateWorkspace(n);

	jobz  = 'V';
	range = 'I';
	uplo  = 'U';

	double * a = NULL, * b = NULL;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
		b = B.GetData();
		std::copy(B_.Data(), B_.Data()+n*n, b);
	}
	else
	{
		a = A_.Data();
		b = B_.Data();
	}

	evects.SetSize(n, iu-il+1);
	dsygvx_( &itype, &jobz, &range, &uplo, &n, a, &n, b, &n, NULL, NULL, &il, &iu,
			&abstol, &m, w.GetData(), evects.Data(), &n, work.GetData(), &lwork,
			iwork.GetData(), ifail.GetData(), &info );

	if(m != iu- il + 1)
		mfem_error("SymEigensolver::Compute(DenseMatrix & A_, Vector & evals, int il, int iu)");

	evals.SetSize(iu-il+1);
	std::copy(w.GetData(), w.GetData()+m, evals.GetData());

	return info;
}

// Compute the il-th to up-th eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d).
int SymEigensolver::Compute(DenseMatrix & A_, Vector & d, Vector & evals, int il, int iu)
{
	if(itype != 1)
		mfem_error("Only A x = lambda D x is supported \n");

	int n = A_.Size();
	double * a = NULL;

	if(n_max < n)
		AllocateWorkspace(n);

	int oldOpt = doNotOverwrite;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
		doNotOverwrite = 0;
	}
	else
	{
		a = A_.Data();
	}

	DenseMatrix sA(a,n,n);
	sA.InvSymmetricScaling(d);


	int r;
	r = Compute(sA, evals,il,iu);

	sA.ClearExternalData();
	doNotOverwrite = oldOpt;

	return r;
}

// Compute the il-th to up-th eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d)
int SymEigensolver::Compute(DenseMatrix & A_, Vector & d, Vector & evals, DenseMatrix & evects, int il, int iu)
{
	if(itype != 1)
		mfem_error("Only A x = lambda D x is supported \n");

	int n = A_.Size();
	double * a = NULL;

	if(n_max < n)
		AllocateWorkspace(n);

	int oldOpt = doNotOverwrite;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
		doNotOverwrite = 0;
	}
	else
	{
		a = A_.Data();
	}

	DenseMatrix sA(a,n,n);
	sA.InvSymmetricScaling(d);


	int r;
	r = Compute(sA, evals, evects, il,iu);

	Vector sqrtd(n);
	for(double * out = sqrtd.GetData(), * in = d.GetData(); in != d.GetData()+n; ++out, ++in)
		*out = sqrt(*in);

	evects.InvLeftScaling(sqrtd);

	sA.ClearExternalData();
	doNotOverwrite = oldOpt;

	return r;
}


int SymEigensolver::Compute(DenseMatrix & A_, Vector & evals, double vl, double vu)
{
	int n = A_.Size();

	if(n_max < n)
		AllocateWorkspace(n);

	jobz  = 'N';
	range = 'V';
	uplo  = 'U';

	double * a = NULL;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
	}
	else
	{
		a = A_.Data();
	}


	dsyevx_( &jobz, &range, &uplo, &n, a, &n, &vl, &vu, NULL, NULL,
			&abstol, &m, w.GetData(), NULL, &n, work.GetData(), &lwork,
			iwork.GetData(), ifail.GetData(), &info );


	evals.SetSize(m);
	std::copy(w.GetData(), w.GetData()+m, evals.GetData());

	return info;
}

int SymEigensolver::Compute(DenseMatrix & A_, Vector & evals, DenseMatrix & evects, double vl, double vu)
{
	int n = A_.Size();

	if(n_max < n)
		AllocateWorkspace(n);

	jobz  = 'V';
	range = 'V';
	uplo  = 'U';

	double * a = NULL;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
	}
	else
	{
		a = A_.Data();
	}

	dsyevx_( &jobz, &range, &uplo, &n, a, &n, NULL, NULL, NULL, NULL,
			&abstol, &m, w.GetData(), Z.GetData(), &n, work.GetData(), &lwork,
			iwork.GetData(), ifail.GetData(), &info );


	evals.SetSize(m);
	std::copy(w.GetData(), w.GetData()+m, evals.GetData());

	evects.SetSize(n, m);
	std::copy( Z.GetData(), Z.GetData()+n*m, evects.Data() );

	return info;
}

int SymEigensolver::Compute(DenseMatrix & A_, DenseMatrix & B_, Vector & evals, double vl, double vu)
{


	int n = A_.Size();

	if(n_max < n)
		AllocateWorkspace(n);

	jobz  = 'N';
	range = 'V';
	uplo  = 'U';

	double * a = NULL, * b = NULL;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
		b = B.GetData();
		std::copy(B_.Data(), B_.Data()+n*n, b);
	}
	else
	{
		a = A_.Data();
		b = B_.Data();
	}

	dsygvx_( &itype, &jobz, &range, &uplo, &n, a, &n, b, &n, &vl, &vu, NULL, NULL,
			&abstol, &m, w.GetData(), NULL, &n, work.GetData(), &lwork,
			iwork.GetData(), ifail.GetData(), &info );

	evals.SetSize(m);
	std::copy(w.GetData(), w.GetData()+m, evals.GetData());

	return info;
}

int SymEigensolver::Compute(DenseMatrix & A_, DenseMatrix & B_, Vector & evals, DenseMatrix & evects, double vl, double vu)
{
	int n = A_.Size();

	if(n_max < n)
		AllocateWorkspace(n);

	jobz  = 'Y';
	range = 'I';
	uplo  = 'U';

	double * a = NULL, * b = NULL;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
		b = B.GetData();
		std::copy(B_.Data(), B_.Data()+n*n, b);
	}
	else
	{
		a = A_.Data();
		b = B_.Data();
	}

	dsygvx_( &itype, &jobz, &range, &uplo, &n, a, &n, b, &n, &vl, &vu, NULL, NULL,
			&abstol, &m, w.GetData(), Z.GetData(), &n, work.GetData(), &lwork,
			iwork.GetData(), ifail.GetData(), &info );

	evals.SetSize(m);
	std::copy(w.GetData(), w.GetData()+m, evals.GetData());

	evects.SetSize(n, m);
	std::copy( Z.GetData(), Z.GetData()+n*m, evects.Data() );

	return info;
}


int SymEigensolver::Compute(DenseMatrix & A_, Vector & d, Vector & evals, double vl, double vu)
{
	if(itype != 1)
		mfem_error("Only A x = lambda D x is supported \n");

	int n = A_.Size();
	double * a = NULL;

	if(n_max < n)
		AllocateWorkspace(n);

	int oldOpt = doNotOverwrite;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
		doNotOverwrite = 0;
	}
	else
	{
		a = A_.Data();
	}

	DenseMatrix sA(a,n,n);
	sA.InvSymmetricScaling(d);


	int r;
	r = Compute(sA, evals,vl,vu);

	sA.ClearExternalData();
	doNotOverwrite = oldOpt;

	return r;
}


int SymEigensolver::Compute(DenseMatrix & A_, Vector & d, Vector & evals, DenseMatrix & evects, double vl, double vu)
{
	if(itype != 1)
		mfem_error("Only A x = lambda D x is supported \n");

	int n = A_.Size();
	double * a = NULL;

	if(n_max < n)
		AllocateWorkspace(n);

	int oldOpt = doNotOverwrite;

	if(doNotOverwrite)
	{
		a = A.GetData();
		std::copy(A_.Data(), A_.Data()+n*n, a);
		doNotOverwrite = 0;
	}
	else
	{
		a = A_.Data();
	}

	DenseMatrix sA(a,n,n);
	sA.InvSymmetricScaling(d);


	int r;
	r = Compute(sA, evals, evects, vl,vu);

	Vector sqrtd(n);
	for(double * out = sqrtd.GetData(), * in = d.GetData(); in != d.GetData()+n; ++out, ++in)
		*out = sqrt(*in);

	evects.InvLeftScaling(sqrtd);

	sA.ClearExternalData();
	doNotOverwrite = oldOpt;

	return r;
}


// Compute all eigenvalues A x = lambda x such that | lambda_i | \leq rel_tol |\lambda_\max|
int SymEigensolver::ComputeSmallerMagnitude(DenseMatrix & A, Vector & evals, double rel_tol)
{
	int n = A.Size();
	int r = Compute(A,w);
#if 1
	{
		Array<double> tmp(w.GetData(), w.Size());
		if( ! tmp.IsSorted() )
			mfem_error("evals are not sorted :( ");
	}

	if( w(0) < -1e-10 )
		mfem_error("A is not spd\n");
#endif

	double tol = w(n-1)*rel_tol;
	int m;
	for( m = 0; m < n && fabs(w(m)) < tol; ++m);

	evals.SetSize(m);
	std::copy( w.GetData(), w.GetData()+m, evals.GetData() );

	return r;
}
// Compute all eigenvectors and eigenvalues A x = lambda x such that | lambda_i | \leq rel_tol |\lambda_\max|
int SymEigensolver::ComputeSmallerMagnitude(DenseMatrix & A, Vector & evals, DenseMatrix & evects, double rel_tol)
{
	int oldOpt = doNotOverwrite;

	doNotOverwrite = 1;
	int n = A.Size();
	int r = Compute(A,w);

	if(r)
		return r;

#if 1
	{
		Array<double> tmp(w.GetData(), w.Size());
		if( ! tmp.IsSorted() )
			mfem_error("evals are not sorted :( ");
	}

	if( w(0) < -1e-10 )
	{
		std::cout << w(0) << " " << abstol << "\n";
		mfem_error("A is not spd\n");
	}

#endif

	double tol = w(n-1)*rel_tol;
	int m;
	for( m = 0; m < n && fabs(w(m)) < tol; ++m);

	doNotOverwrite = oldOpt;

	r = Compute(A, evals, evects, 1, m);

	return r;

}
// Compute all eigenvalues of the generalized eigenvalue problem A x = lambda B x such that | lambda_i | \leq rel_tol |\lambda_\max|
int SymEigensolver::ComputeSmallerMagnitude(DenseMatrix & A, DenseMatrix & B, Vector & evals, double rel_tol)
{
	int n = A.Size();
	int r = Compute(A,B,w);
#if 1
	{
		Array<double> tmp(w.GetData(), w.Size());
		if( ! tmp.IsSorted() )
			mfem_error("evals are not sorted :( ");
	}

	if( w(0) < -1e-10 )
		mfem_error("A is not spd\n");
#endif

	double tol = w(n-1)*rel_tol;
	int m;
	for( m = 0; m < n && fabs(w(m)) < tol; ++m);

	evals.SetSize(m);
	std::copy( w.GetData(), w.GetData()+m, evals.GetData() );

	return r;
}
// Compute all eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda B x such that | lambda_i | \leq rel_tol |\lambda_\max|
int SymEigensolver::ComputeSmallerMagnitude(DenseMatrix & A, DenseMatrix & B, Vector & evals, DenseMatrix & evects, double rel_tol)
{
	int oldOpt = doNotOverwrite;

	doNotOverwrite = 1;
	int n = A.Size();
	int r = Compute(A,B,w);

	if(r)
		return r;

#if 1
	{
		Array<double> tmp(w.GetData(), w.Size());
		if( ! tmp.IsSorted() )
			mfem_error("evals are not sorted :( ");
	}

	if( w(0) < -1.e-10 )
	{
		std::cout << w(0) << "\n";
		mfem_error("A is not spd\n");
	}
#endif

	double tol = w(n-1)*rel_tol;
	int m;
	for( m = 0; m < n && fabs(w(m)) < tol; ++m);

	doNotOverwrite = oldOpt;

	m = std::max(1,m);
	r = Compute(A, B, evals, evects, 1, m);

	return r;

}
// Compute all eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d) such that | lambda_i | \leq rel_tol |\lambda_\max|
int SymEigensolver::ComputeSmallerMagnitude(DenseMatrix & A, Vector & d, Vector & evals, double rel_tol)
{
	if(itype != 1)
	mfem_error("Only A x = lambda D x is supported \n");

	int n = A.Size();
	int r = Compute(A,d,w);
#if 1
	{
		Array<double> tmp(w.GetData(), w.Size());
		if( ! tmp.IsSorted() )
			mfem_error("evals are not sorted :( ");
	}

	if( w(0) < -1e-10 )
		mfem_error("A is not spd\n");
#endif

	double tol = w(n-1)*rel_tol;
	int m;
	for( m = 0; m < n && fabs(w(m)) <= tol; ++m);

	evals.SetSize(m);
	std::copy( w.GetData(), w.GetData()+m, evals.GetData() );

	return r;
}
// Compute all eigenvectors and eigenvalues of the generalized eigenvalue problem A x = lambda D x, D = diag(d) such that | lambda_i | \leq rel_tol |\lambda_\max|
int SymEigensolver::ComputeSmallerMagnitude(DenseMatrix & A, Vector & d, Vector & evals, DenseMatrix & evects, double rel_tol)
{
	if(itype != 1)
		mfem_error("Only A x = lambda D x is supported \n");

	int oldOpt = doNotOverwrite;

	doNotOverwrite = 1;
	int n = A.Size();
	int r = Compute(A,d,w);

	if(r)
		return r;

#if 1
	{
		Array<double> tmp(w.GetData(), w.Size());
		if( ! tmp.IsSorted() )
			mfem_error("evals are not sorted :( ");
	}

	if( w(0) < -1.e-10 )
	{
		std::cout << w(0) << " " << abstol << "\n";
		mfem_error("A is not spd\n");
	}
#endif

	double tol = w(n-1)*rel_tol;
	int m;
	for( m = 0; m < n && fabs(w(m)) <= tol; ++m);

	doNotOverwrite = oldOpt;

	r = Compute(A, d, evals, evects, 1, m);

	return r;
}

void SymEigensolver::AllocateWorkspace(int n)
{
	double qwork, qworkg;
	n_max = n;
	lwork=-1;

	A.SetSize(n*n);
	B.SetSize(n*n);
	w.SetSize(n);
	Z.SetSize(n*n);
	iwork.SetSize(5*n);
	ifail.SetSize(n);

	double vl(-1.), vu(1.);
	int il(1), iu(n);

	dsyevx_( &jobz, &range, &uplo, &n, (double*)NULL, &n, &vl, &vu, &il, &iu, &abstol, &m, NULL, NULL, &n, &qwork, &lwork,
			iwork.GetData(), ifail.GetData(), &info );

	dsygvx_( &itype, &jobz, &range, &uplo, &n, NULL, &n, NULL, &n, &vl, &vu, &il, &iu, &abstol, &m, NULL, NULL, &n, &qworkg, &lwork,
			iwork.GetData(), ifail.GetData(), &info );

	lwork = std::max( (int)qwork, (int)qworkg ) + 1;
	work.SetSize(lwork);
}
