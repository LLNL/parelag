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
 void dgesvd_(
      const char* jobu,
      const char* jobvt,
      const int* m,
      const int* n,
      double* a,
      const int* lda,
      double* s,
      double* u,
      const int* ldu,
      double* vt,
      const int* ldvt,
      double* work,
      const int* lwork,
      int* info
      );
}

SVD_Calculator::SVD_Calculator():
	flag(0),
	jobu('N'),
	jobvt('N'),
	work(NULL),
	lwork(-1),
	info(0),
	qwork(0),
	maxNRows(-1),
	maxNCols(-1)
{

}

SVD_Calculator::~SVD_Calculator()
{
	delete[] work;
}


void SVD_Calculator::setFlag(int flag_)
{
	flag = flag_;

	char job = 'A';
	if(flag & SKINNY)
		job = 'S';
	else
		job = 'A';

	if(flag & COMPUTE_U)
		jobu = job;
	else
		jobu = 'N';

	if(flag & COMPUTE_VT)
		jobvt = 'S';
	else
		jobvt = 'N';

}

void SVD_Calculator::setFlagOA()
{
	flag = COMPUTE_U | COMPUTE_VT | SKINNY;
	jobu = 'O';
	jobvt = 'A';
}

void SVD_Calculator::setFlagON()
{
	flag = COMPUTE_U |  SKINNY;
	jobu = 'O';
	jobvt = 'N';
}

void SVD_Calculator::AllocateOptimalSize(int maxNRows_, int maxNCols_)
{
	maxNRows = maxNRows_;
	maxNCols = maxNCols_;

	double * a = new double[maxNRows*maxNCols];
	std::fill(a, a+maxNRows*maxNCols, 0.0);
	double * s = a;
	double * u = a;
	double * vt = a;

	lwork = -1;
	dgesvd_(&jobu, &jobvt, &maxNRows, &maxNCols, a, &maxNRows,
	           s, u, &maxNRows, vt, &maxNCols, &qwork, &lwork, &info);

	delete[] a;

	lwork = (int) qwork;
	work = new double[lwork];

}

void SVD_Calculator::Compute(MultiVector & A, Vector & singularValues, MultiVector & U, MultiVector * VT, int flag_)
{
	if(VT == NULL)
		flag_ &= !COMPUTE_VT;

	int nrows = A.Size();
	int ncols = A.NumberOfVectors();

	if(flag != flag || nrows > maxNRows || ncols > maxNCols)
	{
		setFlag(flag_);
		AllocateOptimalSize(maxNRows,maxNCols);
	}

	singularValues.SetSize(ncols);

	if(flag & SKINNY)
		U.SetSizeAndNumberOfVectors(nrows, ncols);
	else
		U.SetSizeAndNumberOfVectors(nrows, nrows);

	int ldA = A.LeadingDimension();
	int ldU = U.LeadingDimension();
	int ldV = 0;

	double * vt = NULL;
	if(flag & COMPUTE_VT)
	{
			VT->SetSizeAndNumberOfVectors(ncols, ncols);
			vt = VT->GetData();
			ldV = VT->LeadingDimension();
	}

	 dgesvd_(&jobu, &jobvt, &nrows, &ncols, A.GetData(), &ldA,
	           singularValues.GetData(), U.GetData(), &ldU, vt, &ncols, work, &lwork, &info);

	   if (info)
	   {
	      std::cout << "DenseMatrix::SingularValues : info = " << info << std::endl;
	      mfem_error();
	   }

}

void SVD_Calculator::ComputeOA(MultiVector & A, Vector & singularValues, MultiVector & VT)
{

	int flag_ = COMPUTE_U | COMPUTE_VT | SKINNY;

	int nrows = A.Size();
	int ncols = A.NumberOfVectors();

	if(flag_ != flag || nrows > maxNRows || ncols > maxNCols)
	{
		setFlagOA();
		AllocateOptimalSize(nrows,ncols);
	}

	singularValues.SetSize(ncols);

	int ldA = A.LeadingDimension();
	int ldV = VT.LeadingDimension();

	double * u = NULL;
	double * vt = VT.GetData();

	if(VT.Size() != ncols || VT.NumberOfVectors() != ncols)
		mfem_error("Dimensions of VT are not correct!");

	 dgesvd_(&jobu, &jobvt, &nrows, &ncols, A.GetData(), &ldA,
	           singularValues.GetData(), u, &ldA, vt, &ldV, work, &lwork, &info);

	   if (info)
	   {
	      std::cout << "DenseMatrix::SingularValues : info = " << info << std::endl;
	      mfem_error();
	   }

}

void SVD_Calculator::ComputeOA(Vector & sqrt_w, MultiVector & A, Vector & singularValues, MultiVector & VT)
{
	A.Scale(sqrt_w);
	ComputeOA(A, singularValues, VT);
	A.InverseScale(sqrt_w);
}

void SVD_Calculator::ComputeON(MultiVector & A, Vector & singularValues)
{

	int flag_ = COMPUTE_U | SKINNY;

	int nrows = A.Size();
	int ncols = A.NumberOfVectors();

	if(flag_ != flag || nrows > maxNRows || ncols > maxNCols)
	{
		setFlagON();
		AllocateOptimalSize(nrows,ncols);
	}

	int nSingValues = (nrows < ncols) ? nrows : ncols;
	singularValues.SetSize(nSingValues);

	int ldA = std::max( A.LeadingDimension(), 1);

	double * u = NULL;
	double * vt = NULL;

	 dgesvd_(&jobu, &jobvt, &nrows, &ncols, A.GetData(), &ldA,
	           singularValues.GetData(), u, &ldA, vt, &ldA, work, &lwork, &info);

	   if (info)
	   {
	      std::cout << "DenseMatrix::SingularValues : info = " << info << std::endl;
	      mfem_error();
	   }

#ifdef ELAG_DEBUG
	   {
		   double val = singularValues(0)+1;
		   for(double * it = singularValues.GetData(); it != singularValues.GetData()+ nSingValues; ++it )
			   if(*it > val)
				   mfem_error("Singular Values are not sorted :(");
			   else
				   val = *it;
	   }
#endif

}

void SVD_Calculator::ComputeON(DenseMatrix & A, Vector & singularValues)
{
	MultiVector tmp(A.Data(), A.Width(), A.Height());
	ComputeON( tmp, singularValues);
}

void SVD_Calculator::ComputeON(Vector & sqrt_w, MultiVector & A, Vector & singularValues)
{
//	sqrt_w.CheckFinite(); A.CheckFinite();
	A.Scale(sqrt_w);
	ComputeON(A, singularValues);
	A.InverseScale(sqrt_w);
//	A.CheckFinite();
}

namespace
{
	double reciprocal(const double & a)
	{
		return 1./a;
	}
}

void SVD_Calculator::ComputeON(DenseMatrix & W, MultiVector & A, Vector & singularValues)
{
	elag_assert(W.Height() == W.Width() );
	SymEigensolver eigs;
	eigs.doNotOverwrite = 1;

	int n = W.Height();
	Vector evals(n);
	DenseMatrix evects(n);
	eigs.Compute(W, evals, evects);

	std::transform(evals.GetData(), evals.GetData()+n, evals.GetData(), sqrt);
	DenseMatrix X(n);
	MultADAt(evects, evals, X);

	MultiVector XA( A.NumberOfVectors(), A.Size() );
	Mult(X,A, XA);

	ComputeON(XA, singularValues);

	std::transform(evals.GetData(), evals.GetData()+n, evals.GetData(), reciprocal);
	MultADAt(evects, evals, X);
	Mult(X,XA, A);

}


