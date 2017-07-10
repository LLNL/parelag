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
   void dsytrf_(char * UPLO, int * N, double * A,int * LDA, int * IPIV,
                double * WORK,int * LWORK, int * INFO );
   void dsytrs_(char * UPLO, int * N, int * NRHS, double * A, int * LDA,
                int * IPIV, double * B, int * LDB, int * INFO );
}

LDLCalculator::LDLCalculator():
   uplo('L'),
   work(0),
   lwork(-1),
   ipiv(0),
   info(0),
   qwork(0.),
   Adata(static_cast<double*>(NULL) ),
   max_n(0),
   n(0)
{

}

LDLCalculator::~LDLCalculator()
{

}

int LDLCalculator::Compute(DenseMatrix & A)
{
   n = A.Height();

   if (n > max_n)
      AllocateOptimalSize(n);

   Adata = A.Data();
   dsytrf_(&uplo, &n, Adata, &n, ipiv.GetData(), work.GetData(), &lwork, &info );

   return info;
}
void LDLCalculator::Mult(const Vector & x, Vector & y)
{
   if ( x.Size() != y.Size() )
      mfem_error("LDLCalculator::Mult #1\n");

   if ( x.Size() != n )
      mfem_error("LDLCalculator::Mult #2\n");

   if (x.GetData() != y.GetData() )
      y = x;

   int nrhs = 1;

   dsytrs_(&uplo, &n, &nrhs, Adata, &n, ipiv.GetData(), y.GetData(), &n, &info );

   if (info != 0)
      mfem_error("LDLCalculator::Mult #3\n");
}
void LDLCalculator::Mult(const MultiVector & x, MultiVector & y)
{
   if ( x.Size() != y.Size() )
      mfem_error("LDLCalculator::Mult #1\n");

   if ( x.Size() != n )
      mfem_error("LDLCalculator::Mult #2\n");

   if (x.NumberOfVectors() != y.NumberOfVectors())
      mfem_error("LDLCalculator::Mult #3\n");

   if (x.GetData() != y.GetData() )
      y = x;

   int nrhs = y.NumberOfVectors();
   int ldb  = y.LeadingDimension();

   dsytrs_(&uplo, &n, &nrhs, Adata, &n, ipiv.GetData(), y.GetData(), &ldb, &info );

   if (info != 0)
      mfem_error("LDLCalculator::Mult #4\n");
}
void LDLCalculator::Mult(const DenseMatrix & x, DenseMatrix & y)
{
   if ( x.Height() != y.Height() )
      mfem_error("LDLCalculator::Mult #1\n");

   if ( x.Height() != n )
      mfem_error("LDLCalculator::Mult #2\n");

   if (x.Width() != y.Width())
      mfem_error("LDLCalculator::Mult #3\n");

   if (x.Data() != y.Data() )
      y = x;

   int nrhs = y.Width();
   int ldb  = y.Height();

   dsytrs_(&uplo, &n, &nrhs, Adata, &n, ipiv.GetData(), y.Data(), &ldb, &info );

   if (info != 0)
      mfem_error("LDLCalculator::Mult #4\n");
}

void LDLCalculator::AllocateOptimalSize(int n)
{
   max_n = n;
   ipiv.SetSize(n);
   lwork = -1;
   dsytrf_(&uplo, &n, Adata, &n, ipiv.GetData(), &qwork, &lwork, &info );

   lwork = (int) qwork;
   work.SetSize(lwork);
}


