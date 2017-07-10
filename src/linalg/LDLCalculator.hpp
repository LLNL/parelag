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

#ifndef LDLCALCULATOR_HPP_
#define LDLCALCULATOR_HPP_

class LDLCalculator
{
public:
   LDLCalculator();
   virtual ~LDLCalculator();

   int Compute(DenseMatrix & A);
   void Mult(const Vector & x, Vector & y);
   void Mult(const MultiVector & x, MultiVector & y);
   void Mult(const DenseMatrix & x, DenseMatrix & y);

   void AllocateOptimalSize(int nax_n);

private:
   char   uplo;
   Vector work;
   int    lwork;
   Array<int>  ipiv;
   int    info;
   double qwork;
   double * Adata;

   int max_n;
   int n;
};

#endif /* LDLCALCULATOR_HPP_ */
