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

#ifndef QDQCALCULATOR_HPP_
#define QDQCALCULATOR_HPP_

#include <mfem.hpp>

#include "ParELAG_MultiVector.hpp"
#include "ParELAG_DenseInverseCalculator.hpp"
#include "utilities/elag_utilities.hpp"

namespace parelag
{

/**
   Solver that interfaces LAPACK, for doing a direct solve
   for a *symmetric* dense matrix using eiganvalue decomposition.

   Generally you construct, Compute() and then Mult()
*/
class QDQCalculator : public DenseInverseCalculator
{
public:
    QDQCalculator();
    virtual ~QDQCalculator() = default;

    QDQCalculator(QDQCalculator const&) = delete;
    QDQCalculator(QDQCalculator&&) = delete;

    QDQCalculator& operator=(QDQCalculator const&) = delete;
    QDQCalculator& operator=(QDQCalculator&&) = delete;

    /**
       Compute the factorization. Only the *upper* triangular part of
       A is used. Symmetry is not checked.
    */
    virtual int Compute(mfem::DenseMatrix & A);

    /**
       Solve the system with right hand side x.
    */
    virtual void Mult(const mfem::Vector & x, mfem::Vector & y) const;

    /**
       Solve the system with multiple right hand sides encoded in x.
    */
    virtual void Mult(const MultiVector & x, MultiVector & y) const;

    /**
       Solve the system with multiple right hand sides encoded in x.
    */
    virtual void Mult(const mfem::DenseMatrix & x, mfem::DenseMatrix & y) const;

private:

    int n_;
    mfem::Vector D_;
    mfem::DenseMatrix Q_;
};

}//namespace parelag
#endif /* QDQCALCULATOR_HPP_ */
