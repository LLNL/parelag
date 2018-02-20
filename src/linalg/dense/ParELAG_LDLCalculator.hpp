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

#ifndef LDLCALCULATOR_HPP_
#define LDLCALCULATOR_HPP_

#include <mfem.hpp>

#include "ParELAG_MultiVector.hpp"
#include "utilities/elag_utilities.hpp"

namespace parelag
{

/**
   Solver that interfaces LAPACK's dsytrf and dsytrf, for doing a direct solve
   for a *symmetric* dense matrix.

   Generally you construct, Compute() and then Mult()

   @todo should this inherit from mfem::Solver ?
*/
class LDLCalculator
{
public:
    LDLCalculator();
    virtual ~LDLCalculator() = default;

    LDLCalculator(LDLCalculator const&) = delete;
    LDLCalculator(LDLCalculator&&) = delete;

    LDLCalculator& operator=(LDLCalculator const&) = delete;
    LDLCalculator& operator=(LDLCalculator&&) = delete;

    /**
       Compute the factorization. Only the *lower* triangular part of
       A is used. Symmetry is not checked.
    */
    int Compute(mfem::DenseMatrix & A);

    /**
       Solve the system with right hand side x.
    */
    void Mult(const mfem::Vector & x, mfem::Vector & y) const;

    /**
       Solve the system with multiple right hand sides encoded in x.
    */
    void Mult(const MultiVector & x, MultiVector & y) const;

    /**
       Solve the system with multiple right hand sides encoded in x.
    */
    void Mult(const mfem::DenseMatrix & x, mfem::DenseMatrix & y) const;

private:
    /** \brief Set work Vector to have appropriate size. */
    void _do_allocate_optimal_size(int nax_n);

    // TODO: think about mutable for some of these variables, and const
    //       methods above

    char uplo_;
    mutable mfem::Vector work_;
    mutable int lwork_;
    mutable mfem::Array<int> ipiv_;
    double * Adata_;

    int max_n_;
    int n_;
};

}//namespace parelag
#endif /* LDLCALCULATOR_HPP_ */
