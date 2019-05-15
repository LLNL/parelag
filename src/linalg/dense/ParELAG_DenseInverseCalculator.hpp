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

#ifndef DENSEINVERSECALCULATOR_HPP_
#define DENSEINVERSECALCULATOR_HPP_

#include <mfem.hpp>

#include "ParELAG_MultiVector.hpp"
#include "utilities/elag_utilities.hpp"

namespace parelag
{

/**
   Interface for classes that implement (small) dense inverses.

   Generally you construct, Compute() and then Mult()

   @todo should this inherit from mfem::Solver or something else in mfem ?
*/
class DenseInverseCalculator
{
public:
    DenseInverseCalculator() {};
    virtual ~DenseInverseCalculator() = default;

    /**
       Compute the necessary data for obtaining the inverse.
    */
    virtual int Compute(mfem::DenseMatrix & A) = 0;

    /**
       Solve the system with right hand side x.
    */
    virtual void Mult(const mfem::Vector & x, mfem::Vector & y) const = 0;

    /**
       Solve the system with multiple right hand sides encoded in x.
    */
    virtual void Mult(const MultiVector & x, MultiVector & y) const = 0;

    /**
       Solve the system with multiple right hand sides encoded in x.
    */
    virtual void Mult(const mfem::DenseMatrix & x, mfem::DenseMatrix & y) const = 0;
};

}//namespace parelag
#endif /* DENSEINVERSECALCULATOR_HPP_ */

