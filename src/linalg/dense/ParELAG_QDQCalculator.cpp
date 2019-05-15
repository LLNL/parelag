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

#include "linalg/dense/ParELAG_QDQCalculator.hpp"

#include "linalg/dense/ParELAG_LAPACK.hpp"
#include "linalg/dense/ParELAG_Eigensolver.hpp"

namespace parelag
{
using namespace mfem;

QDQCalculator::QDQCalculator():
    n_(0)
{}

int QDQCalculator::Compute(DenseMatrix & A)
{
    n_ = A.Height();
    PARELAG_ASSERT(n_ > 0);
    PARELAG_TEST_FOR_EXCEPTION(
        A.Width() != n_,
        std::runtime_error,
        "QDQCalculator::Compute(): A not square. "
        "A.Height() = " << A.Height() << "; A.Width() = " << A.Width() << ".");

    std::vector<double> evals;
    SymEigensolver eigs;
    eigs.ComputeAll(A, evals, Q_);
    PARELAG_ASSERT(evals.size() == n_ && Q_.Height() == Q_.Width()
                   && Q_.Height() == n_);
    D_.SetSize(n_);
    for (int i=0; i < n_; ++i)
    {
        const double val = evals[i];
        PARELAG_TEST_FOR_EXCEPTION(
            (-1e-12 < val && val < 1e-12),
            std::runtime_error,
            "QDQCalculator::Compute(): An almost zero eigenvalue. "
            "i = " << i << "; evals[i] = " << val << ".");
            D_(i) = 1.0/val;
    }
}

void QDQCalculator::Mult(const Vector & x, Vector & y) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        x.Size() != y.Size(),
        std::runtime_error,
        "QDQCalculator::Mult(): x and y not compatible sizes. "
        "x.Size() = " << x.Size() << "; y.Size() = " << y.Size() << ".");

    PARELAG_TEST_FOR_EXCEPTION(
        x.Size() != n_,
        std::runtime_error,
        "QDQCalculator::Mult(): x is the wrong size. "
        "x.Size() = " << x.Size() << "; n_ = " << n_ << ".");

    Vector tmp(n_);
    Q_.MultTranspose(x, tmp);
    for (int i=0; i < n_; ++i)
        tmp(i) *= D_(i);
    Q_.Mult(tmp, y);
}

void QDQCalculator::Mult(const MultiVector & x, MultiVector & y) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        x.Size() != y.Size(),
        std::runtime_error,
        "QDQCalculator::Mult(): x and y not compatible sizes. "
        "x.Size() = " << x.Size() << "; y.Size() = " << y.Size() << ".");

    PARELAG_TEST_FOR_EXCEPTION(
        x.Size() != n_,
        std::runtime_error,
        "QDQCalculator::Mult(): x is the wrong size. "
        "x.Size() = " << x.Size() << "; n_ = " << n_ << ".");

    PARELAG_TEST_FOR_EXCEPTION(
        x.NumberOfVectors() != y.NumberOfVectors(),
        std::runtime_error,
        "QDQCalculator::Mult(): x and y incompatible number of vectors. "
        "x.NumberOfVectors() = " << x.NumberOfVectors() << "; "
        "y.NumberOfVectors() = " << y.NumberOfVectors() << ".");

    MultiVector tmp(x.NumberOfVectors(), n_);
    MultTranspose(Q_, x, tmp);
    tmp.Scale(D_);
    parelag::Mult(Q_, tmp, y);
}

void QDQCalculator::Mult(const DenseMatrix & x, DenseMatrix & y) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        x.Height() != y.Height(),
        std::runtime_error,
        "QDQCalculator::Mult(): x and y not compatible sizes. "
        "x.Size() = " << x.Size() << "; y.Size() = " << y.Size() << ".");

    PARELAG_TEST_FOR_EXCEPTION(
        x.Height() != n_,
        std::runtime_error,
        "QDQCalculator::Mult(): x is the wrong size. "
        "x.Size() = " << x.Size() << "; n_ = " << n_ << ".");

    PARELAG_TEST_FOR_EXCEPTION(
        x.Width() != y.Width(),
        std::runtime_error,
        "QDQCalculator::Mult(): x and y incompatible widths. "
        "x.Width() = " << x.Width() << "; y.Width() = " << y.Width() << ".");

    DenseMatrix tmp(n_, x.Width());
    MultAtB(Q_, x, tmp);
    tmp.LeftScaling(D_);
    mfem::Mult(Q_, tmp, y);
}

}//namespace parelag
