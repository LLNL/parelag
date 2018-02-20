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

#include "linalg/dense/ParELAG_LDLCalculator.hpp"

#include "linalg/dense/ParELAG_LAPACK.hpp"

namespace parelag
{
using namespace mfem;

LDLCalculator::LDLCalculator():
    uplo_('L'),
    work_(0),
    lwork_(-1),
    ipiv_(0),
    Adata_(static_cast<double*>(NULL) ),
    max_n_(0),
    n_(0)
{}

int LDLCalculator::Compute(DenseMatrix & A)
{
    n_ = A.Height();
    PARELAG_ASSERT(n_ > 0);

    if (n_ > max_n_)
        _do_allocate_optimal_size(n_);

    Adata_ = A.Data();
    return Lapack<double>::SYTRF(
        uplo_, n_, Adata_, n_, ipiv_.GetData(), work_.GetData(), lwork_);
}

void LDLCalculator::Mult(const Vector & x, Vector & y) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        x.Size() != y.Size(),
        std::runtime_error,
        "LDLCalculator::Mult(): x and y not compatible sizes. "
        "x.Size() = " << x.Size() << "; y.Size() = " << y.Size() << ".");

    PARELAG_TEST_FOR_EXCEPTION(
        x.Size() != n_,
        std::runtime_error,
        "LDLCalculator::Mult(): x is the wrong size. "
        "x.Size() = " << x.Size() << "; n_ = " << n_ << ".");

    if (x.GetData() != y.GetData())
        y = x;

    Lapack<double,ExceptionOnNonzeroError>::SYTRS(
        uplo_, n_, 1, Adata_, n_, ipiv_.GetData(), y.GetData(), n_);
}

void LDLCalculator::Mult(const MultiVector & x, MultiVector & y) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        x.Size() != y.Size(),
        std::runtime_error,
        "LDLCalculator::Mult(): x and y not compatible sizes. "
        "x.Size() = " << x.Size() << "; y.Size() = " << y.Size() << ".");

    PARELAG_TEST_FOR_EXCEPTION(
        x.Size() != n_,
        std::runtime_error,
        "LDLCalculator::Mult(): x is the wrong size. "
        "x.Size() = " << x.Size() << "; n_ = " << n_ << ".");

    PARELAG_TEST_FOR_EXCEPTION(
        x.NumberOfVectors() != y.NumberOfVectors(),
        std::runtime_error,
        "LDLCalculator::Mult(): x and y incompatible number of vectors. "
        "x.NumberOfVectors() = " << x.NumberOfVectors() << "; "
        "y.NumberOfVectors() = " << y.NumberOfVectors() << ".");

    if (x.GetData() != y.GetData())
        y = x;

    Lapack<double,ExceptionOnNonzeroError>::SYTRS(
        uplo_, n_, y.NumberOfVectors(), Adata_, n_, ipiv_.GetData(),
        y.GetData(), y.LeadingDimension());
}

void LDLCalculator::Mult(const DenseMatrix & x, DenseMatrix & y) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        x.Height() != y.Height(),
        std::runtime_error,
        "LDLCalculator::Mult(): x and y not compatible sizes. "
        "x.Size() = " << x.Size() << "; y.Size() = " << y.Size() << ".");

    PARELAG_TEST_FOR_EXCEPTION(
        x.Height() != n_,
        std::runtime_error,
        "LDLCalculator::Mult(): x is the wrong size. "
        "x.Size() = " << x.Size() << "; n_ = " << n_ << ".");

    PARELAG_TEST_FOR_EXCEPTION(
        x.Width() != y.Width(),
        std::runtime_error,
        "LDLCalculator::Mult(): x and y incompatible widths. "
        "x.Width() = " << x.Width() << "; y.Width() = " << y.Width() << ".");

    if (x.Data() != y.Data())
        y = x;

    Lapack<double,ExceptionOnNonzeroError>::SYTRS(
        uplo_, n_, y.Width(), Adata_, n_, ipiv_.GetData(),
        y.Data(), y.Height());
}

void LDLCalculator::_do_allocate_optimal_size(int loc_n)
{
    double qwork = 0.;

    max_n_ = loc_n;
    ipiv_.SetSize(loc_n);
    Lapack<double,ExceptionOnNonzeroError>::SYTRF(
        uplo_, loc_n, Adata_, loc_n, ipiv_.GetData(), &qwork, -1);

    lwork_ = static_cast<int>(qwork);
    work_.SetSize(lwork_);
}
}//namespace parelag
