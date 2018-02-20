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

#include "linalg/solver_core/ParELAG_SymmetrizedUmfpack.hpp"

#include "utilities/elagError.hpp"

namespace parelag
{
using namespace mfem;

SymmetrizedUmfpack::SymmetrizedUmfpack(SparseMatrix &A):
    solver_(A),
    help_(A.Size())
{
    help_ = 0.;
}

SymmetrizedUmfpack::SymmetrizedUmfpack(BlockMatrix &A):
    Amono_(A.CreateMonolithic()),
    help_(Amono_->Size())
{
    solver_.SetOperator(*Amono_);
}

void SymmetrizedUmfpack::SetOperator(const Operator &op)
{
    MFEM_VERIFY(op.Width() == op.Height(), "");
    height = width = op.Width();

    const SparseMatrix * A1 = dynamic_cast<const SparseMatrix *>(&op);
    if(A1)
    {
        solver_.SetOperator(*A1);
        help_.SetSize(A1->Size());
        help_ = 0.;
        return;
    }

    const BlockMatrix * A2 = dynamic_cast<const BlockMatrix*>(&op);
    if(A2)
    {
        Amono_.reset(A2->CreateMonolithic());
        solver_.SetOperator(*Amono_);
        help_.SetSize(A2->Height());
        help_ = 0.;
        return;
    }

    PARELAG_TEST_FOR_EXCEPTION(
        !A1 && !A2,
        std::runtime_error,
        "SymmetrizedUmfpack::SetOperator(...): "
        "Operator must be SparseMatrix or BlockMatrix!");
}

void SymmetrizedUmfpack::Mult(const Vector &b, Vector &x) const
{
    solver_.Mult(b,help_);
    solver_.MultTranspose(b, x);

    add(.5, help_, .5, x, x);
}

void SymmetrizedUmfpack::MultTranspose(const Vector &b, Vector &x) const
{
    Mult(b,x); // Solver is symmetric...
}

void SymmetrizedUmfpack::Mult(const MultiVector &b, MultiVector &x) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        b.NumberOfVectors() != x.NumberOfVectors(),
        std::runtime_error,
        "SymmetrizedUmfpack::Mult #1");

    const int nrhs = b.NumberOfVectors();

    Vector bview, xview;
    for(int i(0); i < nrhs; ++i)
    {
        const_cast<MultiVector &>(b).GetVectorView(i, bview);
        x.GetVectorView(i, xview);
        this->Mult(bview, xview);
    }
}

void SymmetrizedUmfpack::MultTranspose(const MultiVector &b, MultiVector &x) const
{
    Mult(b,x); // Solver is symmetric...
}

void SymmetrizedUmfpack::Mult(const DenseMatrix &b, DenseMatrix &x) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        b.Width() != x.Width(),
        std::runtime_error,
        "SymmetrizedUmfpack::Mult #1");

    const int nrhs = b.Width();
    Vector bview, xview;
    for(int i(0); i < nrhs; ++i)
    {
        const_cast<DenseMatrix &>(b).GetColumnReference(i, bview);
        x.GetColumnReference(i,xview);
        this->Mult(bview, xview);
    }
}

void SymmetrizedUmfpack::MultTranspose(const DenseMatrix &b, DenseMatrix &x) const
{
    Mult(b,x); // Solver is symmetric...
}

}//namespace parelag
