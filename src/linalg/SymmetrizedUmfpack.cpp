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


SymmetrizedUmfpack::SymmetrizedUmfpack():
	solver(),
	Amono(NULL),
	help(0)
{ }

SymmetrizedUmfpack::SymmetrizedUmfpack(SparseMatrix &A):
		solver(A),
		Amono(NULL),
		help(A.Size())
{
	help = 0.;
}

SymmetrizedUmfpack::SymmetrizedUmfpack(BlockMatrix &A):
		solver(NULL),
		Amono(A.CreateMonolithic()),
		help(Amono->Size())
{
	solver.SetOperator(*Amono);
}

void SymmetrizedUmfpack::SetOperator(const Operator &op)
{
    MFEM_VERIFY(op.Width() == op.Height(), "");
    height = width = op.Width();

	const SparseMatrix * A1 = dynamic_cast<const SparseMatrix *>(&op);
	if(A1)
	{
		solver.SetOperator(*A1);
		help.SetSize(A1->Size());
		help = 0.;
		return;
	}

	const BlockMatrix * A2 = dynamic_cast<const BlockMatrix*>(&op);
	if(A2)
	{
		delete Amono;
		Amono = A2->CreateMonolithic();
		solver.SetOperator(*Amono);
		help.SetSize(A2->Height());
		help = 0.;
		return;
	}

	mfem_error("");

}

void SymmetrizedUmfpack::Mult(const Vector &b, Vector &x) const
{
	solver.Mult(b,help);
	solver.MultTranspose(b, x);

	add(.5, help, .5, x, x);
}

void SymmetrizedUmfpack::MultTranspose(const Vector &b, Vector &x) const
{
	solver.Mult(b,help);
	solver.MultTranspose(b, x);

	add(.5, help, .5, x, x);
}

void SymmetrizedUmfpack::Mult(const MultiVector &b, MultiVector &x) const
{
	if(b.NumberOfVectors() != x.NumberOfVectors() )
		mfem_error("SymmetrizedUmfpack::Mult #1");

	int nrhs = b.NumberOfVectors();

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
	if(b.NumberOfVectors() != x.NumberOfVectors() )
		mfem_error("SymmetrizedUmfpack::MultTransose #1");

	int nrhs = b.NumberOfVectors();

	Vector bview, xview;
	for(int i(0); i < nrhs; ++i)
	{
		const_cast<MultiVector &>(b).GetVectorView(i, bview);
		x.GetVectorView(i, xview);
		this->MultTranspose(bview, xview);
	}
}

void SymmetrizedUmfpack::Mult(const DenseMatrix &b, DenseMatrix &x) const
{
    if(b.Width() != x.Width())
            mfem_error("SymmetrizedUmfpack::Mult #1");

    int nrhs = b.Width();
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
   if(b.Width() != x.Width())
            mfem_error("SymmetrizedUmfpack::MultTranspose #1");

    int nrhs = b.Width();
    Vector bview, xview;
    for(int i(0); i < nrhs; ++i)
    {
       const_cast<DenseMatrix &>(b).GetColumnReference(i, bview);
       x.GetColumnReference(i,xview);
       this->MultTranspose(bview, xview);
    }

}

SymmetrizedUmfpack::~SymmetrizedUmfpack()
{
	delete Amono;
}

