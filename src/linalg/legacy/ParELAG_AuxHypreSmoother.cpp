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

#include "ParELAG_AuxHypreSmoother.hpp"

#include "utilities/elagError.hpp"

namespace parelag
{
using namespace mfem;

AuxHypreSmoother::AuxHypreSmoother(const HypreParMatrix &_A, const HypreParMatrix &_C, int type, int relax_times, double relax_weight, double omega, int poly_order, double poly_fraction):
    Solver(_A.Height()),
    A(&_A),
    C(&_C),
    CtAC(nullptr),
    S(nullptr),
    X(C->Width()),
    Y(C->Width())

{
    elag_assert( A->Height() == A->Width()  );
    elag_assert( A->Height() == C->Height() );

    CtAC = RAP( const_cast<HypreParMatrix *>(A), const_cast<HypreParMatrix *>(C) );
    hypre_ParCSRMatrixFixZeroRows(*CtAC);
    S = new HypreSmoother(*CtAC, type, relax_times, relax_weight, omega, poly_order, poly_fraction);
    S->iterative_mode = false;
}

void AuxHypreSmoother::SetOperator(const Operator & A_)
{
    elag_assert(A_.Height() == height);
    elag_assert(A_.Width() == width);

    A = dynamic_cast<const HypreParMatrix *>(&A_);
    elag_assert(A);

    delete S;
    delete CtAC;

    CtAC = RAP( const_cast<HypreParMatrix *>(A), const_cast<HypreParMatrix *>(C) );
    hypre_ParCSRMatrixFixZeroRows(*CtAC);
    S->SetOperator(*CtAC);
    S->iterative_mode = false;
}

void AuxHypreSmoother::Mult(const Vector & x, Vector & y) const
{
    if(iterative_mode)
    {
        res.SetSize(x.Size());
        A->Mult(y, res);
        add(1., x, -1., res, res);
        C->MultTranspose(x,X);
        S->Mult(X,Y);
        C->Mult(Y,res);
        y.Add(1., res);
    }
    else
    {
        C->MultTranspose(x,X);
        S->Mult(X,Y);
        C->Mult(Y,y);
    }
}

AuxHypreSmoother::~AuxHypreSmoother()
{
    delete S;
    delete CtAC;
}
}//namespace parelag
