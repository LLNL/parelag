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

/*
 * MLHiptmairSolver.cpp
 *
 *  Created on: Apr 17, 2015
 *      Author: uvilla
 */

#include "linalg/legacy/ParELAG_MLHiptmairSolver.hpp"

namespace parelag
{
using namespace mfem;

int HdivProblem3D::form(2);
int HdivProblem2D::form(1);
int HcurlProblem::form(1);

HypreSmootherData::HypreSmootherData():
    type(HypreSmoother::l1GS),
    relax_times(1),
    relax_weight(1.0),
    omega(1.0),
    poly_order(2),
    poly_fraction(.3)
{
}

template<class PROBLEM>
MLHiptmairSolver<PROBLEM>::MLHiptmairSolver(Array<DeRhamSequence *> & seqs,
                                            Array<int> & label_ess):
    seqs(seqs),
    label_ess(label_ess.GetData(), label_ess.Size()),
    nLevels(seqs.Size()),
    A(nLevels, nullptr),
    Aaux(nLevels-1),
    D(nLevels-1),
    P(nLevels-1),
    SA(nLevels-1),
    Saux(nLevels-1),
    coarseSolver(nullptr),
    v(nLevels),
    d(nLevels),
    t(nLevels),
    arithmeticComplexity(0.),
    operatorComplexity(0.)
{
    SA = nullptr;
    Saux = nullptr;
    v = nullptr;
    d = nullptr;
    t = nullptr;

    for(int i = 0; i < nLevels-1; ++i)
    {
        P[i] = seqs[i]->ComputeTrueP(Problem::form, label_ess);
        D[i] = seqs[i]->ComputeTrueD(Problem::form-1, label_ess);
    }

    for(int i = 0; i < nLevels; ++i)
    {
        d[i] = new Vector(seqs[i]->GetNumberOfTrueDofs(Problem::form));
        t[i] = new Vector(seqs[i]->GetNumberOfTrueDofs(Problem::form));
        *d[i] = 0.0;
        *t[i] = 0.0;
    }

    for(int i = 1; i < nLevels; ++i)
    {
        v[i] = new Vector(seqs[i]->GetNumberOfTrueDofs(Problem::form));
        *v[i] = 0.0;
    }

}

template<class PROBLEM>
MLHiptmairSolver<PROBLEM>::~MLHiptmairSolver()
{
    for(int i = 0; i < nLevels; ++i)
    {
        delete v[i];
        delete d[i];
        delete t[i];
    }

    delete coarseSolver;

    for(int i = 0; i < nLevels-1; ++i)
    {
        delete SA[i];
        delete Saux[i];
    }

    for(int i = 1; i < nLevels; ++i)
        delete A[i];
}

template<class PROBLEM>
void MLHiptmairSolver<PROBLEM>::SetOperator(const Operator & A)
{
    const HypreParMatrix * Am = dynamic_cast<const HypreParMatrix *>(&A);
    elag_assert(Am);
    SetMatrix(const_cast<HypreParMatrix *>(Am));
}

template<class PROBLEM>
void MLHiptmairSolver<PROBLEM>::SetMatrix(HypreParMatrix * Afine)
{
    A[0] = Afine;
    compute();
}

template<class PROBLEM>
void MLHiptmairSolver<PROBLEM>::SetHypreSmootherProperties(HypreSmootherData & data)
{
    Sdata = data;
}

template<class PROBLEM>
void MLHiptmairSolver<PROBLEM>::SetCoarseSolverProperties(CoarseSolverData & data)
{
    coarseSolverData = data;
}

template<class PROBLEM>
void MLHiptmairSolver<PROBLEM>::Mult (const Vector & x, Vector & y) const
{
    if(iterative_mode)
        mfem_error("MLPreconditioner does not support iterative_mode");

    if(nLevels == 1)
    {
        coarseSolver->Mult(x,y);
    }
    else
    {
        int level = 0;
        *d[level] = x;
        v[level] = &y;
        MGCycle(level);
        v[level] = NULL;
    }
}

template<class PROBLEM>
void MLHiptmairSolver<PROBLEM>::compute()
{
    cleanUp();

    for(int i = 0; i < nLevels-1; ++i)
    {
        Aaux[i].reset(RAP(A[i], D[i].get()));
        // canonically A[i] would be a mass matrix, but it doesn't
        // matter because the range of D[i] is in the nullspace of the
        // derivative part of A[i] so all that's left is the mass
        // matrix components anyway
        hypre_ParCSRMatrixFixZeroRows( *(Aaux[i]) );
        A[i+1] = RAP(A[i], P[i].get());
    }

    for(int i = 1; i < nLevels; ++i)
        hypre_ParCSRMatrixFixZeroRows( *(A[i]) );

    for(int i = 0; i < nLevels-1; ++i)
    {
        SA[i] = new HypreSmoother(*(A[i]), Sdata.type, Sdata.relax_times,
                                  Sdata.relax_weight, Sdata.omega,
                                  Sdata.poly_order, Sdata.poly_fraction);

        Saux[i] = new HypreSmoother(*(Aaux[i]), Sdata.type, Sdata.relax_times,
                                    Sdata.relax_weight, Sdata.omega,
                                    Sdata.poly_order, Sdata.poly_fraction);
    }

    coarseSolver = new CoarseSolver(*(A.back()), seqs.Last(), coarseSolverData);
}

template<class PROBLEM>
void MLHiptmairSolver<PROBLEM>::MGCycle(int level) const
{
    Vector& tmp = *t[level];
    tmp = *d[level];
    // Recursive multigrid algorithm

    // PreSmooth
    presmoothing(level, *d[level],*v[level]);

    // Compute the defect: d = r - A_k x_k
    A[level]->Mult(-1.0, *v[level], 1.0, tmp);

    // Restrict the defect
    P[level]->MultTranspose(tmp,*d[level+1]);

    if(level == nLevels-2) // Solve coarse problem
        coarseSolver->Mult(*d[level+1],*v[level+1]);
    else // Recursive call
        this->MGCycle(level+1);

    // Prolongate the correction and correct the approximation
    P[level]->Mult(1.0, *v[level+1], 1.0, *v[level]);

    // PostSmooth
    A[level]->Mult(-1.0, *v[level],1.0, *d[level]);
    postsmoothing(level, *d[level],tmp);
    *v[level] += tmp;
}

// ATB I believe Umberto is doing SA and Saux in opposite order from PSV Algorithm F1.1
template<class PROBLEM>
void MLHiptmairSolver<PROBLEM>::presmoothing(int level, const Vector & res, Vector & sol) const
{
    SA[level]->Mult(res,sol);
    res1.SetSize(res.Size());
    res1 = res;
    A[level]->Mult(-1.0, sol, 1.0, res1);
    resaux.SetSize(Aaux[level]->Height());
    solaux.SetSize(Aaux[level]->Height());
    D[level]->MultTranspose(res1, resaux);
    Saux[level]->Mult(resaux, solaux);
    D[level]->Mult(1.0, solaux, 1.0, sol);
}

template<class PROBLEM>
void MLHiptmairSolver<PROBLEM>::postsmoothing(int level, const Vector & res, Vector & sol) const
{
    resaux.SetSize(Aaux[level]->Height());
    solaux.SetSize(Aaux[level]->Height());
    D[level]->MultTranspose(res, resaux);
    Saux[level]->Mult(resaux, solaux);
    D[level]->Mult(solaux, sol);
    res1.SetSize(res.Size());
    sol1.SetSize(res.Size());
    res1 = res;
    A[level]->Mult(-1.0, sol, 1.0, res1);
    SA[level]->Mult(res1, sol1);
    sol += sol1;
}

// what distinguished this from the destructor?
template<class PROBLEM>
void MLHiptmairSolver<PROBLEM>::cleanUp()
{
    delete coarseSolver;
    coarseSolver = nullptr;

    for(int i = 0; i < nLevels-1; ++i)
    {
        delete SA[i];   SA[i] = nullptr;
        delete Saux[i]; Saux[i] = nullptr;
        Aaux[i].reset();
    }

    for(int i = 1; i < nLevels; ++i)
    {
        delete A[i];
        A[i] = nullptr;
    }
}

template class MLHiptmairSolver<HdivProblem3D>;
template class MLHiptmairSolver<HdivProblem2D>;
template class MLHiptmairSolver<HcurlProblem>;
}//namespace parelag
