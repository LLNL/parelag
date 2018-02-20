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

#include "RankOneOperator.hpp"

#include "utilities/elagError.hpp"

namespace parelag
{
using namespace mfem;

RankOneOperator::RankOneOperator(MPI_Comm comm_, const Vector & u_, const Vector & v_):
    Operator(u_.Size(), v_.Size()),
    comm(comm_),
    u(&u_),
    v(&v_)
{

}

void RankOneOperator::Mult(const Vector & x, Vector & y) const
{
    elag_assert(x.Size() == width);
    elag_assert(y.Size() == height);
    double val = dot(*v,x);
    y = 0.;
    y.Add(val, *u);
}

void RankOneOperator::MultTranspose(const Vector & x, Vector & y) const
{
    elag_assert(x.Size() == height);
    elag_assert(y.Size() == width);
    double val = dot(*u,x);
    y = 0.;
    y.Add(val, *v);
}

RankOneOperator::~RankOneOperator()
{

}

double RankOneOperator::dot(const Vector & a, const Vector & b) const
{
    double local_dot = (a * b);
    double global_dot;

    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);

    return global_dot;
}

//----------------------------------------

RankOnePerturbation::RankOnePerturbation(MPI_Comm comm_, const Operator & A_, const Vector & u_, const Vector & v_):
    Operator(u_.Size(), v_.Size()),
    comm(comm_),
    A(&A_),
    u(&u_),
    v(&v_)
{
    elag_assert(height == A->Height());
    elag_assert(width == A->Width());
}

void RankOnePerturbation::Mult(const Vector & x, Vector & y) const
{
    elag_assert(x.Size() == width);
    elag_assert(y.Size() == height);
    double val = dot(*v,x);
    A->Mult(x,y);
    y.Add(val, *u);
}

void RankOnePerturbation::MultTranspose(const Vector & x, Vector & y) const
{
    elag_assert(x.Size() == height);
    elag_assert(y.Size() == width);
    double val = dot(*u,x);
    A->MultTranspose(x,y);
    y.Add(val, *v);
}

RankOnePerturbation::~RankOnePerturbation()
{

}

double RankOnePerturbation::dot(const Vector & a, const Vector & b) const
{
    double local_dot = (a * b);
    double global_dot;

    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);

    return global_dot;
}
}//namespace parelag
