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


#include "linalg/solver_ops/ParELAG_BramblePasciakSolver.hpp"

#include "utilities/MemoryUtils.hpp"

namespace parelag
{

BramblePasciakSolver::BramblePasciakSolver(
    std::shared_ptr<mfem::Operator> transform_op,
    std::shared_ptr<mfem::Solver> solver)
    : Solver{solver->Height(),solver->Width(),false},
      TransformationOp_{std::move(transform_op)},
      Solver_{std::move(solver)},
      TransformedRHS_{TransformationOp_->Height()}
{
    TransformedRHS_ = 0.0;
}


void BramblePasciakSolver::Mult(
    mfem::Vector const& rhs, mfem::Vector& sol) const
{
    TransformationOp_->Mult(rhs,TransformedRHS_);

    Solver_->Mult(TransformedRHS_,sol);
}


void BramblePasciakSolver::MultTranspose(
    mfem::Vector const& rhs, mfem::Vector& sol) const
{
    TransformationOp_->MultTranspose(rhs,TransformedRHS_);

    Solver_->MultTranspose(TransformedRHS_,sol);
}


void BramblePasciakSolver::SetTransformationOperator(
    std::shared_ptr<mfem::Operator> transform_op)
{
    PARELAG_ASSERT(transform_op);
    TransformationOp_ = std::move(transform_op);
}


std::shared_ptr<mfem::Operator>
BramblePasciakSolver::GetTransformationOperator() const noexcept
{
    return TransformationOp_;
}


void BramblePasciakSolver::_do_set_operator(
    const std::shared_ptr<mfem::Operator>& op)
{
    auto tmp = std::dynamic_pointer_cast<Solver>(Solver_);
    PARELAG_ASSERT(tmp);

    tmp->SetOperator(op);
}

}// namespace parelag
