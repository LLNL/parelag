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


#include "ParELAG_BlockDiagonalSolver.hpp"

namespace parelag
{

BlockDiagonalSolver::BlockDiagonalSolver(
    std::shared_ptr<mfem::Operator> op)
    : Solver{op->Width(),op->Height(),false},
      A_{std::dynamic_pointer_cast<MfemBlockOperator>(op)}
{
    PARELAG_TEST_FOR_EXCEPTION(
        not A_,
        std::runtime_error,
        "BlockDiagonalSolver::BlockDiagonalSolver(...): "
        "Operator must be an MfemBlockOperator!");
}


BlockDiagonalSolver::BlockDiagonalSolver(
    std::shared_ptr<mfem::Operator> op,
    std::vector<std::shared_ptr<mfem::Solver>> inv_ops,
    std::vector<std::shared_ptr<mfem::Operator>> aux_ops)
    : Solver{op->Width(),op->Height(),false},
      A_{std::dynamic_pointer_cast<MfemBlockOperator>(op)},
      inv_ops_{std::move(inv_ops)},
      aux_ops_{std::move(aux_ops)}
{
    PARELAG_TEST_FOR_EXCEPTION(
        not A_,
        std::runtime_error,
        "BlockDiagonalSolver::BlockDiagonalSolver(...): "
        "Operator must be an MfemBlockOperator!");
}


void BlockDiagonalSolver::_do_set_operator(
    const std::shared_ptr<mfem::Operator>& op)
{
    A_ = std::dynamic_pointer_cast<MfemBlockOperator>(op);

    width = A_->Height();
    height = A_->Width();

    PARELAG_TEST_FOR_EXCEPTION(
        not A_,
        std::runtime_error,
        "BlockDiagonalSolver::SetOperator(...): "
        "Operator must be an MfemBlockOperator!");
}


void BlockDiagonalSolver::Mult(const mfem::Vector& rhs, mfem::Vector& sol) const
{
    mfem::Array<int>
        row_offsets(const_cast<int*>(A_->ViewRowOffsets().data()),
                    A_->ViewRowOffsets().size()),
        col_offsets(const_cast<int*>(A_->ViewRowOffsets().data()),
                    A_->ViewRowOffsets().size());

    std::unique_ptr<mfem::BlockVector> rhs_view;
    std::unique_ptr<mfem::BlockVector> sol_view;

    // Compute the residual
    if (this->IsPreconditioner())
    {
        // Resid = rhs; sol = 0.0
        rhs_view = make_unique<mfem::BlockVector>(rhs.GetData(),row_offsets);
        sol_view = make_unique<mfem::BlockVector>(sol.GetData(),col_offsets);
    }
    else
    {
        // Resid = rhs - A_*sol; sol = sol
        rhs_view = make_unique<mfem::BlockVector>(row_offsets);
        sol_view = make_unique<mfem::BlockVector>(col_offsets);

        A_->Mult(sol,*rhs_view);
        *rhs_view *= -1.;
        *rhs_view += rhs;
    }

    *sol_view = 0.0;

    // Update the solution block-by-block
    for (size_t block_id = 0; block_id < inv_ops_.size(); ++block_id)
    {
        PARELAG_ASSERT_DEBUG(not inv_ops_[block_id]->iterative_mode);

        // X_I = X_I + M_I^{-1}(Resid)
        inv_ops_[block_id]->Mult(rhs_view->GetBlock(block_id),
                                 sol_view->GetBlock(block_id));
    }

    // Apply correction
    if (not this->IsPreconditioner())
        sol += *sol_view;
}


void BlockDiagonalSolver::MultTranspose(
    const mfem::Vector& rhs, mfem::Vector& sol) const
{
    mfem::Array<int>
        row_offsets(const_cast<int*>(A_->ViewRowOffsets().data()),
                    A_->ViewRowOffsets().size()),
        col_offsets(const_cast<int*>(A_->ViewRowOffsets().data()),
                    A_->ViewRowOffsets().size());

    std::unique_ptr<mfem::BlockVector> rhs_view;
    std::unique_ptr<mfem::BlockVector> sol_view;

    if (this->IsPreconditioner())
    {
        sol_view = make_unique<mfem::BlockVector>(sol.GetData(),col_offsets);
        rhs_view = make_unique<mfem::BlockVector>(rhs.GetData(),row_offsets);
    }
    else
    {
        sol_view = make_unique<mfem::BlockVector>(col_offsets);
        rhs_view = make_unique<mfem::BlockVector>(row_offsets);
        A_->Mult(sol,*rhs_view);
        *rhs_view *= -1.;
        *rhs_view += rhs;
    }

    *sol_view = 0.0;

    for (size_t block_id = 0; block_id < inv_ops_.size(); ++block_id)
    {
        inv_ops_[block_id]->MultTranspose(rhs_view->GetBlock(block_id),
                                          sol_view->GetBlock(block_id));
    }

    if (not this->IsPreconditioner())
        sol += *sol_view;
}


void BlockDiagonalSolver::SetInverseDiagonals(
    std::vector<std::shared_ptr<mfem::Solver>> inv_ops)
{
    inv_ops_ = std::move(inv_ops);
}


void BlockDiagonalSolver::SetInverseDiagonal(
    size_t block_id, std::shared_ptr<mfem::Solver> inv_op)
{
    if (inv_ops_.size() <= block_id)
        inv_ops_.resize(block_id+1);
    inv_ops_[block_id] = std::move(inv_op);
}

}// namespace parelag
