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


#include "ParELAG_BlockTriangularSolver.hpp"

namespace parelag
{

BlockTriangularSolver::BlockTriangularSolver(
    std::shared_ptr<mfem::Operator> op, Triangle tri)
    : Solver{op->Width(),op->Height(),false},
      A_{std::dynamic_pointer_cast<MfemBlockOperator>(op)},
      tri_{tri}
{
    PARELAG_TEST_FOR_EXCEPTION(
        not A_,
        std::runtime_error,
        "BlockTriangularSolver::BlockTriangularSolver(...): "
        "Operator must be an MfemBlockOperator!");

    PARELAG_ASSERT(A_->GetNumBlockRows() == A_->GetNumBlockCols());
    inv_ops_.resize(A_->GetNumBlockRows());
}


BlockTriangularSolver::BlockTriangularSolver(
    std::shared_ptr<mfem::Operator> op,
    std::vector<std::shared_ptr<mfem::Solver>> inv_ops,
    std::vector<std::shared_ptr<mfem::Operator>> aux_ops, Triangle tri)
    : Solver{op->Width(),op->Height(),false},
      A_{std::dynamic_pointer_cast<MfemBlockOperator>(op)},
      inv_ops_{std::move(inv_ops)},
      aux_ops_{std::move(aux_ops)},
      tri_{tri}
{
    PARELAG_TEST_FOR_EXCEPTION(
        not A_,
        std::runtime_error,
        "BlockTriangularSolver::BlockTriangularSolver(...): "
        "Operator must be an MfemBlockOperator!");

    PARELAG_ASSERT(A_->GetNumBlockRows() == A_->GetNumBlockCols());
    inv_ops_.resize(A_->GetNumBlockRows());
}


void BlockTriangularSolver::_do_set_operator(
    const std::shared_ptr<mfem::Operator>& op)
{
    A_ = std::dynamic_pointer_cast<MfemBlockOperator>(op);

    PARELAG_TEST_FOR_EXCEPTION(
        not A_,
        std::runtime_error,
        "BlockTriangularSolver::SetOperator(...): "
        "Operator must be an MfemBlockOperator!");

    width = A_->Height();
    height = A_->Width();
}


void BlockTriangularSolver::Mult(const mfem::Vector& rhs, mfem::Vector& sol) const
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

    switch (tri_)
    {
    case (Triangle::UPPER_TRIANGLE):
        _do_upper_mult(*rhs_view,*sol_view);
        break;
    case (Triangle::LOWER_TRIANGLE):
        _do_lower_mult(*rhs_view,*sol_view);
        break;
    }

    // Apply correction
    if (not this->IsPreconditioner())
        sol += *sol_view;
}


void BlockTriangularSolver::MultTranspose(
    const mfem::Vector& rhs, mfem::Vector& sol) const
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

    switch (tri_)
    {
    case (Triangle::UPPER_TRIANGLE):
        _do_upper_mult_transp(*rhs_view,*sol_view);
        break;
    case (Triangle::LOWER_TRIANGLE):
        _do_lower_mult_transp(*rhs_view,*sol_view);
        break;
    }

    // Apply correction
    if (not this->IsPreconditioner())
        sol += *sol_view;
}


void BlockTriangularSolver::SetInverseDiagonals(
    std::vector<std::shared_ptr<mfem::Solver>> inv_ops)
{
    inv_ops_ = std::move(inv_ops);
}


void BlockTriangularSolver::SetInverseDiagonal(
    size_t block_id, std::shared_ptr<mfem::Solver> inv_op)
{
    inv_ops_[block_id] = std::move(inv_op);
}


void BlockTriangularSolver::_do_lower_mult(
    const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
{
    // Logically lower triangular, so count UP
    for (size_t block_i = 0; block_i < inv_ops_.size(); ++block_i)
    {
        // Deep copy the data
        if (block_i > 0)
        {
            tmp_rhs_ = rhs.GetBlock(block_i);
            tmp_.SetSize(tmp_rhs_.Size());
        }

        for (size_t block_j = 0; block_j < block_i; ++block_j)
        {
            A_->GetBlock(block_i,block_j).Mult(
                sol.GetBlock(block_j),tmp_);

            tmp_rhs_ -= tmp_;
        }

        PARELAG_ASSERT(not inv_ops_[block_i]->iterative_mode);

        if (block_i == 0)
            inv_ops_[block_i]->Mult(
                rhs.GetBlock(block_i), sol.GetBlock(block_i));
        else
            inv_ops_[block_i]->Mult(tmp_rhs_, sol.GetBlock(block_i));
    }
}


void BlockTriangularSolver::_do_lower_mult_transp(
    const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
{
    // Logically upper triangular, so count DOWN
    for (size_t block_i = inv_ops_.size(); block_i-- != 0;)
    {
        // Deep copy the data
        if (block_i < inv_ops_.size() - 1)
        {
            tmp_rhs_ = rhs.GetBlock(block_i);
            tmp_.SetSize(tmp_rhs_.Size());
        }

        for (size_t block_j = block_i+1; block_j < inv_ops_.size(); ++block_j)
        {
            A_->GetBlock(block_j,block_i).MultTranspose(
                sol.GetBlock(block_j),tmp_);

            tmp_rhs_ -= tmp_;
        }

        PARELAG_ASSERT(not inv_ops_[block_i]->iterative_mode);

        if (block_i == inv_ops_.size() - 1)
            inv_ops_[block_i]->MultTranspose(
                rhs.GetBlock(block_i), sol.GetBlock(block_i));
        else
            inv_ops_[block_i]->MultTranspose(tmp_rhs_, sol.GetBlock(block_i));
    }
}


void BlockTriangularSolver::_do_upper_mult(
    const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
{
    // Logically upper triangular, so count DOWN
    for (size_t block_i = inv_ops_.size(); block_i-- != 0;)
    {
        // Deep copy the data
        if (block_i < inv_ops_.size() - 1)
        {
            tmp_rhs_ = rhs.GetBlock(block_i);
            tmp_.SetSize(tmp_rhs_.Size());
        }

        for (size_t block_j = block_i+1; block_j < inv_ops_.size(); ++block_j)
        {
            A_->GetBlock(block_i,block_j).Mult(
                sol.GetBlock(block_j),tmp_);

            tmp_rhs_ -= tmp_;
        }

        if (block_i == inv_ops_.size() - 1)
            inv_ops_[block_i]->Mult(
                rhs.GetBlock(block_i), sol.GetBlock(block_i));
        else
            inv_ops_[block_i]->Mult(tmp_rhs_, sol.GetBlock(block_i));
    }
}


void BlockTriangularSolver::_do_upper_mult_transp(
    const mfem::BlockVector& rhs, mfem::BlockVector& sol) const
{
    // Logically lower triangular, so count up
    for (size_t block_i = 0; block_i < inv_ops_.size(); ++block_i)
    {
        // Deep copy the data
        if (block_i > 0)
        {
            tmp_rhs_ = rhs.GetBlock(block_i);
            tmp_.SetSize(tmp_rhs_.Size());
        }

        for (size_t block_j = 0; block_j < block_i; ++block_j)
        {
            A_->GetBlock(block_j,block_i).MultTranspose(
                sol.GetBlock(block_j),tmp_);

            tmp_rhs_ -= tmp_;
        }

        if (block_i == 0)
            inv_ops_[block_i]->MultTranspose(
                rhs.GetBlock(block_i), sol.GetBlock(block_i));
        else
            inv_ops_[block_i]->MultTranspose(tmp_rhs_, sol.GetBlock(block_i));
    }

}

}// namespace parelag
