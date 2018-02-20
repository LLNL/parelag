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


#include "ParELAG_Block2x2LDUInverseOperator.hpp"

#include "linalg/utilities/ParELAG_MG_Utils.hpp"

namespace parelag
{

Block2x2LDUInverseOperator::Block2x2LDUInverseOperator(
    std::shared_ptr<mfem::Operator> const& A,
    std::shared_ptr<mfem::Solver> invA00_1,
    std::shared_ptr<mfem::Solver> invA00_2,
    std::shared_ptr<mfem::Solver> invA00_3,
    std::shared_ptr<mfem::Solver> invS,
    std::shared_ptr<mfem::Operator> S,
    double DampingFactor)
    : Solver{A->Width(),A->Height(),true},
      invA00_1_{std::move(invA00_1)},
      invA00_2_{std::move(invA00_2)},
      invA00_3_{std::move(invA00_3)},
      invS_{std::move(invS)},
      S_{std::move(S)},
      DampingFactor_{DampingFactor}
{
    this->SetOperator(A);

    // Vector in the range space of A_
    Residual_ = make_unique<mfem::BlockVector>(
        mfem::Array<int>(
            const_cast<int*>(A_->ViewRowOffsets().data()),
            A_->ViewColumnOffsets().size()));

    // Vectors in the domain space of A_
    Correction_ = make_unique<mfem::BlockVector>(
        mfem::Array<int>(
            const_cast<int*>(A_->ViewColumnOffsets().data()),
            A_->ViewColumnOffsets().size()));
    Tmp_ = make_unique<mfem::BlockVector>(
        mfem::Array<int>(
            const_cast<int*>(A_->ViewColumnOffsets().data()),
            A_->ViewColumnOffsets().size()));
}


void Block2x2LDUInverseOperator::_do_set_operator(
    std::shared_ptr<mfem::Operator> const& A)
{
    auto newA = std::dynamic_pointer_cast<MfemBlockOperator>(A);
    PARELAG_TEST_FOR_EXCEPTION(
        !newA, std::logic_error,
        "Block2x2LDUInverseOperator::SetOperator(): "
        "Given operator is not a BlockOperator!");
    PARELAG_ASSERT(newA->GetNumBlockRows() == 2);
    PARELAG_ASSERT(newA->GetNumBlockCols() == 2);

    A_ = newA;
}


void Block2x2LDUInverseOperator::Mult(
    mfem::Vector const& rhs, mfem::Vector& sol) const
{
    mfem::Array<int>
        row_offsets(const_cast<int*>(A_->ViewRowOffsets().data()),
                    A_->ViewRowOffsets().size()),
        col_offsets(const_cast<int*>(A_->ViewRowOffsets().data()),
                    A_->ViewRowOffsets().size());

    mfem::BlockVector rhs_view(rhs.GetData(),row_offsets);
    mfem::BlockVector sol_view(sol.GetData(),col_offsets);

    if (this->IsPreconditioner())
        *Residual_ = rhs_view;// Ignore initial guess
    else
        Residual_ = mg_utils::ComputeResidual(*A_,sol_view,rhs_view);

    //
    // dp = S^{-1}(r_g - A10*\tilde{du})
    //

    // \tilde{du} = A_{2}^{-1}*r_f
    if (invA00_1_->iterative_mode)
        Tmp_->GetBlock(0) = 0.;
    invA00_2_->Mult(Residual_->GetBlock(0),Tmp_->GetBlock(0));

    // A10*\tilde{du}
    A_->GetBlock(1,0).Mult(Tmp_->GetBlock(0),Tmp_->GetBlock(1));
    Residual_->GetBlock(1) -= Tmp_->GetBlock(1);

    // dp = S^{-1}(r_g - A10*\tilde{du}
    if (invS_->iterative_mode)
        Correction_->GetBlock(1) = 0.;
    invS_->Mult(Residual_->GetBlock(1),Correction_->GetBlock(1));

    //
    // du = A_{1}^{-1}*r_f - A_{3}^{-1}*A01*dp
    //

    // A01*dp
    A_->GetBlock(0,1).Mult(Correction_->GetBlock(1),Correction_->GetBlock(0));

    // A_3^{-1}*A01*dp
    if (invA00_3_->iterative_mode)
        Tmp_->GetBlock(0) = 0.;
    invA00_3_->Mult(Correction_->GetBlock(0), Tmp_->GetBlock(0));

    // \hat{du} = A_{1}^{-1}*r_f
    if (invA00_1_->iterative_mode)
        Correction_->GetBlock(0) = 0.;
    invA00_1_->Mult(Residual_->GetBlock(0),Correction_->GetBlock(0));

    // du = \hat{du} - A_{3}^{-1}*A01*p
    Correction_->GetBlock(0) -= Tmp_->GetBlock(0);

    if (DampingFactor_ != 1.0)
        *Correction_ *= DampingFactor_;

    // [u; p] = [u; p] + [du; dp]
    if (this->IsPreconditioner())
        sol = *Correction_;
    else
        sol += *Correction_;
}


void Block2x2LDUInverseOperator::MultTranspose(
    mfem::Vector const&, mfem::Vector&) const
{
    PARELAG_NOT_IMPLEMENTED();
}


}// namespace parelag
