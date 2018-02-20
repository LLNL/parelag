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


#include "linalg/solver_ops/ParELAG_HiptmairSmoother.hpp"

#include "linalg/utilities/ParELAG_MG_Utils.hpp"

namespace parelag
{

HiptmairSmoother::HiptmairSmoother(
    std::shared_ptr<mfem::Operator> A,
    std::shared_ptr<mfem::Operator> A_Aux,
    std::shared_ptr<mfem::Operator> D_Op,
    std::shared_ptr<mfem::Solver> PrimarySolver,
    std::shared_ptr<mfem::Solver> AuxiliarySolver)
    : Solver{A->Height(),A->Width(),true},
      A_{std::move(A)},
      A_aux_{std::move(A_Aux)},
      D_op_{std::move(D_Op)},
      PrimarySolver_{std::move(PrimarySolver)},
      AuxiliarySolver_{std::move(AuxiliarySolver)}
{
    PARELAG_ASSERT(A_);
    PARELAG_ASSERT(D_op_);
    PARELAG_ASSERT(PrimarySolver_);
    PARELAG_ASSERT(AuxiliarySolver_);

    AuxB_ = make_unique<mfem::Vector>(D_op_->Width());
    AuxX_ = make_unique<mfem::Vector>(D_op_->Width());

    PrimaryVec_ = make_unique<mfem::Vector>(
        A_->Height()>A_->Width() ? A_->Height() : A_->Width() );
}


void HiptmairSmoother::Mult(const mfem::Vector& B,
                            mfem::Vector& X) const
{
    if (this->IsPreconditioner())
        X = 0.0;

    // Smoothing sweep on primary system
    PARELAG_ASSERT(PrimarySolver_->iterative_mode);
    PrimarySolver_->Mult(B,X);

    // Compute residual R = B - A*X
    PrimaryVec_ = mg_utils::ComputeResidual(*A_,X,B);

    // Restrict Defect to auxiliary space
    *AuxB_ = 0.0;// Just in case
    D_op_->MultTranspose(*PrimaryVec_,*AuxB_);// cB = D^T*Defect

    // Relax on A_aux*Correction_aux = D_aux
    PARELAG_ASSERT(not AuxiliarySolver_->iterative_mode);
    AuxiliarySolver_->Mult(*AuxB_,*AuxX_);

    // Aux_Correction = D*Correction_aux
    PrimaryVec_->SetSize(A_->Width());
    //*PrimaryVec_ = 0.0;// Just in case
    D_op_->Mult(*AuxX_,*PrimaryVec_);

    // X = X + Correction
    X += *PrimaryVec_;
}


void HiptmairSmoother::MultTranspose(const mfem::Vector& B,
                                     mfem::Vector& X) const
{
    if (this->IsPreconditioner())
        X = 0.0;

    // Restrict to Auxiliary space
    *AuxB_ = 0.0;// Just in case
    if (this->IsPreconditioner())
        D_op_->MultTranspose(B,*AuxB_);// cB = D^T*Defect
    else
    {
        PrimaryVec_ = mg_utils::ComputeResidual(*A_,X,B);
        D_op_->MultTranspose(*PrimaryVec_,*AuxB_);
    }

    // Relax on A_aux*x_aux = D_aux
    PARELAG_ASSERT(not AuxiliarySolver_->iterative_mode);
    AuxiliarySolver_->Mult(*AuxB_,*AuxX_);

    // Aux_Correction = D*Correction_aux
    PrimaryVec_->SetSize(A_->Width());// Just in case
    D_op_->Mult(*AuxX_,*PrimaryVec_);

    // X = X + Correction
    X += *PrimaryVec_;

    // Smoothing sweep on primary system
    PARELAG_ASSERT(PrimarySolver_->iterative_mode);
    PrimarySolver_->Mult(B,X);
}

}// namespace parelag
