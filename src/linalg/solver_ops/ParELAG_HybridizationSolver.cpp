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

#include "ParELAG_HybridizationSolver.hpp"

namespace parelag
{

HybridizationSolver::HybridizationSolver(std::shared_ptr<HybridHdivL2> hybridization,
    std::shared_ptr<mfem::Solver> solver,
    std::shared_ptr<DeRhamSequence> sequence,
    std::shared_ptr<mfem::SparseMatrix> D_Scale,
    bool act_on_trueDofs)
    : hybridization_(std::move(hybridization)),
      Solver_(std::move(solver)),
      act_on_trueDofs_(act_on_trueDofs),
      sequence_(sequence),
      Offsets_(3),
      TrueOffsets_(3),
      D_Scale_(std::move(D_Scale))
{
    int nTrueDofs = hybridization_->
            GetDofMultiplier().GetDofTrueDof().GetTrueLocalSize();
    pHybridRHS_.SetSize(nTrueDofs);
    pHybridRHS_ = 0.0;
    pHybridSol_.SetSize(nTrueDofs);
    pHybridSol_ = 0.0;

    const int l2_form = sequence->GetNumForms()-1;
    const int hdiv_form = l2_form-1;
    Offsets_[0] = 0;
    Offsets_[1] = sequence->GetNumDofs(hdiv_form);
    Offsets_[2] = Offsets_[1] + sequence->GetNumDofs(l2_form);
    TrueOffsets_[0] = 0;
    TrueOffsets_[1] = sequence->GetNumTrueDofs(hdiv_form);
    TrueOffsets_[2] = TrueOffsets_[1] + sequence->GetNumTrueDofs(l2_form);
}

void HybridizationSolver::_do_set_operator(const std::shared_ptr<mfem::Operator>& op)
{
    auto tmp = std::dynamic_pointer_cast<Solver>(Solver_);
    PARELAG_ASSERT(tmp);

    tmp->SetOperator(op);
}

void HybridizationSolver::Mult(
    const mfem::Vector& rhs, mfem::Vector& sol) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        not this->IsPreconditioner(),
        std::runtime_error,
        "HybridizationSolver::Mult(...):\n"
        "The HybridizationSolver cannot be used in iterative mode.");

    const int hdiv_form = sequence_->GetNumForms()-2;
    auto& hdiv_dofTrueDof = sequence_->GetDofHandler(hdiv_form)->GetDofTrueDof();

    mfem::Vector non_true_rhs;
    mfem::Vector non_true_sol;
    mfem::BlockVector rhs_view;
    mfem::BlockVector sol_view;
    mfem::BlockVector true_rhs_view;
    mfem::BlockVector true_sol_view;

    if (act_on_trueDofs_)
    {
       true_rhs_view.Update(rhs.GetData(), TrueOffsets_);
       true_sol_view.Update(sol.GetData(), TrueOffsets_);

       non_true_rhs.SetSize(Offsets_.Last());
       non_true_sol.SetSize(Offsets_.Last());
       non_true_sol = 0.0;
       rhs_view.Update(non_true_rhs.GetData(), Offsets_);
       sol_view.Update(non_true_sol.GetData(), Offsets_);

       SerialCSRMatrix hdiv_assign;
       hdiv_dofTrueDof.get_entity_trueEntity()->GetDiag(hdiv_assign);

       hdiv_assign.Mult(true_rhs_view.GetBlock(0), rhs_view.GetBlock(0));
       rhs_view.GetBlock(1) = true_rhs_view.GetBlock(1);
    }
    else
    {
        rhs_view.Update(rhs.GetData(), Offsets_);
        sol_view.Update(sol.GetData(), Offsets_);
    }

    auto& mult_dofTrueDof = hybridization_->GetDofMultiplier().GetDofTrueDof();

    // Transform RHS to the hybridized form and essential data
    mfem::Vector HybridRHS;
    mfem::Vector essentialData(mult_dofTrueDof.GetLocalSize());
    essentialData = 0.;
    hybridization_->RHSTransform(rhs_view,HybridRHS,essentialData);
    PARELAG_ASSERT(!HybridRHS.CheckFinite());

    // Eliminate essential condition for Lagrange multipliers
    mfem::SparseMatrix HB_mat_copy(hybridization_->GetHybridSystem(), false);
    for (int i = 0; i < HB_mat_copy.Size(); i++)
        if (hybridization_->GetEssentialMultiplierDofs()[i])
            HB_mat_copy.EliminateRowCol(i,essentialData(i),HybridRHS);
    mult_dofTrueDof.Assemble(HybridRHS,pHybridRHS_);

    // Change the scaling of the RHS
    mfem::Vector Scaled_pHybridRHS(pHybridRHS_.Size());
    Scaled_pHybridRHS = 0.0;
    D_Scale_->Mult(pHybridRHS_, Scaled_pHybridRHS);

    mfem::Vector Scaled_pHybridSol(pHybridSol_.Size());

    // Solve the scaled hybridized system
    if (this->IsPreconditioner() && Solver_->iterative_mode)
        Scaled_pHybridSol = 0.0;
    Solver_->Mult(Scaled_pHybridRHS,Scaled_pHybridSol);

    // Change the scaling of the solution
    D_Scale_->Mult(Scaled_pHybridSol, pHybridSol_);

    mfem::Vector HybridSol(mult_dofTrueDof.GetLocalSize());
    mult_dofTrueDof.Distribute(pHybridSol_, HybridSol);
    // Transform back to the non-hybridized form
    hybridization_->RecoverOriginalSolution(HybridSol,sol_view);

    if (act_on_trueDofs_)
    {
       hdiv_dofTrueDof.IgnoreNonLocal(sol_view.GetBlock(0),
                                      true_sol_view.GetBlock(0));
       true_sol_view.GetBlock(1) = sol_view.GetBlock(1);
    }
}

void HybridizationSolver::MultTranspose(
    const mfem::Vector& rhs, mfem::Vector& sol) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        not this->IsPreconditioner(),
        std::runtime_error,
        "HybridizationSolver::Mult(...):\n"
        "The HybridizationSolver cannot be used in iterative mode.");

    const int hdiv_form = sequence_->GetNumForms()-2;
    auto& hdiv_dofTrueDof = sequence_->GetDofHandler(hdiv_form)->GetDofTrueDof();

    mfem::Vector non_true_rhs;
    mfem::Vector non_true_sol;
    mfem::BlockVector rhs_view;
    mfem::BlockVector sol_view;
    mfem::BlockVector true_rhs_view;
    mfem::BlockVector true_sol_view;

    if (act_on_trueDofs_)
    {
       true_rhs_view.Update(rhs.GetData(), TrueOffsets_);
       true_sol_view.Update(sol.GetData(), TrueOffsets_);

       non_true_rhs.SetSize(Offsets_.Last());
       non_true_sol.SetSize(Offsets_.Last());
       non_true_sol = 0.0;
       rhs_view.Update(non_true_rhs.GetData(), Offsets_);
       sol_view.Update(non_true_sol.GetData(), Offsets_);

       SerialCSRMatrix hdiv_assign;
       hdiv_dofTrueDof.get_entity_trueEntity()->GetDiag(hdiv_assign);

       hdiv_assign.Mult(true_rhs_view.GetBlock(0), rhs_view.GetBlock(0));
       rhs_view.GetBlock(1) = true_rhs_view.GetBlock(1);
    }
    else
    {
        rhs_view.Update(rhs.GetData(), Offsets_);
        sol_view.Update(sol.GetData(), Offsets_);
    }
    auto& mult_dofTrueDof = hybridization_->GetDofMultiplier().GetDofTrueDof();

    // Transform RHS to the hybridized form and essential data
    mfem::Vector HybridRHS;
    mfem::Vector essentialData(mult_dofTrueDof.GetLocalSize());
    essentialData = 0.;
    hybridization_->RHSTransform(rhs_view,HybridRHS,essentialData);
    PARELAG_ASSERT(!HybridRHS.CheckFinite());

    // Eliminate essential condition for Lagrange multipliers
    mfem::SparseMatrix HB_mat_copy(hybridization_->GetHybridSystem(), false);
    for (int i = 0; i < HB_mat_copy.Size(); i++)
        if (hybridization_->GetEssentialMultiplierDofs()[i])
            HB_mat_copy.EliminateRowCol(i,essentialData(i),HybridRHS);
    mult_dofTrueDof.Assemble(HybridRHS,pHybridRHS_);

    // Change the scaling of the RHS
    mfem::Vector Scaled_pHybridRHS(pHybridRHS_.Size());
    Scaled_pHybridRHS = 0.0;
    D_Scale_->Mult(pHybridRHS_, Scaled_pHybridRHS);

    mfem::Vector Scaled_pHybridSol(pHybridSol_.Size());

    // Solve the scaled hybridized system
    if (this->IsPreconditioner() && Solver_->iterative_mode)
        Scaled_pHybridSol = 0.0;
    Solver_->MultTranspose(Scaled_pHybridRHS,Scaled_pHybridSol);

    // Change the scaling of the solution
    D_Scale_->Mult(Scaled_pHybridSol, pHybridSol_);

    mfem::Vector HybridSol(mult_dofTrueDof.GetLocalSize());
    mult_dofTrueDof.Distribute(pHybridSol_, HybridSol);
    // Transform back to the non-hybridized form
    hybridization_->RecoverOriginalSolution(HybridSol,sol_view);

    if (act_on_trueDofs_)
    {
       hdiv_dofTrueDof.IgnoreNonLocal(sol_view.GetBlock(0),
                                      true_sol_view.GetBlock(0));
       true_sol_view.GetBlock(1) = sol_view.GetBlock(1);
    }
}


}// namespace parelag
