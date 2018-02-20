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

HybridizationSolver::HybridizationSolver(
    std::shared_ptr<HybridHdivL2> Hybridization,
    std::shared_ptr<mfem::Solver> solver,
    mfem::Array<int>& Offsets,
    std::shared_ptr<mfem::SparseMatrix> D_Scale)
    : Hybridization_(std::move(Hybridization)),
      Solver_(std::move(solver)),
      D_Scale_(std::move(D_Scale))
{
    int nTrueDofs = Hybridization_->
            GetDofMultiplier()->GetDofTrueDof().GetTrueLocalSize();
    pHybridRHS_.SetSize(nTrueDofs);
    pHybridRHS_ = 0.0;
    pHybridSol_.SetSize(nTrueDofs);
    pHybridSol_ = 0.0;

    Offsets_.SetSize(Offsets.Size());
    for (int i = 0; i < Offsets.Size(); i++)
    	Offsets_[i] = Offsets[i];
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

    mfem::BlockVector rhs_view(rhs.GetData(), Offsets_);
    mfem::BlockVector sol_view(sol.GetData(), Offsets_);

	SharingMap& mult_dofTrueDof =
			Hybridization_->GetDofMultiplier()->GetDofTrueDof();

    // Transform RHS to the hybridized form and essential data
    mfem::Vector HybridRHS;
    mfem::Vector essentialData(mult_dofTrueDof.GetLocalSize());
    essentialData = 0.;
    Hybridization_->RHSTransform(rhs_view,HybridRHS,essentialData);
    PARELAG_ASSERT(!HybridRHS.CheckFinite());

    // Eliminate essential condition for Lagrange multipliers
    mfem::SparseMatrix HB_mat_copy(*Hybridization_->GetHybridSystem(), false);
    for (int i = 0; i < HB_mat_copy.Size(); i++)
        if (Hybridization_->GetEssentialMultiplierDofs()[i])
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
    Hybridization_->RecoverOriginalSolution(HybridSol,sol_view);
}

void HybridizationSolver::MultTranspose(
    const mfem::Vector& rhs, mfem::Vector& sol) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        not this->IsPreconditioner(),
        std::runtime_error,
        "HybridizationSolver::Mult(...):\n"
        "The HybridizationSolver cannot be used in iterative mode.");

    mfem::BlockVector rhs_view(rhs.GetData(), Offsets_);
    mfem::BlockVector sol_view(sol.GetData(), Offsets_);

    SharingMap& mult_dofTrueDof =
            Hybridization_->GetDofMultiplier()->GetDofTrueDof();

    // Transform RHS to the hybridized form and essential data
    mfem::Vector HybridRHS;
    mfem::Vector essentialData(mult_dofTrueDof.GetLocalSize());
    essentialData = 0.;
    Hybridization_->RHSTransform(rhs_view,HybridRHS,essentialData);
    PARELAG_ASSERT(!HybridRHS.CheckFinite());

    // Eliminate essential condition for Lagrange multipliers
    mfem::SparseMatrix HB_mat_copy(*Hybridization_->GetHybridSystem(), false);
    for (int i = 0; i < HB_mat_copy.Size(); i++)
        if (Hybridization_->GetEssentialMultiplierDofs()[i])
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
    Hybridization_->RecoverOriginalSolution(HybridSol,sol_view);
}


}// namespace parelag
