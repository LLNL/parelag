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

TwoLevelAdditiveSchwarz::TwoLevelAdditiveSchwarz(
        ParallelCSRMatrix& op,
        const mfem::Array<mfem::Array<int> >& local_dofs,
        const SerialCSRMatrix& coarse_map)
    : Solver(op.NumRows(), false),
      local_dofs_(local_dofs),
      coarse_map_(coarse_map),
      local_ops_(local_dofs.Size()),
      local_solvers_(local_dofs.Size())
{
    // Set up local solvers
    SerialCSRMatrix op_diag;
    op.GetDiag(op_diag);
    for (int i = 0; i < local_dofs.Size(); ++i)
    {
        local_ops_[i].SetSize(local_dofs[i].Size());
        op_diag.GetSubMatrix(local_dofs[i], local_dofs[i], local_ops_[i]);
        local_solvers_[i].Compute(local_ops_[i]);
    }

    // Set up coarse solver
    int num_local_cdofs = coarse_map.NumCols();
    mfem::Array<int> cdof_starts;
    ParPartialSums_AssumedPartitionCheck(op.GetComm(), num_local_cdofs, cdof_starts);
    int num_global_cdofs = cdof_starts.Last();
    SerialCSRMatrix c_map(coarse_map);
    ParallelCSRMatrix parallel_c_map(op.GetComm(), op.N(), num_global_cdofs,
                                     op.ColPart(), cdof_starts, &c_map);

    coarse_op_.reset(RAP(&op, &parallel_c_map));
    coarse_solver_ = make_unique<mfem::HypreBoomerAMG>(*coarse_op_);
    coarse_solver_->SetPrintLevel(-1);
}

void TwoLevelAdditiveSchwarz::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    mfem::Vector x_local, y_local, x_coarse, y_coarse;
    for (int i = 0; i < local_dofs_.Size(); ++i)
    {
        x.GetSubVector(local_dofs_[i], x_local);
        y_local.SetSize(x_local.Size());
        y_local = 0.0;
        local_solvers_[i].Mult(x_local, y_local);
        y.SetSubVector(local_dofs_[i], y_local);
    }

    x_coarse.SetSize(coarse_map_.NumCols());
    y_coarse.SetSize(coarse_map_.NumCols());
    x_coarse = 0.0;
    y_coarse = 0.0;
    coarse_map_.MultTranspose(x, x_coarse);
    coarse_solver_->Mult(x_coarse, y_coarse);
    coarse_map_.AddMultTranspose(y_coarse, y);
}


}// namespace parelag
