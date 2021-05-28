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


#include "linalg/factories/ParELAG_HiptmairSmootherFactory.hpp"

#include "amge/DeRhamSequence.hpp"
#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "linalg/solver_ops/ParELAG_HiptmairSmoother.hpp"
#include "utilities/MemoryUtils.hpp"
#include "utilities/ParELAG_TimeManager.hpp"

namespace parelag
{

void HiptmairSmootherFactory::_do_set_default_parameters()
{
    auto& params = GetParameters();
    params.Get("Primary Smoother", "Default Hypre");
    params.Get("Auxiliary Smoother", "Default Hypre");
}


void HiptmairSmootherFactory::_do_initialize(const ParameterList&)
{
    // Nested preconditioner, so I assert that the Lib_ member is non-null
    PARELAG_ASSERT(HasValidSolverLibrary());

    // Build the primary smoother
    std::string primary_smoother_name =
        GetParameters().Get("Primary Smoother", "Default Hypre");

    PrimaryFact_ = GetSolverLibrary().GetSolverFactory(primary_smoother_name);

    std::string aux_smoother_name =
        GetParameters().Get("Auxiliary Smoother", "Default Hypre");

    AuxiliaryFact_ = GetSolverLibrary().GetSolverFactory(aux_smoother_name);
}


std::unique_ptr<mfem::Solver>
HiptmairSmootherFactory::_do_build_solver(
    const std::shared_ptr<mfem::Operator>& op,
    SolverState& state ) const
{
    auto my_state = dynamic_cast<NestedSolverState *>(&state);
    PARELAG_ASSERT(my_state);

    // Get the things I need from my state:
    auto primary_state = std::shared_ptr<SolverState>{
        PrimaryFact_->GetDefaultState()};
    auto aux_state = std::shared_ptr<SolverState>{
        AuxiliaryFact_->GetDefaultState()};

    if (my_state->IsSubState("Primary"))
        primary_state->MergeState(*my_state->GetSubState("Primary"));
    if (my_state->IsSubState("Auxiliary"))
        aux_state->MergeState(*my_state->GetSubState("Auxiliary"));

    primary_state->MergeState(*my_state);
    aux_state->MergeState(*my_state);

    // Aux_state has form lower than primary_state!
    std::vector<int> aux_forms;
    std::transform(primary_state->GetForms().begin(),
                   primary_state->GetForms().end(),
                   std::back_inserter(aux_forms),
                   [](const int& a){ return a-1; });

    aux_state->SetForms(std::move(aux_forms));

    // Get the D operator
    auto d_op = my_state->GetOperator("D");

    // Not found. Instead, get the D operator from the DeRhamSequence
    if (!d_op)
    {
        Timer d_timer = TimeManager::AddTimer("Hiptmair: Build D");
        auto& sequence = state.GetDeRhamSequence();
        auto form = state.GetForms().front();
        PARELAG_ASSERT(form > 0);

        // Boundary conditions...
        auto ess_attr = state.GetBoundaryLabels(0);
        if (ess_attr.size() > 0)
        {
            mfem::Array<int> label_ess(ess_attr.data(),ess_attr.size());
            d_op = sequence.ComputeTrueD(form-1,label_ess);
        }
        else
            d_op = sequence.ComputeTrueD(form-1);
    }

    PARELAG_ASSERT(primary_state);
    PARELAG_ASSERT(aux_state);
    PARELAG_ASSERT(d_op);

    // Create the primary smoother from "op" and the primary state
    Timer pri_timer = TimeManager::AddTimer("Hiptmair: Build primary smoother");
    auto PrimarySmoo = std::shared_ptr<mfem::Solver>{
        PrimaryFact_->BuildSolver(op,*primary_state)};
    PrimarySmoo->iterative_mode = true;
    pri_timer.Stop();

    // Get/Create the Auxiliary operator
    Timer aux_op_timer = TimeManager::AddTimer("Hiptmair: Build auxiliary "
                                               "operator");
    auto aux_op = my_state->GetOperator("Auxiliary A");
    if (aux_op == nullptr)
        aux_op = _do_compute_aux_operator(*op,*d_op);
    aux_op_timer.Stop();

    // Make sure we have a valid pointer. We should, since
    // ComputeAuxOperator throws (std::runtime_error) if it cannot
    // construct a valid operator.
    PARELAG_ASSERT(aux_op);

    // Create the auxiliary smoother
    Timer aux_timer = TimeManager::AddTimer("Hiptmair: Build auxiliary "
                                            "smoother");
    auto AuxiliarySmoo = std::shared_ptr<mfem::Solver>{
        AuxiliaryFact_->BuildSolver(aux_op,*aux_state)};
    AuxiliarySmoo->iterative_mode = false;
    aux_timer.Stop();

    // Now I should be able to create the hybrid smoother
    return make_unique<HiptmairSmoother>(
        op, aux_op, d_op, PrimarySmoo, AuxiliarySmoo);
}


std::unique_ptr<mfem::Operator>
HiptmairSmootherFactory::_do_compute_aux_operator(
    mfem::Operator& A, mfem::Operator& D ) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        not (A.Width() == D.Height()),
        std::runtime_error,
        "A and D do not have compatible sizes to compute D^T*A*D!\n"
        "A = " << A.Height() << "x" << A.Width() << "\n"
        "D = " << D.Height() << "x" << D.Width() << "\n"
        "Perhaps you have passed the wrong form for your operator?\n");

    // This is *SO* ugly...
    auto A_hyp = dynamic_cast<mfem::HypreParMatrix *>(&A);
    auto D_hyp = dynamic_cast<mfem::HypreParMatrix *>(&D);
    if (A_hyp && D_hyp)
    {
        std::unique_ptr<mfem::HypreParMatrix> ret{
            mfem::RAP(A_hyp,D_hyp) };
        hypre_ParCSRMatrixFixZeroRows(*ret);

        return std::move(ret);
    }

    auto A_sparse = dynamic_cast<const mfem::SparseMatrix *>(&A);
    auto D_sparse = dynamic_cast<const mfem::SparseMatrix *>(&D);

    PARELAG_TEST_FOR_EXCEPTION(
        (not A_sparse) || (not D_sparse),
        std::runtime_error,
        "HiptmairSMootherFactory::_do_compute_aux_operator(): "
        "Either A and D do not have the same type or "
        "the type of A and D is not currently supported.");

    return std::unique_ptr<mfem::SparseMatrix>{
        mfem::RAP(*D_sparse,*A_sparse,*D_sparse) };
}

}// namespace parelag
