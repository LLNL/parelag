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


#include "linalg/factories/ParELAG_Block2x2LDUSolverFactory.hpp"

#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "linalg/solver_ops/ParELAG_Block2x2LDUInverseOperator.hpp"

namespace parelag
{

Block2x2LDUSolverFactory::Block2x2LDUSolverFactory(
    const ParameterList& params)
{ SetParameters(params); }


Block2x2LDUSolverFactory::Block2x2LDUSolverFactory(
    std::shared_ptr<SolverFactory> a00_1_inverse_factory,
    std::shared_ptr<SolverFactory> a00_2_inverse_factory,
    std::shared_ptr<SolverFactory> a00_3_inverse_factory,
    std::shared_ptr<SolverFactory> s_inverse_factory,
    std::shared_ptr<SchurComplementFactory> s_factory,
    const ParameterList& params)
    : InvA00_1_Fact_{std::move(a00_1_inverse_factory)},
      InvA00_2_Fact_{std::move(a00_2_inverse_factory)},
      InvA00_3_Fact_{std::move(a00_3_inverse_factory)},
      InvS_Fact_{std::move(s_inverse_factory)},
      S_Fact_{std::move(s_factory)}
{ SetParameters(params); }


std::unique_ptr<mfem::Solver>
Block2x2LDUSolverFactory::_do_build_block_solver(
    const std::shared_ptr<MfemBlockOperator>& blop, SolverState& state) const
{
    PARELAG_ASSERT(blop);
    PARELAG_ASSERT(blop->GetNumBlockRows() == 2);
    PARELAG_ASSERT(blop->GetNumBlockCols() == 2);

    auto my_state = dynamic_cast<NestedSolverState *>(&state);
    PARELAG_ASSERT(my_state);

    // Get the things I need from my state:
    auto A00_1_state = std::shared_ptr<SolverState>{
        InvA00_1_Fact_->GetDefaultState()};
    auto A00_2_state = std::shared_ptr<SolverState>{
        InvA00_2_Fact_->GetDefaultState()};
    auto A00_3_state = std::shared_ptr<SolverState>{
        InvA00_3_Fact_->GetDefaultState()};
    auto InvS_state = std::shared_ptr<SolverState>{
        InvS_Fact_->GetDefaultState()};

    if (my_state->IsSubState("A00_1"))
        A00_1_state->MergeState(*my_state->GetSubState("A00_1"));
    if (my_state->IsSubState("A00_2"))
        A00_2_state->MergeState(*my_state->GetSubState("A00_2"));
    if (my_state->IsSubState("A00_3"))
        A00_3_state->MergeState(*my_state->GetSubState("A00_3"));
    if (my_state->IsSubState("InvS"))
        InvS_state->MergeState(*my_state->GetSubState("InvS"));

    A00_1_state->MergeState(*my_state);
    A00_2_state->MergeState(*my_state);
    A00_3_state->MergeState(*my_state);

    InvS_state->MergeState(*my_state);
    InvS_state->SetForms({A00_1_state->GetForms()[1]});

    A00_3_state->SetForms({A00_1_state->GetForms()[0]});
    A00_2_state->SetForms({A00_1_state->GetForms()[0]});
    A00_1_state->SetForms({A00_1_state->GetForms()[0]});

    // Get A00
    auto A00 = blop->GetBlockPtr(0,0);

    auto invA00_1 = std::shared_ptr<mfem::Solver>{
        InvA00_1_Fact_->BuildSolver(A00,*A00_1_state)};
    auto invA00_2 = std::shared_ptr<mfem::Solver>{
        InvA00_2_Fact_->BuildSolver(A00,*A00_2_state)};
    auto invA00_3 = std::shared_ptr<mfem::Solver>{
        InvA00_3_Fact_->BuildSolver(A00,*A00_3_state)};

    std::shared_ptr<mfem::Operator> A11;
    if (not S_Fact_)
        A11 = blop->GetBlockPtr(1,1);
    else
    {
        A11 = S_Fact_->BuildOperator(*blop,*InvS_state);
        if (UseNegativeS_)
        {
            auto* s_mat = dynamic_cast<mfem::HypreParMatrix*>(A11.get());
            if (s_mat)
                *s_mat *= -1.0;
        }
    }

    auto invS = std::shared_ptr<mfem::Solver>{
        InvS_Fact_->BuildSolver(A11,*InvS_state)};

    return make_unique<Block2x2LDUInverseOperator>(
        blop, invA00_1, invA00_2, invA00_3, invS, A11, Damping_Factor_);
}


void Block2x2LDUSolverFactory::_do_set_default_parameters()
{
    auto& params = GetParameters();
    params.Get<bool>("Use Negative S",true);
    params.Get<double>("Damping Factor",1.0);
    params.Get<double>("Alpha",1.0);
    params.Get<std::string>("S Type","NONE");
}


void Block2x2LDUSolverFactory::_do_initialize(const ParameterList&)
{
    auto& params = GetParameters();

    Damping_Factor_ = params.Get<double>("Damping Factor",1.0);
    Alpha_ = params.Get<double>("Alpha",1.0);
    UseNegativeS_ = params.Get<bool>("Use Negative S",true);

    // Nested preconditioner, so I assert good library object
    PARELAG_ASSERT(HasValidSolverLibrary());

    // Build the A00_1 solver factory
    std::string invA00_1_name =
        params.Get("A00_1 Inverse", "Default Hypre");

    InvA00_1_Fact_ = GetSolverLibrary().GetSolverFactory(invA00_1_name);

    // Build the A00_2 solver factory
    std::string invA00_2_name =
        params.Get("A00_2 Inverse", "Default Hypre");

    InvA00_2_Fact_ = GetSolverLibrary().GetSolverFactory(invA00_2_name);

    // Build the A00_3 solver factory
    std::string invA00_3_name =
        params.Get("A00_3 Inverse", "Default Hypre");

    InvA00_3_Fact_ = GetSolverLibrary().GetSolverFactory(invA00_3_name);

    // Build the S factory
    std::string S_name = params.Get("S Type", "NONE");

    if (S_name != "NONE")
        S_Fact_ = std::make_shared<SchurComplementFactory>(S_name,Alpha_);
    else
        S_Fact_ = nullptr;

    // Build the S solver factory
    std::string invS_name =
        params.Get("S Inverse", "Default Hypre");

    InvS_Fact_ = GetSolverLibrary().GetSolverFactory(invS_name);
}

}// namespace parelag
