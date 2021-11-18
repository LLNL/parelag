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


#include "linalg/factories/ParELAG_AMGeSolverFactory.hpp"

#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "linalg/solver_ops/ParELAG_Hierarchy.hpp"

#include "amge/DeRhamSequence.hpp"
#include "utilities/ParELAG_TimeManager.hpp"

namespace parelag
{

std::unique_ptr<mfem::Solver>
AMGeSolverFactory::_do_build_solver(
    const std::shared_ptr<mfem::Operator>& op,
    SolverState& state ) const
{
    using Op_Ptr = std::shared_ptr<mfem::Operator>;

    auto my_state = dynamic_cast<NestedSolverState *>(&state);
    PARELAG_ASSERT(my_state);

    auto sequence = state.GetDeRhamSequencePtr();

    // Create the hierarchy - this does not set smoothers or
    // coarse-grid solver
    if (Forms_.size() == 0)
        Forms_ = state.GetForms();

    PARELAG_ASSERT(Forms_.size() > 0);

    std::unique_ptr<mfem::Solver> out;
    if (Forms_.size() == 1)
    {
        Timer build_from_derham_timer = TimeManager::AddTimer(
            "Build Hierarchy: build from deRham Sequence");

        // Get the essential boundary labels
        auto& ess_attr = state.GetBoundaryLabels(0);

        out = buildHierarchyFromDeRhamSequence(
            op,*sequence,ess_attr,Forms_.front(),MaxLevels_);
    }
    else
    {
        Timer build_from_derham_timer = TimeManager::AddTimer(
            "Build Hierarchy: build from deRham Sequence");

        // FIXME --- needs to be more general to support blocks
        auto& ess_attr = state.GetBoundaryLabels();

        out = buildBlockedHierarchyFromDeRhamSequence(
            op,*sequence,ess_attr,Forms_,MaxLevels_);
    }

    Hierarchy* H = dynamic_cast<Hierarchy*>(out.get());
    PARELAG_ASSERT(H);

    // Fill the hierarchy
    const auto CoarsestLevelID = H->GetNumLevels() - 1;
    for (const auto& level : *H)
    {
        Level& lev = *level;
        auto LevelID = lev.GetLevelID();

        // Create level timer
        std::ostringstream timer_name;
        timer_name << "Build Hierarchy: fill level " << LevelID;
        Timer fill_level_timer = TimeManager::AddTimer(timer_name.str());

        // Get the operator for this level
        auto A_ptr = lev.Get<Op_Ptr>("A");

        // Print out some operator information
        auto A_hyp = std::dynamic_pointer_cast<mfem::HypreParMatrix>(A_ptr);
        if (A_hyp)
        {
            int myid;
            MPI_Comm_rank(A_hyp->GetComm(),&myid);

            if (!myid)
                std::cout << "Level " << LevelID << ": A = "
                          << A_hyp->M() << "x" << A_hyp->N()
                          << ", nnz = " << A_hyp->NNZ() << "\n";
        }

#if 0
        // FIXME: Add support for pre == post
        // Decide if pre- and post-smoother are the same.
        const bool same_pre_and_post_smoother =
            (PreSmootherFact_ == PostSmootherFact_);
#endif

        // Set up the Smoothers
        if (LevelID < CoarsestLevelID)
        {
            // Handle the states
            auto pre_state = std::shared_ptr<SolverState>{
                PreSmootherFact_->GetDefaultState()};
            auto post_state = std::shared_ptr<SolverState>{
                PostSmootherFact_->GetDefaultState()};

            // Deal with the state
            if (my_state->IsSubState("PreSmoother"))
                pre_state->MergeState(*my_state->GetSubState("PreSmoother"));

            if (my_state->IsSubState("PostSmoother"))
                post_state->MergeState(*my_state->GetSubState("PostSmoother"));

            pre_state->MergeState(state);
            post_state->MergeState(state);

            pre_state->SetDeRhamSequence(sequence);
            post_state->SetDeRhamSequence(sequence);

            // Set the smoothing operators
            Timer smoother_timer =
                TimeManager::AddTimer(std::string("Build smoother: level ").
                                      append(std::to_string(LevelID)));
            auto pre = PreSmootherFact_->BuildSolver(A_ptr,*pre_state);
            auto post = PostSmootherFact_->BuildSolver(A_ptr,*post_state);
            smoother_timer.Stop();
            pre->iterative_mode = true;
            post->iterative_mode = true;

            lev.Set<Op_Ptr>("PreSmoother",std::move(pre));

            lev.Set<Op_Ptr>("PostSmoother",std::move(post));

        }
        else
        {
            auto coarse_state = std::shared_ptr<SolverState>{
                CoarseSolverFact_->GetDefaultState()};

            // Deal with state
            if (my_state->IsSubState("Coarse solver"))
                coarse_state->MergeState(
                    *my_state->GetSubState("Coarse solver"));

            coarse_state->MergeState(state);
            coarse_state->SetDeRhamSequence(sequence);

            Timer solver_timer =
                TimeManager::AddTimer(std::string("Build coarse solver: "
                                      "level ").append(std::to_string(LevelID)));
            lev.Set<Op_Ptr>("CoarseSolver",
                            CoarseSolverFact_->BuildSolver(
                                A_ptr,*coarse_state));
        }

        sequence = sequence->CoarserSequence();
    }

    return out;
}

void AMGeSolverFactory::_do_initialize(ParameterList const&)
{
    auto& params = GetParameters();

    // I have to be a little more careful here. The desired behavior is:
    //
    // 1. If "Smoother" defined, use for both pre- and post-smoothing
    // 2. Otherwise, use "PreSmoother" for pre-smoothing and
    //    "PostSmoother" for post-smoothing.
    // 3. If "PreSmoother" or "PostSmoother" is not defined, do nothing.
    if (params.IsParameter("Smoother"))
    {
        auto smoother_name = params.Get<std::string>("Smoother");
        PreSmootherFact_ = GetSolverLibrary().GetSolverFactory(smoother_name);
        PostSmootherFact_ = PreSmootherFact_;
    }
    else
    {
        if (params.IsParameter("PreSmoother"))
        {
            auto pre_smoother_name = params.Get<std::string>("PreSmoother");
            PreSmootherFact_ = GetSolverLibrary().GetSolverFactory(pre_smoother_name);
        }
        if (params.IsParameter("PostSmoother"))
        {
            auto post_smoother_name = params.Get<std::string>("PostSmoother");
            PostSmootherFact_ = GetSolverLibrary().GetSolverFactory(post_smoother_name);
        }
    }

    if (params.IsParameter("Coarse solver"))
    {
        auto solver_name = params.Get<std::string>("Coarse solver");
        CoarseSolverFact_ = GetSolverLibrary().GetSolverFactory(solver_name);
    }

    MaxLevels_ = params.Get("Maximum levels",-1);
    Forms_ = params.Get("Forms",std::vector<int>());
}

}// namespace parelag
