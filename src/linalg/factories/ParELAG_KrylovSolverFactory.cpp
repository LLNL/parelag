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


#include "linalg/factories/ParELAG_KrylovSolverFactory.hpp"

#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "linalg/solver_ops/ParELAG_KrylovSolver.hpp"

#include "utilities/ParELAG_TimeManager.hpp"

namespace parelag
{

std::unique_ptr<mfem::Solver> KrylovSolverFactory::_do_build_solver(
    const std::shared_ptr<mfem::Operator>& op, SolverState& state ) const
{
    auto my_state = dynamic_cast<NestedSolverState *>(&state);
    PARELAG_ASSERT(my_state);

    std::shared_ptr<mfem::Solver> prec;
    if (Prec_Factory_)
    {
        // Get the things I need from my state:
        auto prec_state = std::shared_ptr<SolverState>
        {
            Prec_Factory_->GetDefaultState()
        };

        if (my_state->IsSubState("Preconditioner"))
            prec_state->MergeState(*my_state->GetSubState("Preconditioner"));

        prec_state->MergeState(*my_state);

        if (DoPrecSetupTiming_)
        {

            //auto suffix = std::string(" (")
            //              .append(std::to_string(op->Height())).append(")");
            auto suffix = GetParameters().Get<std::string>("Timer name suffix");

            Timer setup = TimeManager::AddTimer(
                              std::string("KrylovSolver -- Setup Preconditioner")
                              .append(suffix));

            prec = Prec_Factory_->BuildSolver(op,*prec_state);
            prec->iterative_mode = false;

            mfem::Vector testx{op->Height()},testb{op->Height()};
            testx = 1.;
            testb = 2.;
            {
                Timer setup = TimeManager::AddTimer(
                                  std::string("KrylovSolver -- Setup Preconditioner")
                                  .append(suffix).append(" -- Iteration 1"));
                prec->Mult(testb,testx);
            }
            {
                Timer setup = TimeManager::AddTimer(
                                  std::string("KrylovSolver -- Setup Preconditioner")
                                  .append(suffix).append(" -- Iteration 2"));
                prec->Mult(testb,testx);
            }
        }
        else
        {
            prec = Prec_Factory_->BuildSolver(op,*prec_state);
        }
        prec->iterative_mode = false;
    }
    else
    {
        prec = nullptr;
    }

    return make_unique<KrylovSolver>(op,prec,GetParameters());
}


void KrylovSolverFactory::_do_set_default_parameters()
{
    auto& params = GetParameters();

    params.Get<int>("Print level",-1);
    params.Get<double>("Relative tolerance",0.0);
    params.Get<double>("Absolute tolerance",0.0);
    params.Get<int>("Maximum iterations",10);
    params.Get<int>("Restart size",50);
    params.Get<bool>("Print final paragraph",false);
    params.Get<bool>("Time preconditioner setup",false);
    params.Get<std::string>("Timer name suffix", "");
}


void KrylovSolverFactory::_do_initialize(const ParameterList&)
{
    // Nested preconditioner, so I assert that the Lib_ member is non-null
    PARELAG_ASSERT(HasValidSolverLibrary());

    // Build the primary smoother
    std::string prec_name =
        GetParameters().Get("Preconditioner", "None");

    if (prec_name != "None")
        Prec_Factory_ = GetSolverLibrary().GetSolverFactory(prec_name);

    DoPrecSetupTiming_ = GetParameters().Get("Time preconditioner setup",false);
}

}// namespace parelag
