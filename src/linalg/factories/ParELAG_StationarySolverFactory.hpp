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


#ifndef PARELAG_STATIONARYSOLVERFACTORY_HPP_
#define PARELAG_STATIONARYSOLVERFACTORY_HPP_

#include "linalg/solver_core/ParELAG_SolverFactory.hpp"
#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "linalg/solver_ops/ParELAG_StationarySolver.hpp"

namespace parelag
{

/** \class StationarySolverFactory
 *  \brief Factory for building a StationarySolver object.
 */
class StationarySolverFactory : public SolverFactory
{
public:

    /** \brief Construct from a list of parameters. */
    StationarySolverFactory(const ParameterList& params = ParameterList())
    { SetParameters(params); }

    /** \brief Construct from a solver factory and some additional
     *         parameters.
     */
    StationarySolverFactory(
        std::shared_ptr<SolverFactory> SolverFact,
        const ParameterList& params = ParameterList())
        : Fact_{std::move(SolverFact)}
    { SetParameters(params); }

private:

    std::unique_ptr<mfem::Solver>
    _do_build_solver(const std::shared_ptr<mfem::Operator>& op,
                SolverState& state) const override
    {
        PARELAG_ASSERT(Fact_);

        auto my_state = dynamic_cast<NestedSolverState *>(&state);
        PARELAG_ASSERT(my_state);

        auto solver_state = std::shared_ptr<SolverState>{Fact_->GetDefaultState()};
        if (my_state->IsSubState("Solver"))
            solver_state->MergeState(*my_state->GetSubState("Solver"));

        solver_state->MergeState(*my_state);

        auto prec = std::shared_ptr<mfem::Solver>{
            Fact_->BuildSolver(op,*solver_state)};
        prec->iterative_mode = false;

        return make_unique<StationarySolver>(
            op,prec,
            GetParameters().Get<double>("Relative Tolerance"),
            GetParameters().Get<double>("Absolute Tolerance"),
            GetParameters().Get<size_t>("Maximum Iterations"),
            GetParameters().Get<bool>("Print Iterations"));
    }

    void _do_set_default_parameters() override
    {
        GetParameters().Get("Maximum Iterations",(size_t) 1);
        GetParameters().Get("Absolute Tolerance",0.0);
        GetParameters().Get("Relative Tolerance",0.0);
        GetParameters().Get("Print Iterations",false);
    }

    void _do_initialize(const ParameterList&) override
    {
        // Nested preconditioner, so I assert that the Lib_ member is non-null
        PARELAG_ASSERT(HasValidSolverLibrary());

        // Build the primary smoother
        std::string solver_name =
            GetParameters().Get("Solver", "INVALID");

        Fact_ = GetSolverLibrary().GetSolverFactory(solver_name);
    }

    std::unique_ptr<SolverState> _do_get_default_state() const override
    { return make_unique<NestedSolverState>(); }

private:

    /** \brief The factory to build the "real" solver. */
    std::shared_ptr<SolverFactory> Fact_;

};// class StationarySolverFactory
}// namespace parelag
#endif /* PARELAG_STATIONARYSOLVERFACTORY_HPP_ */
