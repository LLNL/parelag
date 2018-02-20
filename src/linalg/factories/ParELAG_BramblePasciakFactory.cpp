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


#include "linalg/factories/ParELAG_BramblePasciakFactory.hpp"

#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "linalg/solver_ops/ParELAG_BramblePasciakSolver.hpp"
#include "linalg/solver_ops/ParELAG_BramblePasciakTransformation.hpp"
#include "linalg/utilities/ParELAG_MG_Utils.hpp"
#include "linalg/utilities/ParELAG_MonolithicBlockedOperatorFactory.hpp"

namespace parelag
{

std::unique_ptr<mfem::Solver> BramblePasciakFactory::_do_build_block_solver(
    const std::shared_ptr<MfemBlockOperator>& op, SolverState& state) const
{
    PARELAG_ASSERT(op);

    auto& params = GetParameters();

    // Build transformation
    const auto trans_strategy =
        params.Get<std::string>("Transformation strategy");

    const auto solver_is_monolithic =
        params.Get<bool>("Solver is monolithic");

    std::unique_ptr<MfemBlockOperator> transformation_op;
    std::shared_ptr<mfem::Operator> transformed_op;
    if (trans_strategy == "Assemble then transform")
    {
        BramblePasciakTransformation bp_trans;

        // Currently this only returns a parallel operator, so
        // "transform then assemble" cannot be supported. Yet.
        transformation_op = bp_trans.BuildOperator(*op,state);

        // Need to compute
        transformed_op = mg_utils::BlockedMatrixMult<mfem::HypreParMatrix>(
            *transformation_op,*op);
    }
    else if (trans_strategy == "Transform then assemble")
    {
        PARELAG_NOT_IMPLEMENTED();
    }
    else
    {
        const bool invalid_transformation_strategy = true;

        PARELAG_TEST_FOR_EXCEPTION(
            invalid_transformation_strategy,
            std::runtime_error,
            "BramblePasciakFactory::BuildBlockSolver(...):\n"
            "Transformation strategy \"" << trans_strategy <<
            "\" is invalid. Options are \"Assemble then transform\" or "
            "\"Transform then assemble\".");
    }

    if (solver_is_monolithic)
    {
        MonolithicBlockedOperatorFactory mono_fact;
        transformed_op = mono_fact.BuildOperator(
            *std::dynamic_pointer_cast<MfemBlockOperator>(transformed_op));

        // FIXME: HOW TO PROPAGATE DOF TO FUNCTION FORWARD
        std::vector<int> dof_map = mono_fact.GetDofToFunctionMap();
        const_cast<ParameterList&>(params).Set<std::vector<int>>(
            "DOF to function map", std::move(dof_map));

        state.SetExtraParameter<std::vector<int>>(
            "DOF to function map", std::move(dof_map));
    }

    // Build solver based on transformation
    auto solver = SolverFact_->BuildSolver(transformed_op,state);
    solver->iterative_mode = false;

    return make_unique<BramblePasciakSolver>(
        std::move(transformation_op),std::move(solver));;
}


void BramblePasciakFactory::_do_set_default_parameters()
{
    // Options: "Assemble then transform" or "transform then assemble"
    GetParameters().Get<std::string>("Transformation strategy",
                                     "Assemble then transform");

    // May be any solver known to the library
    GetParameters().Get<std::string>("Solver","Invalid");
}


void BramblePasciakFactory::_do_initialize(const ParameterList&)
{
    // Depends on a library being available
    PARELAG_ASSERT(HasValidSolverLibrary());

    const std::string solver_name =
        GetParameters().Get<std::string>("Solver");

    SolverFact_ = GetSolverLibrary().GetSolverFactory(solver_name);
}

}// namespace parelag
