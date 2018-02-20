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


#include "linalg/factories/ParELAG_BoomerAMGSolverFactory.hpp"

#include "linalg/solver_ops/ParELAG_BoomerAMGSolverWrapper.hpp"
#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"


namespace parelag
{

std::unique_ptr<mfem::Solver> BoomerAMGSolverFactory::_do_build_solver(
        const std::shared_ptr<mfem::Operator>& op, SolverState& state ) const
{
    std::vector<int> dof_to_function =
        state.GetExtraParameter<std::vector<int>>(
            "DOF to function map",std::vector<int>());
    const_cast<ParameterList&>(GetParameters()).Set<std::vector<int>>(
        "DOF to function map", std::move(dof_to_function));

    return make_unique<BoomerAMGSolverWrapper>(
        op,const_cast<ParameterList&>(GetParameters()));
}


void BoomerAMGSolverFactory::_do_set_default_parameters()
{
    auto& params = GetParameters();
    params.Get<double>("Tolerance", 0.0);
    params.Get<int>("Maximum iterations", 1);
    params.Get<int>("Coarsening type", 10);
    params.Get<int>("Aggressive coarsening levels", 1);
    params.Get<int>("Relaxation type", 6);
    params.Get<double>("Theta", 0.25);
    params.Get<int>("Interpolation type", 6);
    params.Get<int>("P max", 4);
    params.Get<int>("Print level", 0);
    params.Get<int>("Maximum levels", 25);
    params.Get<int>("Number of functions", 1);
    params.Get<double>("Strong Threshold", 0.5);
}


}// namespace parelag
