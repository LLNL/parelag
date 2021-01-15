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


#include "linalg/factories/ParELAG_ADSSolverFactory.hpp"

#include "amge/DeRhamSequence.hpp"
#include "linalg/solver_ops/ParELAG_ADSSolverWrapper.hpp"

namespace parelag
{

std::unique_ptr<mfem::Solver> ADSSolverFactory::_do_build_solver(
    const std::shared_ptr<mfem::Operator>& op, SolverState& state ) const
{
    const auto& sequence = state.GetDeRhamSequence();

    return make_unique<ADSSolverWrapper>(
        op,sequence,const_cast<ParameterList&>(GetParameters()));
}

void ADSSolverFactory::_do_set_default_parameters()
{
    auto& params = GetParameters();

    params.Get<int>("Cycle type", 11);
    params.Get<double>("Tolerance", 0.0);
    params.Get<int>("Maximum iterations", 1);
    params.Get<int>("Print level", 1);
    params.Get<int>("Relaxation type", 2);
    params.Get<int>("Relaxation sweeps", 1);
    params.Get<double>("Relaxation weight", 1.0);
    params.Get<double>("Relaxation omega", 1.0);

    ParameterList& ams_params = params.Sublist("AMS Parameters");
    ams_params.Get<int>("Cycle type", 14);

    // NB: These do not seem to be used in this context (i.e., in ADSSolverWrapper).
    //     They would have been used in an actual AMS factory.
    ams_params.Get<int>("Relaxation type", 2);
    ams_params.Get<int>("Relaxation sweeps", 1);
    ams_params.Get<double>("Relaxation weight", 1.0);
    ams_params.Get<double>("Relaxation omega", 1.0);
    ams_params.Get<bool>("Beta is zero", false);

    ParameterList& ptap_mg_params = ams_params.Sublist("PtAP AMG Parameters");
    ptap_mg_params.Get<int>("Coarsening type", 10);
    ptap_mg_params.Get<int>("Aggressive coarsening levels", 1);
    ptap_mg_params.Get<int>("Relaxation type", 6);
    ptap_mg_params.Get<double>("Theta", 0.25);
    ptap_mg_params.Get<int>("Interpolation type", 6);
    ptap_mg_params.Get<int>("P max", 4);

    // NB: These do not seem to be used in this context (i.e., in ADSSolverWrapper).
    //     They would have been used in an actual BoomerAMG factory.
    ptap_mg_params.Get<int>("Print level", 0);
    ptap_mg_params.Get<int>("Number of functions", 1);
    ptap_mg_params.Get<int>("Maximum levels", 25);

    // NB: These do not seem to be used in this context (i.e., in ADSSolverWrapper).
    //     They would have been used in an actual AMS factory.
    ParameterList& gtag_mg_params = ams_params.Sublist("GtAG AMG Parameters");
    gtag_mg_params.Get<int>("Coarsening type", 10);
    gtag_mg_params.Get<int>("Aggressive coarsening levels", 1);
    gtag_mg_params.Get<int>("Relaxation type", 6);
    gtag_mg_params.Get<double>("Theta", 0.25);
    gtag_mg_params.Get<int>("Interpolation type", 6);
    gtag_mg_params.Get<int>("P max", 4);

    // NB: These would not have been used even in an actual AMS factory (more precisely, in AMSSolverWrapper),
    //     but would have been used in an actual BoomerAMG factory.
    gtag_mg_params.Get<int>("Print level", 0);
    gtag_mg_params.Get<int>("Number of functions", 1);
    gtag_mg_params.Get<int>("Maximum levels", 25);

    ParameterList& amg_params = params.Sublist("AMG Parameters");
    amg_params.Get<int>("Coarsening type", 10);
    amg_params.Get<int>("Aggressive coarsening levels", 1);
    amg_params.Get<int>("Relaxation type", 6);
    amg_params.Get<double>("Theta", 0.25);
    amg_params.Get<int>("Interpolation type", 6);
    amg_params.Get<int>("P max", 4);

    // NB: These do not seem to be used in this context (i.e., in ADSSolverWrapper).
    //     They would have been used in an actual BoomerAMG factory.
    amg_params.Get<int>("Print level", 0);
    amg_params.Get<int>("Number of functions", 1);
    amg_params.Get<int>("Maximum levels", 25);
}

void ADSSolverFactory::_do_initialize(ParameterList const&)
{
}
}// namespace parelag
