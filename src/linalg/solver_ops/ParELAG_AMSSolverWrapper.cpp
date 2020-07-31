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


#include "linalg/solver_ops/ParELAG_AMSSolverWrapper.hpp"

#include "amge/DeRhamSequence.hpp"

namespace parelag
{

AMSSolverWrapper::AMSSolverWrapper(
    const std::shared_ptr<mfem::Operator>& A,
    const DeRhamSequence& seq,
    ParameterList& Params)
    : HypreSolver{std::dynamic_pointer_cast<mfem::HypreParMatrix>(A).get()},
      A_{std::dynamic_pointer_cast<mfem::HypreParMatrix>(A)}
{
    PARELAG_ASSERT(A_)
    PARELAG_ASSERT(seq.GetNumForms() == 4);

    G_ = seq.ComputeTrueD(0);

    // Get the cycle types
    const auto cycle_type = Params.Get<int>("Cycle type",13);

    // Create the projection operators for Nedelec space
    if (cycle_type > 10)
        seq.ComputeTrueProjectorFromH1ConformingSpace(
            1, Pix_, Piy_, Piz_);
    else
        Pi_ = seq.ComputeTrueProjectorFromH1ConformingSpace(1);

    const int dim = seq.GetNumberOfForms() - 1;
    Params.Set("Number of dimensions",dim);

    // Create the actual AMS object
    HYPRE_AMSCreate(&ams_);

    _do_set_parameters(Params);
}

void AMSSolverWrapper::_do_set_parameters(ParameterList& Params)
{
    const auto cycle_type = Params.Get<int>("Cycle type",11);

    // Set the AMS parameters
    HYPRE_AMSSetCycleType(ams_, cycle_type);
    HYPRE_AMSSetDimension(ams_, Params.Get<int>("Number of dimensions"));
    HYPRE_AMSSetTol(ams_, Params.Get<double>("Tolerance",0.0));
    HYPRE_AMSSetMaxIter(ams_, Params.Get<int>("Maximum iterations",1));
    HYPRE_AMSSetPrintLevel(ams_, Params.Get<int>("Print level",1));

    // Set the AMS operators
    HYPRE_AMSSetDiscreteGradient(ams_, *G_);

    ParameterList& ptap_mg_params = Params.Sublist("PtAP AMG Parameters");
    ParameterList& gtag_mg_params = Params.Sublist("GtAG AMG Parameters");


    // set additional AMS options
    HYPRE_AMSSetSmoothingOptions(
        ams_, Params.Get<int>("Relaxation type",2),
        Params.Get<int>("Relaxation sweeps",1),
        Params.Get<double>("Relaxation weight",1.0),
        Params.Get<double>("Relaxation omega",1.0));

    HYPRE_AMSSetAlphaAMGOptions(
        ams_, ptap_mg_params.Get<int>("Coarsening type",10),
        ptap_mg_params.Get<int>("Aggressive coarsening levels",1),
        ptap_mg_params.Get<int>("Relaxation type",6),
        ptap_mg_params.Get<double>("Theta",0.25),
        ptap_mg_params.Get<int>("Interpolation type",6),
        ptap_mg_params.Get<int>("P max",4));

    if (not Params.Get<bool>("Beta is zero", false))
    {
        HYPRE_AMSSetBetaAMGOptions(
            ams_, gtag_mg_params.Get<int>("Coarsening type",10),
            gtag_mg_params.Get<int>("Aggressive coarsening levels",1),
            gtag_mg_params.Get<int>("Relaxation type",6),
            gtag_mg_params.Get<double>("Theta",0.25),
            gtag_mg_params.Get<int>("Interpolation type",6),
            gtag_mg_params.Get<int>("P max",4));
    }
    else
    {
        HYPRE_AMSSetBetaPoissonMatrix(ams_, nullptr);
    }

    // A bunch of sanity checking that can probably go away given the
    // only constructor guarantees that these conditions are true.
    PARELAG_TEST_FOR_EXCEPTION(
        cycle_type < 10 && not Pi_,
        std::logic_error,
        "AMSSolverWrapper::_do_set_parameters(...):\n"
        "if AMS cycle type < 10 then Pi_ should be provided.");

    PARELAG_TEST_FOR_EXCEPTION(
        cycle_type >= 10 && not Pix_,
        std::logic_error,
        "AMSSolverWrapper::_do_set_parameters(...):\n"
        "if AMS cycle type >= 10 then Pix, Piy, Piz"
        "should be provided.");

    PARELAG_TEST_FOR_EXCEPTION(
        cycle_type >= 10 && not Piy_,
        std::logic_error,
        "AMSSolverWrapper::_do_set_parameters(...):\n"
        "if AMS cycle type >= 10 then Pix, Piy, Piz"
        "should be provided.");

    PARELAG_TEST_FOR_EXCEPTION(
        cycle_type >= 10 && not Piz_,
        std::logic_error,
        "AMSSolverWrapper::_do_set_parameters(...):\n"
        "if AMS cycle type >= 10 then Pix, Piy, Piz:"
        "should be provided.");

    if (cycle_type > 10)
        HYPRE_AMSSetInterpolations(ams_, nullptr, *Pix_, *Piy_, *Piz_);
    else
        HYPRE_AMSSetInterpolations(ams_, *Pi_, nullptr, nullptr, nullptr);

}

AMSSolverWrapper::~AMSSolverWrapper()
{
    HYPRE_AMSDestroy(ams_);
}

}// namespace parelag
