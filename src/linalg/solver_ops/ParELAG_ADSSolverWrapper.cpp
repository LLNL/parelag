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


#include "linalg/solver_ops/ParELAG_ADSSolverWrapper.hpp"

#include "amge/DeRhamSequence.hpp"

namespace parelag
{

ADSSolverWrapper::ADSSolverWrapper(
    const std::shared_ptr<mfem::Operator>& A,
    const DeRhamSequence& seq,
    ParameterList& Params)
    : HypreSolver{std::dynamic_pointer_cast<mfem::HypreParMatrix>(A).get()},
      A_{std::dynamic_pointer_cast<mfem::HypreParMatrix>(A)}
{
    PARELAG_ASSERT(A_)
    PARELAG_ASSERT(seq.GetNumForms() == 4);

    // Get the gradient and curl matrices
    G_ = seq.ComputeTrueD(0);
    C_ = seq.ComputeTrueD(1);

    // Get the cycle types
    const auto cycle_type = Params.Get<int>("Cycle type",11);

    ParameterList& ams_params = Params.Sublist("AMS Parameters");
    const auto ams_cycle_type = ams_params.Get<int>("Cycle type",13);

    // Create the projection operators for Nedelec space
    if (ams_cycle_type > 10)
        seq.ComputeTrueProjectorFromH1ConformingSpace(
            1, ND_Pix_, ND_Piy_, ND_Piz_);
    else
        ND_Pi_ = seq.ComputeTrueProjectorFromH1ConformingSpace(1);

    // Create the projection operators for RT space
    if (cycle_type > 10 )
        seq.ComputeTrueProjectorFromH1ConformingSpace(
            2, RT_Pix_, RT_Piy_, RT_Piz_);
    else
        RT_Pi_ = seq.ComputeTrueProjectorFromH1ConformingSpace(2);

    // Create the actual ADS object
    HYPRE_ADSCreate(&ads_);

    _do_set_parameters(Params);
}

void ADSSolverWrapper::_do_set_parameters(ParameterList& Params)
{

#if MFEM_HYPRE_VERSION <= 22200
    // Ummm why?
    PARELAG_ASSERT(
        hypre_ParCSRMatrixOwnsRowStarts(
            static_cast<hypre_ParCSRMatrix *>(*C_)) == 0 );
    PARELAG_ASSERT(
        hypre_ParCSRMatrixOwnsColStarts(
            static_cast<hypre_ParCSRMatrix *>(*C_)) == 0 );
#endif

    const auto cycle_type = Params.Get<int>("Cycle type",11);

    // Set the ADS parameters
    HYPRE_ADSSetCycleType(ads_, cycle_type);
    HYPRE_ADSSetTol(ads_, Params.Get<double>("Tolerance",0.0));
    HYPRE_ADSSetMaxIter(ads_, Params.Get<int>("Maximum iterations",1));
    HYPRE_ADSSetPrintLevel(ads_, Params.Get<int>("Print level",1));

    // Set the ADS operators
    HYPRE_ADSSetDiscreteCurl(ads_, *C_);
    HYPRE_ADSSetDiscreteGradient(ads_, *G_);

    ParameterList& amg_params = Params.Sublist("AMG Parameters");
    ParameterList& ams_params = Params.Sublist("AMS Parameters");
    ParameterList& ptap_mg_params = ams_params.Sublist("PtAP AMG Parameters");

    const auto ams_cycle_type = ams_params.Get<int>("Cycle type",13);

    // set additional ADS options
    HYPRE_ADSSetSmoothingOptions(
        ads_, Params.Get<int>("Relaxation type",2),
        Params.Get<int>("Relaxation sweeps",1),
        Params.Get<double>("Relaxation weight",1.0),
        Params.Get<double>("Relaxation omega",1.0));

    HYPRE_ADSSetAMGOptions(
        ads_, amg_params.Get<int>("Coarsening type",10),
        amg_params.Get<int>("Aggressive coarsening levels",1),
        amg_params.Get<int>("Relaxation type",6),
        amg_params.Get<double>("Theta",0.25),
        amg_params.Get<int>("Interpolation type",6),
        amg_params.Get<int>("P max",4));

    HYPRE_ADSSetAMSOptions(
        ads_,ams_cycle_type,
        ptap_mg_params.Get<int>("Coarsening type",10),
        ptap_mg_params.Get<int>("Aggressive coarsening levels",1),
        ptap_mg_params.Get<int>("Relaxation type",8),
        ptap_mg_params.Get<double>("Theta",0.25),
        ptap_mg_params.Get<int>("Interpolation type",6),
        ptap_mg_params.Get<int>("P max",4));

    // A bunch of sanity checking that can probably go away given the
    // only constructor guarantees that these conditions are true.
    PARELAG_TEST_FOR_EXCEPTION(
        cycle_type < 10 && not RT_Pi_,
        std::logic_error,
        "ADSSolverWrapper::_do_set_parameters(...):\n"
        "if ADS cycle type < 10 then RT_Pi should be provided.");

    PARELAG_TEST_FOR_EXCEPTION(
        cycle_type >= 10 && not RT_Pix_,
        std::logic_error,
        "ADSSolverWrapper::_do_set_parameters(...):\n"
        "if ADS cycle type >= 10 then RT_Pix, RT_Piy, RT_Piz"
        "should be provided.");

    PARELAG_TEST_FOR_EXCEPTION(
        cycle_type >= 10 && not RT_Piy_,
        std::logic_error,
        "ADSSolverWrapper::_do_set_parameters(...):\n"
        "if ADS cycle type >= 10 then RT_Pix, RT_Piy, RT_Piz"
        "should be provided.");

    PARELAG_TEST_FOR_EXCEPTION(
        cycle_type >= 10 && not RT_Piz_,
        std::logic_error,
        "ADSSolverWrapper::_do_set_parameters(...):\n"
        "if ADS cycle type >= 10 then RT_Pix, RT_Piy, RT_Piz:"
        "should be provided.");

    PARELAG_TEST_FOR_EXCEPTION(
        ams_cycle_type < 10 && not ND_Pi_,
        std::logic_error,
        "ADSSolverWrapper::_do_set_parameters(...):\n"
        "if AMS cycle type < 10 then ND_Pi should be provided.");

    PARELAG_TEST_FOR_EXCEPTION(
        ams_cycle_type >= 10 && not ND_Pix_,
        std::logic_error,
        "ADSSolverWrapper::_do_set_parameters(...):\n"
        "if AMS cycle type >= 10 then ND_Pi_x, ND_Pi_y, ND_Pi_z"
        "should be provided.");

    PARELAG_TEST_FOR_EXCEPTION(
        ams_cycle_type >= 10 && not ND_Piy_,
        std::logic_error,
        "ADSSolverWrapper::_do_set_parameters(...):\n"
        "if AMS cycle type >= 10 then ND_Pi_x, ND_Pi_y, ND_Pi_z"
        "should be provided.");

    PARELAG_TEST_FOR_EXCEPTION(
        ams_cycle_type >= 10 && not ND_Piz_,
        std::logic_error,
        "ADSSolverWrapper::_do_set_parameters(...):\n"
        "if AMS cycle type >= 10 then ND_Pi_x, ND_Pi_y, ND_Pi_z"
        "should be provided.");

    // Set the interpolations in the appropriate format
    hypre_ParCSRMatrix *RT_Pi_p, *RT_Pix_p, *RT_Piy_p, *RT_Piz_p,
        *ND_Pi_p, *ND_Pix_p, *ND_Piy_p, *ND_Piz_p;
    if (cycle_type < 10)
    {
        RT_Pi_p = *RT_Pi_;
        RT_Pix_p = nullptr;
        RT_Piy_p = nullptr;
        RT_Piz_p = nullptr;
    }
    else
    {
        RT_Pi_p = nullptr;
        RT_Pix_p = *RT_Pix_;
        RT_Piy_p = *RT_Piy_;
        RT_Piz_p = *RT_Piz_;
    }

    if (ams_cycle_type < 10)
    {
        ND_Pi_p = *ND_Pi_;
        ND_Pix_p = nullptr;
        ND_Piy_p = nullptr;
        ND_Piz_p = nullptr;
    }
    else
    {
        ND_Pi_p = nullptr;
        ND_Pix_p = *ND_Pix_;
        ND_Piy_p = *ND_Piy_;
        ND_Piz_p = *ND_Piz_;
    }
    HYPRE_ADSSetInterpolations(
        ads_, RT_Pi_p, RT_Pix_p, RT_Piy_p, RT_Piz_p,
        ND_Pi_p, ND_Pix_p, ND_Piy_p, ND_Piz_p);
}

ADSSolverWrapper::~ADSSolverWrapper()
{
    HYPRE_ADSDestroy(ads_);
}

}// namespace parelag
