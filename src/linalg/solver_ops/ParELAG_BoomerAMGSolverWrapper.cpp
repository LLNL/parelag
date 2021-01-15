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


#include "linalg/solver_ops/ParELAG_BoomerAMGSolverWrapper.hpp"

namespace parelag
{

BoomerAMGSolverWrapper::BoomerAMGSolverWrapper(
    const std::shared_ptr<Operator>& A,
    ParameterList& Params)
    : Solver{A->Width(),A->Height(),false},
      A_{std::dynamic_pointer_cast<mfem::HypreParMatrix>(A)}
{
    PARELAG_ASSERT(A_);

    amg_.SetOperator(*A_);
    amg_.iterative_mode = true;

    _do_set_parameters(Params);
}


void BoomerAMGSolverWrapper::_do_set_operator(
    const std::shared_ptr<mfem::Operator>& A)
{
    auto tmp = std::dynamic_pointer_cast<mfem::HypreParMatrix>(A);
    PARELAG_ASSERT(tmp);

    A_ = std::move(tmp);
    amg_.SetOperator(*A_);
    amg_.iterative_mode = true;
}


void BoomerAMGSolverWrapper::_do_set_parameters(ParameterList& Params)
{
    auto amg_h = (HYPRE_Solver)amg_;

    HYPRE_BoomerAMGSetCoarsenType(
        amg_h, Params.Get<int>("Coarsening type",10));
    HYPRE_BoomerAMGSetAggNumLevels(
        amg_h, Params.Get<int>("Aggressive coarsening levels",1));
    HYPRE_BoomerAMGSetRelaxType(
        amg_h, Params.Get<int>("Relaxation type",8));
    HYPRE_BoomerAMGSetNumSweeps(
        amg_h, Params.Get<int>("Relaxation sweeps",1));
    HYPRE_BoomerAMGSetMaxLevels(
        amg_h, Params.Get<int>("Maximum levels",25));
    HYPRE_BoomerAMGSetTol(
        amg_h, Params.Get<double>("Tolerance",0.0));
    HYPRE_BoomerAMGSetMaxIter(
        amg_h, Params.Get<int>("Maximum iterations",1));
    HYPRE_BoomerAMGSetStrongThreshold(
        amg_h, Params.Get<double>("Theta",0.25));
    HYPRE_BoomerAMGSetInterpType(
        amg_h, Params.Get<int>("Interpolation type",6));
    HYPRE_BoomerAMGSetPMaxElmts(
        amg_h, Params.Get<int>("P max",4));
    HYPRE_BoomerAMGSetPrintLevel(
        amg_h, Params.Get<int>("Print level",0));

    HYPRE_BoomerAMGSetNumFunctions(
        amg_h, Params.Get<int>("Number of functions",1));

    if (Params.Get<int>("Number of functions",1) > 1)
    {
        HYPRE_BoomerAMGSetStrongThreshold(
            amg_h, Params.Get<double>("Strong Threshold",0.5));

        auto& DofToFunction_vec = Params.Get<std::vector<int>>(
            "DOF to function map",std::vector<int>());
        HYPRE_Int* dof_to_function =
            (HYPRE_Int*) malloc(sizeof(HYPRE_Int)*DofToFunction_vec.size());
        std::copy(DofToFunction_vec.begin(),DofToFunction_vec.end(),
                  dof_to_function);

        HYPRE_BoomerAMGSetDofFunc(amg_h,dof_to_function);
    }
}

}// namespace parelag
