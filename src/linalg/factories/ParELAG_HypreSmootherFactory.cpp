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


#include "linalg/factories/ParELAG_HypreSmootherFactory.hpp"
#include "linalg/solver_ops/ParELAG_HypreSmootherWrapper.hpp"

namespace parelag
{

std::unique_ptr<mfem::Solver>
HypreSmootherFactory::_do_build_solver(
    const std::shared_ptr<mfem::Operator>& op, SolverState&) const
{
    std::unique_ptr<mfem::Solver> ret;
    switch (Type_)
    {
    case 413:
        ret = make_unique<mfem::HypreDiagScale>(
            dynamic_cast<mfem::HypreParMatrix&>(*op));
        break;
    default:
        ret = make_unique<HypreSmootherWrapper>(
            op, Type_, const_cast<ParameterList&>(GetParameters()));
        break;
    }

    ret->iterative_mode = true;
    return ret;
    // TODO (trb 02/09/16): Add options support for
    // "SetFIRCoefficients", the "window" stuff, and
    // "SetTaubinOptions"
}


void HypreSmootherFactory::_do_set_default_parameters()
{
    auto& params = GetParameters();

    // Use the string for type
    params.Get("Type","L1 Gauss-Seidel");

    // General parameters
    params.Get<int>("Sweeps",1);
    params.Get<double>("Damping Factor",1.0);
    params.Get<double>("Omega",1.0);

    // Chebyshev parameters
    params.Get<int>("Cheby Poly Order",2);
    params.Get<double>("Cheby Poly Fraction",0.3);

    // TODO (trb 02/09/16): Add options for FIR and Taubin smoothers.
    // TODO (trb 02/09/16): What are "FIR" and "Taubin"...?
}


void HypreSmootherFactory::_do_initialize(const ParameterList&)
{
    Type_ = StringTypeToInt(GetParameters().Get<std::string >("Type"));
}


std::string HypreSmootherFactory::IntTypeToString(int type) const noexcept
{
    switch (type)
    {
    case 0: return "Jacobi";
    case 1: return "L1 Jacobi";
    case 2: return "L1 Gauss-Seidel";
    case 3: return "Kaczmarz";
    case 4: return "L1 Gauss-Seidel Truncated";
    case 5: return "Lumped Jacobi";
    case 6: return "Gauss-Seidel";
    case 16: return "Chebyshev";
    case 1001: return "Taubin";
    case 1002: return "FIR";
    case 413: return "Hypre Jacobi";
    default: return "";
    }
}

int HypreSmootherFactory::StringTypeToInt(const std::string& type) const noexcept
{
    // These are the types that MFEM cares to enumerate, plus
    // Kaczmarz, which is documented in they hypre .c file...
    if (type == "Jacobi")                    return 0;
    if (type == "L1 Jacobi")                 return 1;
    if (type == "L1 Gauss-Seidel")           return 2;
    if (type == "Kaczmarz")                  return 3;
    if (type == "L1 Gauss-Seidel Truncated") return 4;
    if (type == "Lumped Jacobi")             return 5;
    if (type == "Gauss-Seidel")              return 6;
    if (type == "Chebyshev")                 return 16;
    if (type == "Taubin")                    return 1001;
    if (type == "FIR")                       return 1002;
    if (type == "Hypre Jacobi")              return 413;

    return -1;
}

}// namespace parelag
