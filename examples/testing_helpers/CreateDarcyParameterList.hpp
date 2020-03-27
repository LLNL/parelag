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

#ifndef _2FORM_HELPERS_HPP__
#define _2FORM_HELPERS_HPP__

#include "utilities/ParELAG_ParameterList.hpp"

namespace parelag
{
namespace testhelpers
{

std::unique_ptr<ParameterList> CreateDarcyTestParameters()
{
    auto ret = make_unique<ParameterList>("Default");

    // Problem parameters
    {
        auto& prob_params = ret->Sublist("Problem parameters");

        prob_params.Set("Mesh file","TestingMesh");
        prob_params.Set("Serial refinement levels",-1);
        prob_params.Set("Parallel refinement levels",2);
        prob_params.Set("Finite element order",0);
        prob_params.Set("Upscaling order",0);
        prob_params.Set("Solve level",0);
        prob_params.Set("Deformation",false);
        prob_params.Set("List of linear solvers",
                        std::list<std::string>{
                                "PCG with Auxiliary Space Preconditioner"});

    }// Problem parameters

    // Output control
    {
        auto& output_params = ret->Sublist("Output control");

        output_params.Set("Visualize solution",false);
        output_params.Set("Print timings",false);
        output_params.Set("Show progress",false);
    }// Output control

    // Preconditioner Library
    {
        auto& prec_lib = ret->Sublist("Preconditioner Library");

        // Hybridization-Darcy
        {
            auto& list = prec_lib.Sublist("Hybridization-Darcy");
            list.Set("Type", "Hybridization");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("Forms", std::vector<int>{2, 3});
                solver_list.Set("Solver", "CG_PCG-AMG");
                solver_list.Set("RescaleIteration", 1);
            }
        }// Hybridization-Darcy

        // CG_PCG-AMG
        {
            auto& list = prec_lib.Sublist("CG_PCG-AMG");
            list.Set("Type","Krylov");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("Solver name","PCG");
                solver_list.Set("Preconditioner","BoomerAMG Solver");
                solver_list.Set("Print level",1);
                solver_list.Set("Maximum iterations",300);
                solver_list.Set("Relative tolerance",1e-6);
                solver_list.Set("Absolute tolerance",1e-6);
            }
        }// CG_PCG-AMG

        // Gauss-Seidel
        {
            auto& list = prec_lib.Sublist("Gauss-Seidel");
            list.Set("Type","Hypre");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("Type","L1 Gauss-Seidel");
                solver_list.Set("Sweeps",1);
                solver_list.Set("Damping Factor",1.0);
                solver_list.Set("Omega",1.0);
                solver_list.Set("Cheby Poly Order",2);
                solver_list.Set("Cheby Poly Fraction",0.3);
            }
        }// Gauss-Seidel

        // BoomerAMG Solver
        {
            auto& list = prec_lib.Sublist("BoomerAMG Solver");
            list.Set("Type","BoomerAMG");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("Coarsening type",10);
                solver_list.Set("Aggressive coarsening levels",1);
                solver_list.Set("Relaxation type",8);
                solver_list.Set("Theta",0.25);
                solver_list.Set("Interpolation type",6);
                solver_list.Set("P max",4);
                solver_list.Set("Print level",0);
                solver_list.Set("Dim",1);
                solver_list.Set("Maximum levels",25);
                solver_list.Set("Tolerance",0.0);
                solver_list.Set("Maximum iterations",1);
            }
        }// BoomerAMG Solver

    }// Preconditioner Library

    return ret;
}


}// namespace testhelper
}// namespace parelag
#endif /* _2FORM_HELPERS_HPP__ */
