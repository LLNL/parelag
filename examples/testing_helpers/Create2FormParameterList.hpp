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

std::unique_ptr<ParameterList> Create2FormTestParameters()
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
        prob_params.Set("Start level",0);
        prob_params.Set("Stop level",1);
        prob_params.Set("Frequency",1.0);
        prob_params.Set("Coefficient subdomains",1);
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

        // PCG with Auxiliary Space Preconditioner
        {
            auto& list = prec_lib.Sublist(
                "PCG with Auxiliary Space Preconditioner");

            list.Set("Type", "Krylov");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("Solver name", "PCG");
                solver_list.Set("Preconditioner", "AMGe-HIP-GS_2");
                solver_list.Set("Print level", 1);
                solver_list.Set("Maximum iterations", 300);
                solver_list.Set("Relative tolerance", 1e-6);
                solver_list.Set("Absolute tolerance", 1e-6);
                solver_list.Set("Restart size", 50);
            }
        }// PCG with Auxiliary Space Preconditioner

        // AMGe-HIP-GS_1
        {
            auto& list = prec_lib.Sublist("AMGe-HIP-GS_2");
            list.Set("Type", "AMGe");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("Maximum levels", -1);
                solver_list.Set("Forms", std::vector<int>{2});
                solver_list.Set("PreSmoother", "Hiptmair-GS-GS");
                solver_list.Set("PostSmoother", "Hiptmair-GS-GS");
                solver_list.Set("Coarse solver", "PCG-ADS");
                solver_list.Set("Cycle type", "V-cycle");
            }
        }// AMGe-HIP-GS_1

        // Hiptmair-GS-GS
        {
            auto& list = prec_lib.Sublist("Hiptmair-GS-GS");
            list.Set("Type", "Hiptmair");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("Primary Smoother", "Gauss-Seidel");
                solver_list.Set("Auxiliary Smoother", "Gauss-Seidel");
            }
        }// Hiptmair-GS-GS

        // PCG-ADS
        {
            auto& list = prec_lib.Sublist("PCG-ADS");
            list.Set("Type", "Krylov");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("Solver name", "PCG");
                solver_list.Set("Preconditioner", "ADS Solver");
                solver_list.Set("Print level", -1);
                solver_list.Set("Maximum iterations", 3);
                solver_list.Set("Relative tolerance", 1e-4);
                solver_list.Set("Absolute tolerance", 1e-4);
                solver_list.Set("Restart size", 50);
            }
        }// PCG-AMS

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

        // ADS Solver
        {
            auto& list = prec_lib.Sublist("ADS Solver");
            list.Set("Type", "ADS");
            {
                auto& solver_list = list.Sublist("Solver Parameters");
                solver_list.Set("Cycle type", 11);
                solver_list.Set("Tolerance", 0.0);
                solver_list.Set("Maximum iterations", 1);
                solver_list.Set("Print level", 1);
                solver_list.Set("Relaxation type", 2);
                solver_list.Set("Relaxation sweeps", 1);
                solver_list.Set("Relaxation weight", 1.0);
                solver_list.Set("Relaxation omega", 1.0);
                {
                    auto& sub_solver_list = solver_list.Sublist("AMS Parameters");
                    sub_solver_list.Set("Cycle type", 14);
                    sub_solver_list.Set("Relaxation type", 2);
                    sub_solver_list.Set("Relaxation sweeps", 1);
                    sub_solver_list.Set("Relaxation weight", 1.0);
                    sub_solver_list.Set("Relaxation omega", 1.0);
                    sub_solver_list.Set("beta_is_zero", false);
                    {
                        auto& sub_sub_solver_list =
                            sub_solver_list.Sublist("PtAP AMG Parameters");

                        sub_sub_solver_list.Set("Coarsening type", 10);
                        sub_sub_solver_list.Set(
                            "Aggressive coarsening levels", 1);
                        sub_sub_solver_list.Set("Relaxation type", 6);
                        sub_sub_solver_list.Set("Theta", 0.25);
                        sub_sub_solver_list.Set("Interpolation type", 6);
                        sub_sub_solver_list.Set("P max", 4);
                        sub_sub_solver_list.Set("Print level", 0);
                        sub_sub_solver_list.Set("Dim", 1);
                        sub_sub_solver_list.Set("Maximum levels", 25);
                    }
                }

                {
                    auto& sub_solver_list =
                        solver_list.Sublist("AMG Parameters");

                    sub_solver_list.Set("Coarsening type", 10);
                    sub_solver_list.Set("Aggressive coarsening levels", 1);
                    sub_solver_list.Set("Relaxation type", 6);
                    sub_solver_list.Set("Theta", 0.25);
                    sub_solver_list.Set("Interpolation type", 6);
                    sub_solver_list.Set("P max", 4);
                    sub_solver_list.Set("Print level", 0);
                    sub_solver_list.Set("Dim", 1);
                    sub_solver_list.Set("Maximum levels", 25);
                }
            }
        }// ADS Solver

    }// Preconditioner Library

    return ret;
}


}// namespace testhelper
}// namespace parelag
#endif /* _2FORM_HELPERS_HPP__ */
