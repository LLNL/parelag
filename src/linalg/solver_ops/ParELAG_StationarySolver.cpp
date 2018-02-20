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


#include "linalg/solver_ops/ParELAG_StationarySolver.hpp"

#include "linalg/utilities/ParELAG_MG_Utils.hpp"
#include "utilities/MPIDataTypes.hpp"

namespace parelag
{

StationarySolver::StationarySolver(
    std::shared_ptr<mfem::Operator> op,
    std::shared_ptr<mfem::Solver> solver,
    double rel_tol, double abs_tol, size_t max_its,
    bool print_iters)
    : Solver{op->NumCols(),op->NumRows(),true},
      Op_{std::move(op)},
      Solver_{std::move(solver)},
      Residual_{Op_->Height()},
      Correction_{Op_->Width()},
      Tmp_{Op_->Height()},
      RelativeTol_{rel_tol},
      AbsoluteTol_{abs_tol},
      MaxIts_{max_its},
      Comm_{mg_utils::GetComm(*Op_)},
      PrintIterations_{print_iters}
{}

void StationarySolver::Mult(const mfem::Vector& rhs, mfem::Vector& sol) const
{
    Solver_->iterative_mode = false;

    int myid;
    MPI_Comm_rank(Comm_,&myid);

    if (this->IsPreconditioner())
    {
        Residual_ = rhs;
        sol = 0.;
    }
    else
    {
        Op_->Mult(sol,Residual_);
        Residual_ *= -1.0;
        Residual_ += rhs;
    }

    double norm_k_1;
    {
        double norm_k_1_loc = Residual_.Norml2() * Residual_.Norml2();

        MPI_Allreduce(&norm_k_1_loc,&norm_k_1,1,GetMPIType(norm_k_1_loc),
                      MPI_SUM,Comm_);
        norm_k_1 = std::sqrt(norm_k_1);
    }

    if (PrintIterations_ && !myid)
        std::cout << "    Iteration 0: ||r|| = " << norm_k_1 << std::endl;

    size_t its = 0;
    double norm_ratio = 1.0;
    double norm_k;
    bool converged = false;

    while (its < MaxIts_)
    {
        // Increase the iterations counter
        ++its;

        Correction_ = 0.0; Tmp_ = 0.0;

        if (false)
        {
            std::cout << "        ||R|| = " << Residual_.Norml2()
                      << std::endl;
        }

        // Compute correction
        Solver_->Mult(Residual_,Correction_);

        if (false)
        {
            std::cout << "        ||C|| = " << Correction_.Norml2()
                      << std::endl;
        }

        // Update solution
        sol += Correction_;

        // Update residual
        Op_->Mult(Correction_,Tmp_);
        Residual_ -= Tmp_;

        // Report stats
        {
            double norm_loc = Residual_.Norml2() * Residual_.Norml2();

            MPI_Allreduce(&norm_loc,&norm_k,1,GetMPIType(norm_loc),
                          MPI_SUM,Comm_);
            norm_k = std::sqrt(norm_k);
        }

        norm_ratio *= norm_k / norm_k_1;

        if (PrintIterations_ && !myid)
            std::cout << "    Iteration " << its << ": ||r|| = "
                      << norm_k << "  (" << norm_ratio << ")" << std::endl;

        // Check convergence
        if ((norm_k < AbsoluteTol_) || (norm_ratio < RelativeTol_))
        {
            converged = true;
            break;
        }

        // Check stagnation
        if (norm_k / norm_k_1 == 1.0)
        {
            if (!myid)
                std::cout << "WARNING: Computed a zero correction! Stopping..."
                          << std::endl;
            break;
        }

        norm_k_1 = norm_k;
    }

    if (PrintIterations_ && !myid)
        std::cout << '\n' << std::string(50,'*') << '\n'
                  << "*  Solver Status: "
                  << (converged ? "Converged" : "Not converged") << '\n'
                  << "*  Solver Iterations: " << its << '\n'
                  << "*  Final norm: " << norm_k << '\n'
                  << std::string(50,'*') << std::endl;
}

void StationarySolver::MultTranspose(
    const mfem::Vector&, mfem::Vector&) const
{
    PARELAG_NOT_IMPLEMENTED();
}

void StationarySolver::_do_set_operator(
    const std::shared_ptr<mfem::Operator>& op)
{
    Op_ = op;
    Residual_.SetSize(Op_->Height());
    Correction_.SetSize(Op_->Width());
    Tmp_.SetSize(Op_->Height());
}

}// namespace parelag
