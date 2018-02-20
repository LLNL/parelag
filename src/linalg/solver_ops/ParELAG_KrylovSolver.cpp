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


#include "linalg/solver_ops/ParELAG_KrylovSolver.hpp"

#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"
#include "linalg/utilities/ParELAG_MG_Utils.hpp"

namespace parelag
{

KrylovSolver::KrylovSolver(
    std::shared_ptr<mfem::Operator> A,
    std::shared_ptr<mfem::Solver> Prec,
    const ParameterList& params)
    : A_{std::move(A)},
      Prec_{std::move(Prec)},
      Comm_{mg_utils::GetComm(*A_)},
      Rank_{-1}
{
    PARELAG_ASSERT(Comm_ != MPI_COMM_NULL);

    MPI_Comm_rank(Comm_,&Rank_);

    std::string name = params.Get<std::string>("Solver name");

    // name -> NAME
    std::transform(name.begin(),name.end(),name.begin(),
                   [](const char& a){return std::toupper(a);});

    if ((name == "CG") || (name == "PCG"))
        Solver_ = make_unique<mfem::CGSolver>(Comm_);
    else if (name == "GMRES")
    {
        auto tmp = make_unique<mfem::GMRESSolver>(Comm_);
        tmp->SetKDim(params.IsParameter("Restart size") ?
                     params.Get<int>("Restart size") : 50);
        Solver_ = std::move(tmp);
    }
    else if (name == "FGMRES")
    {
        auto tmp = make_unique<mfem::FGMRESSolver>(Comm_);
        tmp->SetKDim(params.IsParameter("Restart size") ?
                     params.Get<int>("Restart size") : 50);
        Solver_ = std::move(tmp);
    }
    else if (name == "BICGSTAB")
        Solver_ = make_unique<mfem::BiCGSTABSolver>(Comm_);
    else if (name == "MINRES")
        Solver_ = make_unique<mfem::MINRESSolver>(Comm_);

    const bool invalid_solver_type = !Solver_;
    PARELAG_TEST_FOR_EXCEPTION(
        invalid_solver_type,
        std::runtime_error,
        "KrylovSolver::KrylovSolver(...): Bad solver type (\"" <<
        name << "\").\n\n"
        "Valid choices are \"CG\", \"GMRES\", \"FGMRES\", "
        "\"BiCGSTAB\", \"MINRES\".");

    // Set the operator *BEFORE* the preconditioner
    PARELAG_ASSERT(A_);
    Solver_->SetOperator(*A_);

    Solver_->iterative_mode = false;

    if (Prec_)
    {
        Prec_->iterative_mode = false;
        Solver_->SetPreconditioner(*Prec_);
    }

    // Parse parameters (mimic MFEM defaults)
    Solver_->SetPrintLevel(params.IsParameter("Print level") ?
                           params.Get<int>("Print level") : -1);
    Solver_->SetRelTol(params.IsParameter("Relative tolerance") ?
                       params.Get<double>("Relative tolerance") : 0.0);
    Solver_->SetAbsTol(params.IsParameter("Absolute tolerance") ?
                       params.Get<double>("Absolute tolerance") : 0.0);
    Solver_->SetMaxIter(params.IsParameter("Maximum iterations") ?
                        params.Get<int>("Maximum iterations"): 10);

    PrintFinalParagraph_ = params.IsParameter("Print final paragraph") ?
        params.Get<bool>("Print final paragraph") : false;
}


void KrylovSolver::_do_set_operator(
    const std::shared_ptr<mfem::Operator>& op)
{
    // An implementation detail; calls to MFEM's SetOperator methods
    // will forward the operator *by reference* to the SetOperator
    // method of their preconditioner. If that preconditioner is
    // *actually* a parelag::Solver, that call with throw. To prevent
    // that, we trick it by temporarily setting the solver to a fake
    // thing. We cannot just use null since it wants a reference. :/
    class VacuousSolver : public mfem::Solver
    {
    public:
        void SetOperator(const mfem::Operator&) override {}
        void Mult(const mfem::Vector&, mfem::Vector&) const override {}
    };

    A_ = op;

    if (auto prec_tmp = std::dynamic_pointer_cast<Solver>(Prec_))
    {
        VacuousSolver vac;
        Solver_->SetPreconditioner(vac);
        Solver_->SetOperator(*A_);
        Solver_->SetPreconditioner(*Prec_);
    }
    else
        Solver_->SetOperator(*A_);

}

}// namespace parelag
