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


#ifndef PARELAG_STRUMPACKSOLVER_IMPL_HPP_
#define PARELAG_STRUMPACKSOLVER_IMPL_HPP_

#include "ParELAG_StrumpackSolver.hpp"

#include "utilities/MemoryUtils.hpp"

namespace
{

/** \brief Map a string to an internal strumpack solver type. */
strumpack::KrylovSolver string_to_solver(std::string str) noexcept
{
    std::transform(str.begin(),str.end(),str.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    if (str == "DIRECT") return strumpack::KrylovSolver::DIRECT;
    if (str == "REFINE") return strumpack::KrylovSolver::REFINE;
    if (str == "PRECONDITIONED GMRES") return strumpack::KrylovSolver::PREC_GMRES;
    if (str == "GMRES") return strumpack::KrylovSolver::GMRES;
    if (str == "PRECONDITIONED BICGSTAB")
        return strumpack::KrylovSolver::PREC_BICGSTAB;
    if (str == "BICGSTAB") return strumpack::KrylovSolver::BICGSTAB;

    // Default
    return strumpack::KrylovSolver::AUTO;
}

/** \brief Map a string to an interal strumpack reordering type. */
strumpack::ReorderingStrategy string_to_reorder(std::string str) noexcept
{
    std::transform(str.begin(),str.end(),str.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    if (str == "GEOMETRIC") return strumpack::ReorderingStrategy::GEOMETRIC;
    if (str == "SCOTCH") return strumpack::ReorderingStrategy::SCOTCH;
    if (str == "PTSCOTCH") return strumpack::ReorderingStrategy::PTSCOTCH;
    if (str == "METIS") return strumpack::ReorderingStrategy::METIS;
    if (str == "PARMETIS") return strumpack::ReorderingStrategy::PARMETIS;
    if (str == "RCM") return strumpack::ReorderingStrategy::RCM;

    // Default
    return strumpack::ReorderingStrategy::METIS;
}

/** \brief Convert error code to string. */
std::string error_code_to_string(strumpack::ReturnCode code) noexcept
{
    switch (code)
    {
    case strumpack::ReturnCode::SUCCESS:
        return "success";
    case strumpack::ReturnCode::MATRIX_NOT_SET:
        return "matrix not set";
    case strumpack::ReturnCode::REORDERING_ERROR:
        return "reordering error";
    default:
        return "unknown";
    }
}
}// namespace

namespace parelag
{

template <typename Scalar,typename Real,typename Ordinal>
StrumpackSolver<Scalar,Real,Ordinal>::
StrumpackSolver(const std::shared_ptr<mfem::Operator>& A)
    : DirectSolver(A->Width(),A->Height(),false)
{
    SetOperator(A);
}


template <typename Scalar,typename Real,typename Ordinal>
void
StrumpackSolver<Scalar,Real,Ordinal>::
SetSolverParameters(const ParameterList& pl)
{
    ParameterList list = pl;

    // The default values are stolen from the StrumpackSparseSolver
    // constructor and/or other strumpack source files

    auto options = Solver_->options();

    // Krylov method

    // Options: AUTO, DIRECT, REFINE, PREC_GMRES, GMRES, PREC_BICGSTAB, BICGSTAB
    options.set_Krylov_solver(
        ::string_to_solver(list.Get("Krylov solver","Direct")));
    options.set_maxit(
        list.Get<int>("Maximum Krylov iterations",500));
    options.set_gmres_restart(list.Get<int>("GMRES restart",30));

    options.set_rel_tol(
        list.Get<real_type>(
            "Relative Krylov tolerance",
            strumpack::default_rel_tol<real_type>()));
    options.set_abs_tol(
        list.Get<real_type>(
            "Absolute Krylov tolerance",
            strumpack::default_abs_tol<real_type>()));

    // Reordering
    options.set_nd_param(
        list.Get<int>("Nested dissection parameter",8));

    // Options: METIS, SCOTCH, GEOMETRIC
    options.set_reordering_method(
        ::string_to_reorder(list.Get("Matrix reordering method","METIS")));

    // Set the Gram-Schmidt type
    {
        std::string gs_type = list.Get("Gram-Schmidt type","Classical");
        std::transform(gs_type.begin(),gs_type.end(),gs_type.begin(),
                       [](unsigned char c) { return std::toupper(c); });

        // CLASSICAL is faster (more scalable) but MODIFIED is stable.
        options.set_GramSchmidt_type(
            (gs_type == "CLASSICAL" ? strumpack::GramSchmidtType::CLASSICAL :
             strumpack::GramSchmidtType::MODIFIED));
    }

    // I have no idea...
    //
    // - job=0: Disable MC64, for many matrices MC64 is not required.
    // - job=1: This is not supported.
    // - job=2: Maximize the smallest diagonal value.
    // - job=3: Same as 2, but using a different algorithm.
    // - job=4: Maximize the sum of the diagonal entries.
    // - job=5: Maximize the product of the diagonal entries
    //          and perform row and column scaling.
    options.set_mc64job(list.Get<int>("MC64 job",0));
    // Differs from strumpack default to avoid extra work by default
    // in parelag. If things don't work, try this guy.


    // HSS parameters
    if (list.Get<bool>("Use HSS",false) == true)
        options.enable_HSS();
    else
        options.disable_HSS();

    // FIXME (trb 05/16/17): With the restructuring, I'm not sure if
    // this went away or was renamed...
    //options.set_minimum_HSS_size(list.Get<int>("Minimum HSS size",512));
    options.HSS_options().set_rel_tol(
        list.Get<real_type>(
            "Relative compression tolerance",
            strumpack::HSS::default_HSS_rel_tol<real_type>()));
    options.HSS_options().set_abs_tol(
        list.Get<real_type>(
            "Absolute compression tolerance",
            strumpack::HSS::default_HSS_abs_tol<real_type>()));

    // Verbosity
    options.set_verbose(list.Get<bool>("Verbose",false));
}

template <typename Scalar,typename Real,typename Ordinal>
void
StrumpackSolver<Scalar,Real,Ordinal>::
SetOperator(const std::shared_ptr<mfem::Operator>& A)
{
    // Check that we have a HypreParMatrix
    auto a_mfem = std::dynamic_pointer_cast<mfem::HypreParMatrix>(A);
    PARELAG_TEST_FOR_EXCEPTION(
        !a_mfem,
        std::logic_error,
        "StrumpackSolver::StrumpackSolver(...): Input matrix is not "
        "an mfem::HypreParMatrix. Cannot set as StrumpackSolver's operator.");

    A_ = A;

    // Check/set the communicator. If our communicator changes, we
    // need a new solver object. Note that this should set the initial
    // solver, unless the matrix has MPI_COMM_NULL as its comm
    if (Comm_ != a_mfem->GetComm())
    {
        Comm_ = a_mfem->GetComm();
        Solver_ = make_unique<solver_type>(Comm_,false);
    }

    // Make sure that we've created a solver by this point
    if (!Solver_)
        Solver_ = make_unique<solver_type>(Comm_,false);

    // Now set the solver matrix
    hypre_ParCSRMatrix * a_parcsr = *a_mfem;

    // This makes a deep copy of the whole matrix, so technically "A"
    // is not required beyond here, but since it's already shared,
    // keeping it around really costs nothing that we aren't already
    // paying...
    Solver_->set_MPIAIJ_matrix(
        hypre_ParCSRMatrixNumRows(a_parcsr),
        hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(a_parcsr)),
        hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(a_parcsr)),
        hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(a_parcsr)),
        hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(a_parcsr)),
        hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(a_parcsr)),
        hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(a_parcsr)),
        hypre_ParCSRMatrixColMapOffd(a_parcsr));
}
template <typename Scalar,typename Real,typename Ordinal>
void
StrumpackSolver<Scalar,Real,Ordinal>::_do_factor()
{
    auto ret = Solver_->reorder();

    PARELAG_TEST_FOR_EXCEPTION(
        ret != strumpack::ReturnCode::SUCCESS,
        std::logic_error,
        "StrumpacKSolver::Factor(...): reorder() exited with error: " <<
        error_code_to_string(ret) << ".");

    ret = Solver_->factor();
    PARELAG_TEST_FOR_EXCEPTION(
        ret != strumpack::ReturnCode::SUCCESS,
        std::logic_error,
        "StrumpacKSolver::Factor(...): factor() exited with error: " <<
        error_code_to_string(ret) << ".");

    IsFactored_ = true;
}

template <typename Scalar,typename Real,typename Ordinal>
void
StrumpackSolver<Scalar,Real,Ordinal>::
Mult(const mfem::Vector& B, mfem::Vector& X) const
{
    PARELAG_ASSERT(IsFactored_);

    auto ret = Solver_->solve(B.GetData(),X.GetData(),this->iterative_mode);

    PARELAG_TEST_FOR_EXCEPTION(
        ret != strumpack::ReturnCode::SUCCESS,
        std::logic_error,
        "StrumpacKSolver::Mult(...): solve() exited with error: " <<
        error_code_to_string(ret) << ".");

}

}// namespace parelag
#endif /* PARELAG_STRUMPACKSOLVER_IMPL_HPP_ */
