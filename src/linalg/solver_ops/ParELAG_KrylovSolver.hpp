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


#ifndef PARELAG_KRYLOVSOLVER_HPP_
#define PARELAG_KRYLOVSOLVER_HPP_

#include <memory>

#include "linalg/solver_core/ParELAG_Solver.hpp"

namespace parelag
{

/** \class KrylovSolver
 *  \brief Wraps MFEM's Krylov solvers.
 *
 *  This essentially wraps all of the MFEM Krylov methods into one
 *  interface. Solvers are keyed by name as follows:
 *
 *  Solver Name | Solver Type
 *  ----------- | --------------------
 *  "CG"        | mfem::CGSolver
 *  "PCG"       | mfem::CGSolver
 *  "GMRES"     | mfem::GMRESSolver
 *  "FGMRES"    | mfem::FGMRESSolver
 *  "MINRES"    | mfem::MINRESSolver
 *  "BiCGStab"  | mfem::BiCGSTABSolver
 *
 *  \note There is no support for accessing the HYPRE Krylov methods
 *        because they require Hypre objects as
 *        preconditioners. Neither mfem::Solver nor parelag::Solver
 *        objects are inherently Hypre objects, so adding support for
 *        this would be difficult and has not been pursued at this
 *        time.
 */
class KrylovSolver : public Solver
{
public:

    /** \brief Construct from an operator, a preconditioner, and some
     *         parameters.
     *
     *  \param A The system operator.
     *  \param Prec The preconditioner to use.
     *  \param params A list of parameters; must include "Solver
     *                name" (see above for possible values).
     */
    KrylovSolver(const std::shared_ptr<mfem::Operator> A,
                 const std::shared_ptr<mfem::Solver> Prec,
                 const ParameterList& params = ParameterList());

    /** \brief Apply the Krylov solver to the system.
     *
     *  \param b The system right-hand side.
     *  \param x The initial guess.
     */
    void Mult(const mfem::Vector& b, mfem::Vector& x) const override
    {
        if (this->IsPreconditioner())
            x = 0.0;
        Solver_->Mult(b,x);
        if (PrintFinalParagraph_ && !Rank_)
            _do_print_final_paragraph();
    }

    /** \brief Apply the... transpose? of a Krylov solver to the
     *         system.
     *
     *  \warning This almost certainly makes no sense and will
     *           probably fail. But it makes code-sense, so why not
     *           include it?  But yeah, almost definitely going to
     *           fail.
     *
     *  \param b The system right-hand side.
     *  \param x The initial guess.
     */
    void MultTranspose(const mfem::Vector& b, mfem::Vector& x) const override
    {
        if (this->IsPreconditioner())
            x = 0.0;
        Solver_->MultTranspose(b,x);
        if (PrintFinalParagraph_ && !Rank_)
            _do_print_final_paragraph();
    }

private:

    /** \brief Print convergence information to std::cout. */
    void _do_print_final_paragraph() const
    {
        std::cout << '\n' << std::string(50,'*') << '\n'
                  << "*  Solver Status: "
                  << (Solver_->GetConverged()?"Converged":"Not converged")
                  << '\n'
                  << "*  Solver Iterations: " << Solver_->GetNumIterations()
                  << '\n'
                  << "*  Final norm: " << Solver_->GetFinalNorm() << '\n'
                  << std::string(50,'*') << std::endl;
    }

private:

    /** \brief Handle the calls to SetOperator of the underlying objects.
     *
     *  \warning Calling Solver_->SetOperator(*op) will reset the
     *  operator for the preconditioner too. We do not want this to
     *  happen. So this temporarily shifts the preconditioner to an
     *  invalid target, and does NOT allow the preconditioner's
     *  operator to change.
     */
    void _do_set_operator(const std::shared_ptr<mfem::Operator>& op) override;

private:

    /** \brief The underlying operator. */
    std::shared_ptr<mfem::Operator> A_;

    /** \brief The preconditioner. */
    std::shared_ptr<mfem::Solver> Prec_;

    /** \brief The actual MFEM solver. */
    std::unique_ptr<mfem::IterativeSolver> Solver_;

    /** \brief The MPI communicator for \c A (for printing). */
    MPI_Comm Comm_;

    /** \brief This process's rank in \c Comm_. */
    int Rank_;

    /** \brief Whether or not to print the convergence info. */
    bool PrintFinalParagraph_;

};// class KrylovSolver
}// namespace parelag
#endif /* PARELAG_KRYLOVSOLVER_HPP_ */
