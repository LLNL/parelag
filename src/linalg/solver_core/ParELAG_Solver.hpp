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


#ifndef PARELAG_SOLVER_HPP_
#define PARELAG_SOLVER_HPP_

#include <memory>

#include <mfem.hpp>

#include "utilities/elagError.hpp"
#include "utilities/ParELAG_ParameterList.hpp"

namespace parelag
{

/** \class Solver
 *  \brief A light wrapper around mfem::Solver.
 *
 *  The primary purpose of this class is to change the semantics
 *  surrounding "iterative_mode" and "SetOperator()", the former
 *  lacking clarity and interface and the latter being insufficient
 *  for the way in which ParELAG treats solvers' operators.
 */
class Solver : public mfem::Solver
{
public:

    /** \brief Constructor forwards arguments to the mfem::Solver
     *  constructor.
     */
    template <typename... Ts>
    Solver(Ts&&... Args)
        : mfem::Solver(std::forward<Ts>(Args)...) {}


    /** \brief MFEM's SetOperator method.
     *
     *  \deprecated Prefer the method taking a shared_ptr
     *  instead. Solver objects may need to maintain the lifetime of
     *  the original Operator upon which they are based. This
     *  by-reference method complicates this process, so we
     *  effectively disable it (at runtime, unfortunately) by throwing
     *  an exception.
     */
    void SetOperator(mfem::Operator const&) override final
    {
        constexpr bool this_function_is_stupid = true;
        PARELAG_TEST_FOR_EXCEPTION(
            this_function_is_stupid,
            not_implemented_error,
            "mfem::Solver::SetOperator(): This function is dumb. "
            "It should take a (shared) pointer, not a reference.");
    }

    /** \brief Set whether this is in preconditioner mode (clobber
     *  initial guess) or not (use initial guess).
     *
     *  \param is_preconditioner \c true if the solver should be
     *                           treated as a preconditioner (i.e.,
     *                           clobber the initial guess passed to
     *                           Mult()).
     */
    void IsPreconditioner(bool is_preconditioner) noexcept
    {
        this->iterative_mode = !(is_preconditioner);
    }

    /** \brief Get whether this is in preconditioner mode.
     *
     *  \return \c true if the solver is a precondititioner (i.e., if
     *          it will clobber the initial guess passed to Mult().
     */
    bool IsPreconditioner() const noexcept
    {
        return !(this->iterative_mode);
    }

    /** \brief Set the underlying operator.
     *
     *  \param op The new operator for the solver.
     */
    void SetOperator(std::shared_ptr<mfem::Operator> const& op)
    {
        _do_set_operator(op);
    }

private:

    /** \brief Implementation of SetOperator(). */
    virtual void _do_set_operator(
        std::shared_ptr<mfem::Operator> const& op) = 0;

};// class Solver
}// namespace parelag
#endif /* PARELAG_SOLVER_HPP_ */
