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


#ifndef PARELAG_DIRECTSOLVER_HPP_
#define PARELAG_DIRECTSOLVER_HPP_

#include "ParELAG_Solver.hpp"
#include "utilities/ParELAG_Meta.hpp"

namespace parelag
{

/** \class DirectSolver
 *  \brief Support for sparse direct solvers.
 */
class DirectSolver : public Solver
{
public:

    /** \brief Forwarding constructor.
     *
     *  All arguments are forwarded to the Solver constructor.
     */
    template <typename... Ts>
    DirectSolver(Ts&&... Args)
        : Solver(std::forward<Ts>(Args)...) {}

    /** \brief Destructor. */
    virtual ~DirectSolver() {}

    /** \brief Compute the (symbolic and) numerical factorization of
     *         the operator.
     */
    void Factor()
    {
        _do_factor();
    }

    /** \brief Set solver parameters using a ParameterList.
     *
     *  \param pl The parameter list with the solver's parameters.
     */
    void SetSolverParameters(ParameterList const& pl)
    {
        _do_set_solver_parameters(pl);
    }

private:

    /** \brief Implementation of Factor(). */
    virtual void _do_factor() = 0;

    /** \brief Implementation of SetSolverParameters(). */
    virtual void _do_set_solver_parameters(ParameterList const&) {}

};// class DirectSolver
}// namespace parelag
#endif /* PARELAG_DIRECTSOLVER_HPP_ */
