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


#ifndef PARELAG_BOOMERAMGSOLVERWRAPPER_HPP_
#define PARELAG_BOOMERAMGSOLVERWRAPPER_HPP_

#include "linalg/solver_core/ParELAG_Solver.hpp"

namespace parelag
{

/** \class BoomerAMGSolverWrapper
 *  \brief Wrap MFEM's wrapper of HYPRE's BoomerAMG method.
 *
 *  This class adds lifetime preserving semantics to the solver.
 */
class BoomerAMGSolverWrapper : public Solver
{
public:

    /** \brief Constructor.
     *
     *  \param A   The matrix
     *  \param params The parameter list describing the solver.
     */
    BoomerAMGSolverWrapper(std::shared_ptr<mfem::Operator> const& A,
                           ParameterList& Params);

    /** \brief Apply the operator to a vector. */
    void Mult(const mfem::Vector& rhs, mfem::Vector& sol) const override
    {
        PARELAG_ASSERT(amg_.iterative_mode);
        if (this->IsPreconditioner())
            sol = 0.0;
        amg_.Mult(rhs,sol);
    }

    /** \brief Apply the transpose of the operator to a vector. */
    void MultTranspose(
        const mfem::Vector& rhs, mfem::Vector& sol) const override
    {
        PARELAG_ASSERT(amg_.iterative_mode);
        if (this->IsPreconditioner())
            sol = 0.0;
        amg_.MultTranspose(rhs,sol);
    }

private:

    /** \brief Set the solver's parameters. */
    void _do_set_parameters(ParameterList& Params);

    void _do_set_operator(const std::shared_ptr<mfem::Operator>& op) override;

private:

    /** \brief The underlying system matrix */
    std::shared_ptr<mfem::HypreParMatrix> A_;

    /** \brief The map from DOF to FunctionID (for blocked problems). */
    std::vector<HYPRE_Int> DofToFunction_;

    /** \brief The underlying MFEM object. */
    mfem::HypreBoomerAMG amg_;

};// class BoomerAMGSolverWrapper
}// namespace parelag
#endif /* PARELAG_BOOMERAMGSOLVERWRAPPER_HPP_ */
