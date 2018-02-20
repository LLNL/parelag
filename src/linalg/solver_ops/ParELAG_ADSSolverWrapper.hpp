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


#ifndef PARELAG_ADSSOLVERWRAPPER_HPP_
#define PARELAG_ADSSOLVERWRAPPER_HPP_

#include "linalg/solver_core/ParELAG_Solver.hpp"

namespace parelag
{
class DeRhamSequence;

/** \class ADSSolverWrapper
 *  \brief Wrap HYPRE's ADS method.
 */
class ADSSolverWrapper : public mfem::HypreSolver
{
public:

    /** \brief Constructor.
     *
     *  \param A   The matrix
     *  \param seq A DeRhamSequence from which to pull extra
     *             operators.
     *  \param params The parameter list describing the solver.
     */
    ADSSolverWrapper(const std::shared_ptr<mfem::Operator>& A,
                     const DeRhamSequence& seq,
                     ParameterList& params);

    /** \brief Destructor. */
    ~ADSSolverWrapper();

    /** \brief Implicit conversion to HYPRE_Solver. */
    operator HYPRE_Solver() const override { return ads_; }

    /** \brief Wrapper for the setup handle. */
    HYPRE_PtrToParSolverFcn SetupFcn() const override
    { return (HYPRE_PtrToParSolverFcn) HYPRE_ADSSetup; }

    /** \brief Wrapper for the solve handle. */
    HYPRE_PtrToParSolverFcn SolveFcn() const override
    { return (HYPRE_PtrToParSolverFcn) HYPRE_ADSSolve; }


private:

    /** \brief Set the solver parameters. */
    void _do_set_parameters(ParameterList& params);

private:

    /** \brief The primary matrix operator */
    std::shared_ptr<mfem::HypreParMatrix> A_;

    /** \brief Discrete gradient matrix */
    std::shared_ptr<mfem::HypreParMatrix> G_;

    /** \brief Discrete curl matrix */
    std::shared_ptr<mfem::HypreParMatrix> C_;

    /** \brief Nedelec interpolation matrix and its components. */
    std::unique_ptr<mfem::HypreParMatrix> ND_Pi_, ND_Pix_, ND_Piy_, ND_Piz_;

    /** \brief Raviart-Thomas interpolation matrix and its components. */
    std::unique_ptr<mfem::HypreParMatrix> RT_Pi_, RT_Pix_, RT_Piy_, RT_Piz_;

    /** \brief Underlying HYPRE object */
    HYPRE_Solver ads_;

};// class ADSSolverWrapper
}// namespace parelag
#endif /* PARELAG_ADSSOLVERWRAPPER_HPP_ */
