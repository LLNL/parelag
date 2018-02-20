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


#ifndef PARELAG_BLOCK2X2GAUSSSEIDELSOLVERFACTORY_HPP_
#define PARELAG_BLOCK2X2GAUSSSEIDELSOLVERFACTORY_HPP_

#include "linalg/factories/ParELAG_SchurComplementFactory.hpp"
#include "linalg/solver_core/ParELAG_BlockSolverFactory.hpp"
#include "linalg/solver_ops/ParELAG_BlockTriangularSolver.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{

/** \class Block2x2GaussSeidelSolverFactory
 *  \brief SolverFactory for creating BlockTriangularSolvers for 2x2
 *         systems.
 *
 *  The use of $S$ instead of $A_11$ is optional.
 */
class Block2x2GaussSeidelSolverFactory : public BlockSolverFactory
{
public:
    /** \name Constructors and destructor */
    ///@{

    /** \brief Constructor.
     *
     *  \param params The parameters for this factory.
     */
    Block2x2GaussSeidelSolverFactory(
        const ParameterList& params = ParameterList());

    /** \brief Constructor taking all factories.
     *
     *  \param a00_inverse_factory The factory for computing
     *                             $A_{00}^{-1}$.
     *  \param a11_inverse_factory The factory for computing
     *                             $A_{11}^{-1}$ or $S^{-1}$.
     *  \param s_factory The factory for computing $S$, i.e. the (1,1)
     *                   block of the preconditioner.
     *  \param params The parameters for this factory.
     */
    Block2x2GaussSeidelSolverFactory(
        std::shared_ptr<SolverFactory> a00_inverse_factory,
        std::shared_ptr<SolverFactory> a11_inverse_factory,
        std::shared_ptr<SchurComplementFactory> s_factory,
        const ParameterList& params = ParameterList());

    /** \brief Destructor. */
    ~Block2x2GaussSeidelSolverFactory() = default;

    ///@}

private:

    /** \name SolverFactory interface */
    ///@{

    /** \brief Implementation of BuildBlockSolver(). */
    std::unique_ptr<mfem::Solver> _do_build_block_solver(
        const std::shared_ptr<MfemBlockOperator>& op,
        SolverState& state ) const override;

    /** \brief Implementation of SetDefaultParameters(). */
    void _do_set_default_parameters() override;

    /** \brief Implementation of Initialize(). */
    void _do_initialize(const ParameterList& Params) override;

    /** \brief Implementation of GetDefaultState(). */
    std::unique_ptr<SolverState> _do_get_default_state() const override
    { return make_unique<NestedSolverState>(); }

    ///@}

private:

    /** \name Private data members */
    ///@{

    /** \brief The solver factory for the (0,0) block. */
    std::shared_ptr<SolverFactory> InvA00_Fact_;

    /** \brief The solver factory for the (1,1) block. */
    std::shared_ptr<SolverFactory> InvA11_Fact_;

    /** \brief The factory for creating Schur complements.*/
    std::shared_ptr<SchurComplementFactory> S_Fact_;

    /** \brief Which triangle should the block Gauss-Seidel solver use. */
    BlockTriangularSolver::Triangle tri_;

    /** \brief Whether to use "-S" in the (1,1) block. */
    bool UseNegativeS_ = true;

    ///@}

};// class Block2x2GaussSeidelSolverFactory
}// namespace parelag
#endif /* PARELAG_BLOCK2X2GAUSSSEIDELSOLVERFACTORY_HPP_ */

// TODO (trb 04/06/2016): Add support for implicitly remaping blocked
// systems into block-2x2 operators. E.g. have some data that looks
// like [[0 2],[1 3]], which would imply that \tilde{A00} is blocks
// from original rows/cols 0 and 2, etc.
