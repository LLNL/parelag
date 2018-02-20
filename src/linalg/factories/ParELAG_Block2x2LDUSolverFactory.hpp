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


#ifndef PARELAG_BLOCK2X2LDUSOLVERFACTORY_HPP_
#define PARELAG_BLOCK2X2LDUSOLVERFACTORY_HPP_

#include "linalg/factories/ParELAG_SchurComplementFactory.hpp"
#include "linalg/solver_core/ParELAG_BlockSolverFactory.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{

/** \class Block2x2LDUSolverFactory
 *  \brief Factory for creating Block2x2LDUInverseOperator objects.
 *
 *  See Block2x2LDUInverseOperator for discussion of the solver.
 */
class Block2x2LDUSolverFactory : public BlockSolverFactory
{
public:
    /** \name Constructors and destructor */
    ///@{

    /** \brief Constructor to build from parameter list
     *
     *  \param params The parameters for this factory.
     */
    Block2x2LDUSolverFactory(
        const ParameterList& params = ParameterList());

    /** \brief Constructor taking all factories.
     *
     *  \param invA00_1 The factory to create the inverse for the
     *                  first application of \f$A_{00}^{-1}\f$ (i.e.,
     *                  \f$\hat{A}_{00}^{-1}\f$).
     *  \param invA00_2 The factory to create the inverse for the
     *                  second application of \f$A_{00}^{-1}\f$ (i.e.,
     *                  \f$\bar{A}_{00}^{-1}\f$).
     *  \param invA00_3 The factory to create the inverse for the
     *                  third application of \f$A_{00}^{-1}\f$ (i.e.,
     *                  \f$\tilde{A}_{00}^{-1}\f$).
     *  \param invS The factory to create the inverse for \f$S\f$.
     *  \param s_factory The factory to create the (approximate) Schur
     *                   complement operator.
     *  \param params The parameters for this factory.
     */
    Block2x2LDUSolverFactory(
        std::shared_ptr<SolverFactory> a00_1_inverse_factory,
        std::shared_ptr<SolverFactory> a00_2_inverse_factory,
        std::shared_ptr<SolverFactory> a00_3_inverse_factory,
        std::shared_ptr<SolverFactory> s_inverse_factory,
        std::shared_ptr<SchurComplementFactory> s_factory,
        const ParameterList& params = ParameterList());

    /** \brief Destructor. */
    ~Block2x2LDUSolverFactory() = default;

    ///@}

private:
    /** \name BlockSolverFactory interface*/
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

   /** \brief The factory to create the inverse for the first
     *         application of \f$A_{00}^{-1}\f$ (i.e.,
     *         \f$\hat{A}_{00}^{-1}\f$).*/
    std::shared_ptr<SolverFactory> InvA00_1_Fact_;


    /** \brief The factory to create the inverse for the second
     *         application of \f$A_{00}^{-1}\f$ (i.e.,
     *         \f$\bar{A}_{00}^{-1}\f$).*/
    std::shared_ptr<SolverFactory> InvA00_2_Fact_;

    /** \brief The factory to create the inverse for the third
     *         application of \f$A_{00}^{-1}\f$ (i.e.,
     *         \f$\tilde{A}_{00}^{-1}\f$).*/
    std::shared_ptr<SolverFactory> InvA00_3_Fact_;

    /** \brief The factory to create the inverse for the approximate
     *         Schur complement. */
    std::shared_ptr<SolverFactory> InvS_Fact_;

    /** \brief The factory to create the approximate Schur
     *         complement. */
    std::shared_ptr<SchurComplementFactory> S_Fact_;

    /** \brief The underrelaxation parameter. */
    double Alpha_ = 1.0;

    /** \brief The weighting parameter on the residual update. */
    double Damping_Factor_ = 1.0;

    /** \brief Whether to use "-S" in the (1,1) block. */
    bool UseNegativeS_ = true;

};// class Block2x2LDUSolverFactory
}// namespace parelag
#endif /* PARELAG_BLOCK2X2LDUSOLVERFACTORY_HPP_ */

// TODO (trb 04/06/2016): Add support for implicitly remaping blocked
// systems into block-2x2 operators. E.g. have some data that looks
// like [[0 2],[1 3]], which would imply that \tilde{A00} is blocks
// from original rows/cols 0 and 2, etc.
