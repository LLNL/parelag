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


#ifndef PARELAG_AMGESOLVERFACTORY_HPP_
#define PARELAG_AMGESOLVERFACTORY_HPP_

#include "linalg/solver_core/ParELAG_SolverFactory.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{

class AMGeSolverFactory : public SolverFactory
{
public:

    /** \brief Default constructor. */
    AMGeSolverFactory() = default;

    /** \brief Construct from factories.
     *
     *  \param SmootherFactory The factory used to create the pre- and
     *                         post-smoother.
     *  \param CoarseSolverFactory The factory used to create the
     *                             coarse-grid solver.
     */
    AMGeSolverFactory(
        std::shared_ptr<SolverFactory> SmootherFactory,
        std::shared_ptr<SolverFactory> CoarseSolverFactory)
        : PreSmootherFact_{SmootherFactory},
          PostSmootherFact_{std::move(SmootherFactory)},
          CoarseSolverFact_{std::move(CoarseSolverFactory)}
    {}

    /** \brief Construct from factories.
     *
     *  \param PreSmootherFactory The factory used to create the
     *                            pre-smoother.
     *  \param PostSmootherFactory The factory used to create the
     *                             post-smoother.
     *  \param CoarseSolverFactory The factory used to create the
     *                             coarse-grid solver.
     */
    AMGeSolverFactory(
        std::shared_ptr<SolverFactory> PreSmootherFactory,
        std::shared_ptr<SolverFactory> PostSmootherFactory,
        std::shared_ptr<SolverFactory> CoarseSolverFactory)
        : PreSmootherFact_{std::move(PreSmootherFactory)},
          PostSmootherFact_{std::move(PostSmootherFactory)},
          CoarseSolverFact_{std::move(CoarseSolverFactory)}
    {}

private:

    /** \brief Implementation of BuildSolver(). */
    std::unique_ptr<mfem::Solver>
    _do_build_solver(const std::shared_ptr<mfem::Operator>& op,
                     SolverState& state) const override;

    /** \brief Implementation of SetDefaultParameters(). */
    void _do_set_default_parameters() override
    {
        auto& params = GetParameters();
        params.Get<int>("Maximum levels",-1);
        params.Get<std::vector<int>>("Forms",std::vector<int>());
    }

    /** \brief Implementation of Initialize(). */
    void _do_initialize(const ParameterList& Params) override;

    /** \brief Implementation of GetDefaultState(). */
    std::unique_ptr<SolverState> _do_get_default_state() const override
    { return make_unique<NestedSolverState>(); }


private:
    /** \brief Number of desired levels in the hierarchy. */
    mutable int MaxLevels_ = -1;

    /** \brief Block structure information. */
    mutable std::vector<int> Forms_;

    /** \brief Factory for pre-smoother. */
    std::shared_ptr<SolverFactory> PreSmootherFact_;

    /** \brief Factory for post-smoother. */
    std::shared_ptr<SolverFactory> PostSmootherFact_;

    /** \brief Factory for coarse solver. */
    std::shared_ptr<SolverFactory> CoarseSolverFact_;

};// class AMGeSolverFactory
}// namespace parelag

#endif /* PARELAG_AMGESOLVERFACTORY_HPP_ */
