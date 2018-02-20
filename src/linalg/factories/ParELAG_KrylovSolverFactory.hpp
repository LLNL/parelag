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


#ifndef PARELAG_KRYLOVSOLVERFACTORY_HPP_
#define PARELAG_KRYLOVSOLVERFACTORY_HPP_

#include "linalg/solver_core/ParELAG_SolverFactory.hpp"

namespace parelag
{

/** \class KrylovSolverFactory
 *  \brief Factory for building MFEM's Krylov solvers.
 */
class KrylovSolverFactory : public SolverFactory
{
public:

    /** \brief Default constructor. */
    KrylovSolverFactory() = default;

private:

    /** \name SolverFactory interface */
    ///@{

    std::unique_ptr<mfem::Solver> _do_build_solver(
        const std::shared_ptr<mfem::Operator>& op,
        SolverState& state ) const override;

    void _do_set_default_parameters() override;

    void _do_initialize(const ParameterList& Params) override;

    std::unique_ptr<SolverState> _do_get_default_state() const override
    { return make_unique<NestedSolverState>(); }

    ///@}

private:

    /** \brief The SolverFactory for building the preconditioner. */
    std::shared_ptr<SolverFactory> Prec_Factory_;

    /** \brief Whether the setup of the preconditioner should be timed. */
    bool DoPrecSetupTiming_ = false;

};// class KrylovSolverFactory
}// namespace parelag
#endif /* PARELAG_KRYLOVSOLVERFACTORY_HPP_ */
