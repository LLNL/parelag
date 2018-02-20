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


#ifndef PARELAG_ADSSOLVERFACTORY_HPP_
#define PARELAG_ADSSOLVERFACTORY_HPP_

#include "linalg/solver_core/ParELAG_SolverFactory.hpp"

namespace parelag
{

/** \class ADSSolverFactory
 *  \brief Constructs ADSSolverWrapper objects.
 */
class ADSSolverFactory : public SolverFactory
{
public:
    /** \brief Default constructor.
     *
     *  \param pl  The parameters for this factory.
     */
    ADSSolverFactory(ParameterList const& pl = ParameterList())
    { SetParameters(pl); }

private:
    /** \name SolverFactory interface */
    ///@{

    /** \brief Implementation of BuildSolver(). */
    std::unique_ptr<mfem::Solver> _do_build_solver(
        const std::shared_ptr<mfem::Operator>& op,
        SolverState& state ) const override;

    /** \brief Implementation of SetDefaultParameters(). */
    void _do_set_default_parameters() override;

    /** \brief Implementation of Initialize(). */
    void _do_initialize(const ParameterList& Params) override;

    ///@}
};// class ADSSolverFactory
}// namespace parelag
#endif /* PARELAG_ADSSOLVERFACTORY_HPP_ */
