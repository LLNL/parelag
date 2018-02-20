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


#ifndef PARELAG_BRAMBLEPASCIAKFACTORY_HPP_
#define PARELAG_BRAMBLEPASCIAKFACTORY_HPP_

#include "linalg/solver_core/ParELAG_BlockSolverFactory.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{

/** \class BramblePasciakFactory
 *  \brief Creates the Bramble-Pasciak transformation and returns a
 *         solver based on it.
 *
 *  See BramblePasciakTransformation for more details about this
 *  solver.
 */
class BramblePasciakFactory : public BlockSolverFactory
{
public:

    /** \name Constructors and destructor */
    ///@{

    /** \brief Default constructor with optional parameter list
     *
     *  \param params The parameters for this factory.
     */
    BramblePasciakFactory(const ParameterList& params = ParameterList())
    {
        SetParameters(params);
    }

    /** \brief Constructor taking a solver factory and a parameter list.
     *
     *  \param fact SolverFactory to be used to create the Solver for
     *              the transformed system.
     *  \param params The parameters for this factory.
     */
    BramblePasciakFactory(std::shared_ptr<SolverFactory> fact,
                          const ParameterList& params = ParameterList())
        : SolverFact_{std::move(fact)}
    {
        SetParameters(params);
    }

    ///@}

private:

    /** \name SolverFactory interface */
    ///@{

    std::unique_ptr<mfem::Solver> _do_build_block_solver(
        const std::shared_ptr<MfemBlockOperator>& op,
        SolverState& state) const override;

    void _do_set_default_parameters() override;

    void _do_initialize(const ParameterList& Params) override;

    std::unique_ptr<SolverState> _do_get_default_state() const override
    { return make_unique<NestedSolverState>(); }

    ///@}

private:

    /** \brief The SolverFactory for creating the Solver for the
     *         transformed system.
     */
    std::shared_ptr<SolverFactory> SolverFact_;

};// class BramblePasciakFactory
}// namespace parelag
#endif /* PARELAG_BRAMBLEPASCIAKFACTORY_HPP_ */
