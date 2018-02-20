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


#ifndef PARELAG_BLOCKSOLVERFACTORY_HPP_
#define PARELAG_BLOCKSOLVERFACTORY_HPP_

#include <memory>

#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"
#include "ParELAG_SolverFactory.hpp"

namespace parelag
{

/** \class BlockSolverFactory
 *  \brief An interface for building solvers for blocked operators.
 */
class BlockSolverFactory : public SolverFactory
{
public:

    /** \brief Build a solver for a blocked system
     *
     *  \param op     The blocked operator.
     *  \param state  The persistent state of the solver.
     */
    std::unique_ptr<mfem::Solver> BuildBlockSolver(
        const std::shared_ptr<MfemBlockOperator>& op,
        SolverState& state ) const
    {
        return _do_build_block_solver(op,state);
    }

private:

    virtual std::unique_ptr<mfem::Solver> _do_build_solver(
        const std::shared_ptr<mfem::Operator>& op,
        SolverState& state ) const override
    {
        auto parelag_op = std::dynamic_pointer_cast<MfemBlockOperator>(op);
        PARELAG_ASSERT(parelag_op);
        return this->BuildBlockSolver(parelag_op,state);
    }

    virtual std::unique_ptr<mfem::Solver> _do_build_block_solver(
        const std::shared_ptr<MfemBlockOperator>& op,
        SolverState& state ) const = 0;

};// class BlockSolverFactory
}// namespace parelag
#endif /* PARELAG_BLOCKSOLVERFACTORY_HPP_ */
