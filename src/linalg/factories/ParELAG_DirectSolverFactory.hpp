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


#ifndef PARELAG_DIRECTSOLVERFACTORY_HPP_
#define PARELAG_DIRECTSOLVERFACTORY_HPP_

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <mfem.hpp>

#include "linalg/solver_core/ParELAG_DirectSolver.hpp"
#include "linalg/solver_core/ParELAG_SolverFactory.hpp"

#include "utilities/ParELAG_ParameterList.hpp"
#include "utilities/ParELAG_Factory.hpp"

namespace parelag
{

/** \class DirectSolverFactory
 *  \brief SolverFactory-compliant factory for building various
 *  sparse direct solvers.
 *
 *  Sometimes, the use of a sparse direct solver can be good for
 *  sanity checking, debugging, or perhaps the problem is actually
 *  small enough for it to be at least a semi-sensible option. For
 *  these cases, we have support for the following solvers:
 *
 *    1. Sequential SuperLU ("superlu"): For HypreParMatrix operators in
 *       parallel, this will gather the whole matrix to each processor,
 *       and each will solve. No redistribution communication is
 *       necessary once the problem has been gathered. For SparseMatrix
 *       operators, the local problem (if square) will be solved.
 *
 *    2. SuperLU_DIST ("superludist"): Requires HypreParMatrix
 *       operators. Currently, only distributed solves are supported,
 *       but we could support a "globally replicated" solve, as both are
 *       supported by the SuperLU_DIST interface.
 *
 *    3. Strumpack ("strumpack"): Requires HypreParMatrix
 *       operators. This is a distributed, threaded solve. I believe
 *       there is also a "globally replicated" version of this
 *       solver. We don't use it if there is such a thing.
 */
class DirectSolverFactory : public SolverFactory
{
public:
    /** \name Constructors and destructor */
    ///@{

    /** \brief Constructor taking only a parameter list.
     *
     *  This should have a field naming the type of solver requested
     *  and a sublist containing any parameters required by that
     *  solver.
     *
     *  Valid names are "SuperLU", "SuperLU Dist", and
     *  "Strumpack". Case doesn't matter.
     *
     *  \param list The parameter list for the solver.
     *
     *  \note Names may be given with '-' or '_' or spaces. These
     *        characters will all be removed prior to verifying the
     *        name. The name is verified in the Build methods.
     */
    DirectSolverFactory(ParameterList const& list = ParameterList());

    /** \brief Constructor with an explicit name.
     *
     *  The name here will override any given on the parameter list!
     *
     *  \param name The name of the desired solver.
     *  \param list The parameter list for the solver.
     *
     *  \note Names may be given with '-' or '_' or spaces. These
     *        characters will all be removed prior to verifying the
     *        name. The name is verified in the Build methods.
     */
    DirectSolverFactory(std::string const& name,
                        ParameterList const& list = ParameterList());

    /** \brief Destructor. */
    ~DirectSolverFactory() {}

    ///@}

private:

    /** \name SolverFactory interface */
    ///@{

    /** \brief Implementation of BuildSolver(). */
    std::unique_ptr<mfem::Solver> _do_build_solver(
        const std::shared_ptr<mfem::Operator>& op,
        SolverState& state) const override;

    /** \brief Implementation of SetDefaultParamters(). */
    void _do_set_default_parameters() override {}

    /** \brief Implementation of Initialize(). */
    void _do_initialize(const ParameterList& Params) override;

    ///@}

    /** \brief Load the list of allowable solvers into the factory. */
    void _default_solvers_initialize();

private:

    /** \brief A string describing the type of DirectSolver to build. */
    std::string Type_;

    /** \brief The identifier type for builders in the factory. */
    using id_type = std::string;

    /** \brief The builder type for the factory. */
    using builder_type = std::function<std::unique_ptr<DirectSolver>()>;

    /** \brief The type of factory that builds DirectSolvers. */
    using factory_type = Factory<DirectSolver,id_type,builder_type>;

    /** \brief A factory for building the known DirectSolvers */
    factory_type DirSolverFactory_;

};// class DirectSolverFactory
}// namespace parelag

#endif /*  PARELAG_DIRECTSOLVERFACTORY_HPP_ */
