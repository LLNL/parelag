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


#include <unordered_set>

#include "ParELAG_Config.h"

#include "linalg/factories/ParELAG_DirectSolverFactory.hpp"
#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"
#include "linalg/utilities/ParELAG_MonolithicBlockedOperatorFactory.hpp"

#include "utilities/MemoryUtils.hpp"

#ifdef ParELAG_ENABLE_SUPERLU
#include "linalg/sparse_direct/ParELAG_SuperLUSolver.hpp"
#endif
#ifdef ParELAG_ENABLE_SUPERLUDIST
#include "linalg/sparse_direct/ParELAG_SuperLUDistSolver.hpp"
#endif
#ifdef ParELAG_ENABLE_STRUMPACK
#include "linalg/sparse_direct/ParELAG_StrumpackSolver.hpp"
#endif

namespace parelag
{


namespace {

/** \brief Ensures formatting of name string.
 *
 *  Remove spaces, dashes, and underscores from type name. Converts to
 *  upper case.
 */
std::string ProcessType(std::string name)
{
    auto ret = std::remove_if(
        name.begin(),name.end(),
        [](char c){return (std::isspace(c) || c == '-' || c== '_'); });

    if (ret != name.end()) name.erase(ret,name.end());

    std::transform(name.begin(),name.end(),name.begin(), ::toupper);

    return name;
}

}// namespace <unnamed>


// Build the default list of valid direct solvers
void DirectSolverFactory::_default_solvers_initialize()
{
#ifdef ParELAG_ENABLE_SUPERLU
    DirSolverFactory_.RegisterBuilder(
        "SUPERLU",make_unique<SuperLUSolver<>>);
#endif
#ifdef ParELAG_ENABLE_SUPERLUDIST
    DirSolverFactory_.RegisterBuilder(
        "SUPERLUDIST",make_unique<SuperLUDistSolver<>>);
#endif
#ifdef ParELAG_ENABLE_STRUMPACK
    DirSolverFactory_.RegisterBuilder(
        "STRUMPACK",make_unique<StrumpackSolver<>>);
#endif
}


DirectSolverFactory::DirectSolverFactory(ParameterList const& list)
    : Type_("INVALID")
{
    Initialize(list);
}


DirectSolverFactory::DirectSolverFactory(
    std::string const& name,ParameterList const& list)
    : Type_(ProcessType(name))
{
    Initialize(list);
}


std::unique_ptr<mfem::Solver> DirectSolverFactory::_do_build_solver(
    const std::shared_ptr<mfem::Operator>& op, SolverState&) const
{
    // Get a direct solver object
    auto out = DirSolverFactory_.CreateObject(Type_);

    // Setup the solver
    {
        // Test for a BlockOperator
        auto tmp = std::dynamic_pointer_cast<MfemBlockOperator>(op);
        if (tmp)
        {
            MonolithicBlockedOperatorFactory fact;
            out->SetOperator(fact.BuildOperator(*tmp));
        }
        else
            out->SetOperator(op);
    }
    out->SetSolverParameters(GetParameters());
    out->Factor();

    return std::move(out);
}


void DirectSolverFactory::_do_initialize(const ParameterList&)
{
    _default_solvers_initialize();
    Type_ = ProcessType(GetParameters().Get<std::string>("Name","INVALID"));
}

}// namespace parelag
