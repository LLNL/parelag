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


#ifndef PARELAG_SOLVERFACTORYCREATOR_HPP_
#define PARELAG_SOLVERFACTORYCREATOR_HPP_

#include "ParELAG_SolverFactory.hpp"
#include "utilities/MemoryUtils.hpp"
#include "utilities/ParELAG_Meta.hpp"

namespace parelag
{

/** \class SolverFactoryCreator
 *  \brief Create an instance of a particular SolverFactory.
 *
 *  Builds a SolverFactory from a parameter list.
 *
 *  \tparam FactoryT  The type of SolverFactory to be created.
 */
template <typename FactoryT>
class SolverFactoryCreator
{
    static_assert(IsBaseOf<SolverFactory,FactoryT>(),
                  "Type not derived from SolverFactory");

public:

    /** \brief Construct a new \c FactoryT.
     *
     *  \param params The parameters for the solver factory being
     *                created.
     *  \param lib The solver library to be used if \c FactoryT
     *             generates nested solvers. May be \c nullptr if not
     *             a nested solver factory.
     *
     *  \return A newly-constructed instance of a SolverFactory subclass.
     */
    std::unique_ptr<SolverFactory> operator()(
        ParameterList const& params, std::shared_ptr<const SolverLibrary> lib) const
    {
        auto ret = make_unique<FactoryT>();
        ret->SetSolverLibrary(std::move(lib));
        ret->Initialize(params);

        return std::move(ret);
    }

};// class SolverFactoryCreator
}// namespace parelag
#endif
