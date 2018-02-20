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


#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "linalg/solver_core/ParELAG_SolverFactoryCreator.hpp"

#include "linalg/factories/ParELAG_ADSSolverFactory.hpp"
#include "linalg/factories/ParELAG_AMGeSolverFactory.hpp"
#include "linalg/factories/ParELAG_AMSSolverFactory.hpp"
#include "linalg/factories/ParELAG_Block2x2GaussSeidelSolverFactory.hpp"
#include "linalg/factories/ParELAG_Block2x2JacobiSolverFactory.hpp"
#include "linalg/factories/ParELAG_Block2x2LDUSolverFactory.hpp"
#include "linalg/factories/ParELAG_BoomerAMGSolverFactory.hpp"
#include "linalg/factories/ParELAG_BramblePasciakFactory.hpp"
#include "linalg/factories/ParELAG_DirectSolverFactory.hpp"
#include "linalg/factories/ParELAG_HiptmairSmootherFactory.hpp"
#include "linalg/factories/ParELAG_HybridizationSolverFactory.hpp"
#include "linalg/factories/ParELAG_HypreSmootherFactory.hpp"
#include "linalg/factories/ParELAG_KrylovSolverFactory.hpp"
#include "linalg/factories/ParELAG_StationarySolverFactory.hpp"

namespace parelag
{

void SolverLibrary::_default_solvers_initialize()
{
    solver_factory_factory_.RegisterBuilder(
        "ADS",SolverFactoryCreator<ADSSolverFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "AMGe",SolverFactoryCreator<AMGeSolverFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "AMS",SolverFactoryCreator<AMSSolverFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "Block GS",SolverFactoryCreator<Block2x2GaussSeidelSolverFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "Block Jacobi",SolverFactoryCreator<Block2x2JacobiSolverFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "Block LDU",SolverFactoryCreator<Block2x2LDUSolverFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "BoomerAMG",SolverFactoryCreator<BoomerAMGSolverFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "Bramble-Pasciak",SolverFactoryCreator<BramblePasciakFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "Direct",SolverFactoryCreator<DirectSolverFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "Hiptmair",SolverFactoryCreator<HiptmairSmootherFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "Hybridization", SolverFactoryCreator<HybridizationSolverFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "Hypre",SolverFactoryCreator<HypreSmootherFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "Krylov",SolverFactoryCreator<KrylovSolverFactory>{});
    solver_factory_factory_.RegisterBuilder(
        "Stationary Iteration",
        SolverFactoryCreator<StationarySolverFactory>{});
}

}// namespace parelag
