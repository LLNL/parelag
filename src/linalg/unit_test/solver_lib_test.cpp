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

#include <iostream>
#include <memory>

#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"

#include "utilities/mpiUtils.hpp"

#include "linalg/factories/ParELAG_HiptmairSmootherFactory.hpp"
#include "linalg/factories/ParELAG_HypreSmootherFactory.hpp"

using namespace parelag;

int main(int argc, char** argv)
{
    mpi_session mpi_(argc,argv);

    // Let's just initialize a library and create a few SolverFactories
    auto lib = SolverLibrary::CreateLibrary();

    ParameterList pl("master");
    ParameterList& subpl = pl.Sublist("Solver1");
    ParameterList& subpl2 = pl.Sublist("Solver2");
    //pl.Sublist("Tom's Super Wonderful Solver!");

    subpl.Set("Type","Hypre");
    subpl.Sublist("Solver Parameters");

    subpl2.Set("Type","Hiptmair");
    ParameterList& subsubpl = subpl2.Sublist("Solver Parameters");
    subsubpl.Set("Primary Smoother","Solver1");
    subsubpl.Set("Auxiliary Smoother","Solver1");

    lib->Initialize(pl);

    // Get the names of all solvers in the library
    auto solver_names = lib->GetSolverNames();

    std::cout << "Solvers in lib = { ";
    for (const auto& name : solver_names)
        std::cout << "\"" << name << "\" ";
    std::cout << "}\n";

    std::cout << "Solver Factories in lib = { ";
    for (const auto& name : lib->GetSolverFactoryNames())
        std::cout << "\"" << name << "\" ";
    std::cout << "}\n";


    // Try creating a HiptmairSmootherFactory
    std::shared_ptr<SolverFactory> fact =
        std::make_shared<HiptmairSmootherFactory>();
    fact->SetSolverLibrary(lib);

    ParameterList testPL;
    testPL.Set("Primary Smoother", "Solver1");
    testPL.Set("Auxiliary Smoother", "Solver1");

    fact->Initialize(testPL);

    auto AsHipt =
        std::dynamic_pointer_cast<HiptmairSmootherFactory>(fact);

    PARELAG_ASSERT(AsHipt);

    auto PrimFact = AsHipt->GetPrimarySmootherFactory();

    PARELAG_ASSERT(PrimFact);

    auto PrimAsHypre =
        std::dynamic_pointer_cast<HypreSmootherFactory>(PrimFact);

    PARELAG_ASSERT(PrimAsHypre);

    auto AuxFact = AsHipt->GetAuxiliarySmootherFactory();

    PARELAG_ASSERT(AuxFact);

    auto AuxAsHypre =
        std::dynamic_pointer_cast<HypreSmootherFactory>(AuxFact);

    PARELAG_ASSERT(AuxAsHypre);

    auto fact2 = lib->GetSolverFactory("Solver2");

    PARELAG_ASSERT(fact2);

    std::cout << "Unit test passed." << std::endl;
}
