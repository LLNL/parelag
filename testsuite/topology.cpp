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

/*
 * topology.cpp
 *
 *  Created on: Apr 23, 2014
 *      Author: villa13
 */

#include <fstream>
#include <sstream>
#include <mfem.hpp>
#include "elag.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

int main (int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    if (argc == 1)
    {
        if (myid == 0)
            std::cerr << "\nUsage: mpirun -np <np> topology <mesh_file>\n\n";
        return EXIT_FAILURE;
    }

    shared_ptr<ParMesh> pmesh;
    {
        // 2. Read the (serial) mesh from the given mesh file on all
        //    processors.  We can handle triangular, quadrilateral,
        //    tetrahedral or hexahedral elements with the same code.
        std::ifstream imesh(argv[1]);
        if (!imesh)
        {
            if (myid == 0)
                std::cerr << "\nCan not open mesh file: " << argv[1] << "\n\n";
            return EXIT_FAILURE;
        }
        auto mesh = make_unique<Mesh>(imesh, 1, 1);
        imesh.close();

        // 3. Refine the serial mesh on all processors to increase the
        //    resolution. In this example we do 'ref_levels' of
        //    uniform refinement. We choose 'ref_levels' to be the
        //    largest number that gives a final mesh with no more than
        //    10,000 elements.
        {
            const int ref_levels = 0;
            //(int)floor(log(10000./mesh->GetNE())/log(2.)/mesh->Dimension());
            for (int l = 0; l < ref_levels; l++)
                mesh->UniformRefinement();
        }

        // 4. Define a parallel mesh by a partitioning of the serial
        //    mesh. Refine this mesh further in parallel to increase
        //    the resolution. Once the parallel mesh is defined, the
        //    serial mesh can be deleted.
        pmesh = make_shared<ParMesh>(comm, *mesh);

        const int par_ref_levels = 0;
        for (int l = 0; l < par_ref_levels; l++)
            pmesh->UniformRefinement();
    }

    auto topo = make_shared<AgglomeratedTopology>(pmesh, pmesh->Dimension());

    for (int i = 0; i < pmesh->Dimension(); ++i)
    {
        topo->GetB(i);
        topo->Weight(i);
        auto tmp = topo->TrueWeight(i);
    }

    std::stringstream msg;
    topo->ShowMe(msg);
    SerializedOutput(comm, std::cout, msg.str());

    for (int i = 0; i < pmesh->Dimension()-1; ++i)
    {
        auto BB = ToUnique(Mult(topo->GetB(i), topo->GetB(i+1)));
        elag_assert(BB->MaxNorm() < 1e-12);
        BB.reset();

        auto& Bi = topo->TrueB(i);
        auto& Bii = topo->TrueB(i+1);

        elag_assert(hypre_ParCSRMatrixMaxNorm(Bi) > 1 - 1e-12);
        elag_assert(hypre_ParCSRMatrixMaxNorm(Bii) > 1 - 1e-12);

        auto pBB = ToUnique(ParMult(&Bi, &Bii));
        elag_assert(hypre_ParCSRMatrixMaxNorm(*pBB) < 1e-12);
        elag_assert(hypre_ParCSRMatrixFrobeniusNorm(*pBB) < 1e-12);
        elag_assert(hypre_ParCSRMatrixNorml1(*pBB) < 1e-12);
        elag_assert(hypre_ParCSRMatrixNormlinf(*pBB) < 1e-12);
    }
    ShowTopologyBdrFacets(topo.get(), pmesh.get());

    return EXIT_SUCCESS;
}
