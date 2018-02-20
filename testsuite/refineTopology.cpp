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
        // 2. Read the (serial) mesh from the given mesh file on all processors.
        //    We can handle triangular, quadrilateral, tetrahedral or hexahedral
        //    elements with the same code.
        std::ifstream imesh(argv[1]);
        if (!imesh)
        {
            if (myid == 0)
                std::cerr << "\nCan not open mesh file: " << argv[1] << "\n\n";
            return EXIT_FAILURE;
        }
        auto mesh = make_unique<Mesh>(imesh, 1, 1);
        imesh.close();

        // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
        //    this mesh further in parallel to increase the resolution. Once the
        //    parallel mesh is defined, the serial mesh can be deleted.
        pmesh = make_shared<ParMesh>(comm, *mesh);
    }

    auto topo = make_shared<AgglomeratedTopology>(pmesh, pmesh->Dimension());
    auto finetopo = topo->UniformRefinement();

    return EXIT_SUCCESS;
}
