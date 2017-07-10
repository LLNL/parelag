/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
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
#include "mfem.hpp"
#include "../src/topology/elag_topology.hpp"
#include "../src/partitioning/elag_partitioning.hpp"

int main (int argc, char *argv[])
{

   int num_procs, myid;

   // 1. Initialize MPI
   MPI_Init(&argc, &argv);
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   Mesh *mesh;

   if (argc == 1)
   {
      if (myid == 0)
         std::cout << "\nUsage: mpirun -np <np> topology <mesh_file>\n" << std::endl;
      MPI_Finalize();
      return 1;
   }

   // 2. Read the (serial) mesh from the given mesh file on all processors.
   //    We can handle triangular, quadrilateral, tetrahedral or hexahedral
   //    elements with the same code.
   std::ifstream imesh(argv[1]);
   if (!imesh)
   {
      if (myid == 0)
         std::cerr << "\nCan not open mesh file: " << argv[1] << '\n' << std::endl;
      MPI_Finalize();
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(comm, *mesh);
   delete mesh;

   AgglomeratedTopology * topo = new AgglomeratedTopology(pmesh, pmesh->Dimension());
   AgglomeratedTopology * finetopo = topo->UniformRefinement();

   delete finetopo;
   delete topo;
   delete pmesh;

   MPI_Finalize();

   return 0;

}
