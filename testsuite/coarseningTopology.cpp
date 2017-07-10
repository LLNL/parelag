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
 * coarseningTopology.cpp
 *
 *  Created on: May 16, 2014
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

   // 3. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels = 0;
         //(int)floor(log(10000./mesh->GetNE())/log(2.)/mesh->Dimension());
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(comm, *mesh);
   delete mesh;
   int par_ref_levels = 3;
   Array<int> level_nElements(par_ref_levels+1);

   for (int l = 0; l < par_ref_levels; l++)
   {
       level_nElements[par_ref_levels-l] = pmesh->GetNE();
       pmesh->UniformRefinement();
   }
   level_nElements[0] = pmesh->GetNE();
   int nLevels = par_ref_levels + 1;

   Array<AgglomeratedTopology *> topo(nLevels);
   topo = static_cast<AgglomeratedTopology *>(NULL);
   topo[0] = new AgglomeratedTopology(pmesh, pmesh->Dimension());

   MFEMRefinedMeshPartitioner partitioner(pmesh->Dimension());
   for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
   {
	   Array<int> partitioning(topo[ilevel]->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT));
	   partitioner.Partition(topo[ilevel]->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT), level_nElements[ilevel+1], partitioning);
	   topo[ilevel+1] = topo[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);
   }

   for(int ilevel = 0; ilevel < topo.Size(); ++ilevel)
   {
	   for(int i = 0; i < pmesh->Dimension(); ++i)
	   {
		   topo[ilevel]->B(i);
		   topo[ilevel]->Weight(i);
		   AgglomeratedTopology::TopologyArray * tmp = topo[ilevel]->TrueWeight(i);
		   delete tmp;
	   }
   }

   std::stringstream msg;
   for(int ilevel = 0; ilevel < topo.Size(); ++ilevel)
   {
	   msg << "Level " << ilevel << "\n";
	   topo[ilevel]->ShowMe(msg);
   }

   SerializedOutput( comm, std::cout, msg.str() );

   for(int ilevel = 0; ilevel < topo.Size(); ++ilevel)
   {
	   for(int i = 0; i < pmesh->Dimension()-1; ++i)
	   {
		   SerialCSRMatrix * BB = Mult(topo[ilevel]->B(i), topo[ilevel]->B(i+1));
		   elag_assert(BB->MaxNorm() < 1e-12);
		   delete BB;

		   AgglomeratedTopology::TopologyParTable & Bi  = topo[ilevel]->TrueB(i);
		   AgglomeratedTopology::TopologyParTable & Bii = topo[ilevel]->TrueB(i+1);

		   elag_assert( hypre_ParCSRMatrixMaxNorm(Bi) > 1 - 1e-12 );
		   elag_assert( hypre_ParCSRMatrixMaxNorm(Bii) > 1 - 1e-12 );

		   ParallelCSRMatrix * pBB = ParMult(&Bi, &Bii);
		   elag_assert( hypre_ParCSRMatrixMaxNorm(*pBB) < 1e-12 );
		   elag_assert( hypre_ParCSRMatrixFrobeniusNorm(*pBB) < 1e-12 );
		   elag_assert( hypre_ParCSRMatrixNorml1(*pBB) < 1e-12 );
		   elag_assert( hypre_ParCSRMatrixNormlinf(*pBB) < 1e-12 );
		   delete pBB;
	   }
   }

   int nDim = pmesh->Dimension();
   Array<int> exactWeights(nDim);
   exactWeights = 1;

   //agglfactor is [8,4,2] in 3D and [4,2] in 2D.
   Array<int> agglfactor( nDim );
   for(int i = 0; i < nDim; ++i)
	   agglfactor[i] = 1<<(nDim-i);

   //Check that the entity weights are the correct ones in the case of mesh uniformely refined.
   for(int ilevel = 0; ilevel < topo.Size(); ++ilevel)
   {
	   for(int icodim = 0; icodim < nDim; ++icodim)
	   {
		   elag_assert(topo[ilevel]->Weight(icodim).Min() == exactWeights[icodim]);
		   elag_assert(topo[ilevel]->Weight(icodim).Max() == exactWeights[icodim]);
		   exactWeights[icodim] *= agglfactor[icodim];
	   }
   }


   ShowTopologyAgglomeratedElements(topo.Last(), pmesh);
   ShowTopologyAgglomeratedFacets(topo.Last(), pmesh);
   ShowTopologyBdrFacets(topo.Last(), pmesh);

   for(int ilevel = 0; ilevel < topo.Size(); ++ilevel)
	   delete topo[ilevel];
   delete pmesh;

   MPI_Finalize();

   return 0;

}
