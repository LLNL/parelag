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
 * coarseningMetis.cpp
 *
 *  Created on: May 16, 2014
 *      Author: barker29
 */

#include <fstream>
#include <sstream>
#include "mfem.hpp"
#include "../src/topology/elag_topology.hpp"
#include "../src/partitioning/elag_partitioning.hpp"

int main (int argc, char *argv[])
{
   int num_procs, myid;

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

   {
      int ref_levels = 2;
         //(int)floor(log(10000./mesh->GetNE())/log(2.)/mesh->Dimension());
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(comm, *mesh);
   delete mesh;
   {
      int par_ref_levels = 0;
      for (int l = 0; l < par_ref_levels; l++)
         pmesh->UniformRefinement();
   }

   Array<AgglomeratedTopology *> topo(2);
   topo = static_cast<AgglomeratedTopology *>(NULL);
   topo[0] = new AgglomeratedTopology(pmesh, pmesh->Dimension());

	int elem_per_agg = 64;
	if (myid == 0)
		std::cout << "Partitioning with metis with " << elem_per_agg << " elements per aggregate." << std::endl;
   int nparts = pmesh->GetNE() / elem_per_agg;
	MetisGraphPartitioner partitioner;
	Array<int> partitioning(topo[0]->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT));
	partitioner.setFlags(MetisGraphPartitioner::KWAY );	// BISECTION
	partitioner.setOption(METIS_OPTION_SEED, 0);         // Fix the seed
	partitioner.setOption(METIS_OPTION_CONTIG,1);        // Ask metis to provide contiguous partitions
	partitioner.setOption(METIS_OPTION_MINCONN,1);       //
	partitioner.setUnbalanceToll(1.05);
	partitioner.doPartition(*(topo[0]->LocalElementElementTable()), nparts, partitioning);

	topo[1] = topo[0]->CoarsenLocalPartitioning(partitioning, 1, 0);

	/*
   MFEMRefinedMeshPartitioner partitioner(pmesh->Dimension());
   Array<int> partitioning(topo[0]->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT));
   partitioner.Partition(topo[0]->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT), nparts, partitioning);
   topo[1] = topo[0]->CoarsenLocalPartitioning(partitioning, 0, 0);
	*/

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

   ShowTopologyAgglomeratedElements(topo[1], pmesh);
   ShowTopologyAgglomeratedFacets(topo[1], pmesh);
   ShowTopologyBdrFacets(topo[1], pmesh);


   for(int ilevel = 0; ilevel < topo.Size(); ++ilevel)
	   delete topo[ilevel];
   delete pmesh;

   MPI_Finalize();

   return 0;

}
