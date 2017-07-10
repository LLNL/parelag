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
 * repartition.cpp
 *
 * trying to knock some processors out
 * and maintain all the data
 *
 * Created on: May 16, 2014
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

	if (num_procs % 2 == 1)
	{
		if (myid == 0)
			std::cout << "Must run on an even number of processors.\n" << std::endl;
		MPI_Finalize();
		return 1;
	}
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
      int ref_levels = 0;
         //(int)floor(log(10000./mesh->GetNE())/log(2.)/mesh->Dimension());
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(comm, *mesh);
   delete mesh;
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
         pmesh->UniformRefinement();
   }

   FiniteElementCollection *fec;
   if (pmesh->GetNodes())
      fec = pmesh->GetNodes()->OwnFEC();
   else
      fec = new LinearFECollection;
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   int size = fespace->GlobalTrueVSize();
   if (myid == 0)
      std::cout << "Number of unknowns: " << size << std::endl;

   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   ParGridFunction x(fespace);
   x = 0.0;

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->Assemble();
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      Array<int> ess_dofs;
      fespace->GetEssentialVDofs(ess_bdr, ess_dofs);
      a->EliminateEssentialBCFromDofs(ess_dofs, x, *b);
   }
   a->Finalize();

   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelAverage();

   delete a;
   delete b;

	A->Print("mat");

	// int *rowpart = A->RowPart();
	Array<int> rowpart(num_procs+1);
	for(int i=0; i<num_procs+1; ++i)
	{
		rowpart[i] = A->RowPart()[i];
	}
	Array<int> desiredrowpart(num_procs+1);
	desiredrowpart[0] = 0;
   std::stringstream msg;
   for(int i=0; i<num_procs+1; ++i)
   {
		msg << "[" << myid << "] rowpart[" << i << "] = " << rowpart[i] << std::endl;
		if (i % 2 == 1)
		{			
			desiredrowpart[i] = rowpart[i+1];
			desiredrowpart[i+1] = rowpart[i+1];
		} 
		msg << "[" << myid << "] desiredrowpart[" << i << "] = " << desiredrowpart[i] << std::endl;
   }
   SerializedOutput(comm, std::cout, msg.str() );

	ProcessorCoarsening pcoarse(comm);
	pcoarse.SetUpSameOrder(rowpart, desiredrowpart);

	HypreParMatrix *C = RepartitionMatrix(pcoarse, *A);
	//HypreParMatrix *C = pcoarse.RepartitionMatrix(A);
	C->Print("rep");

   delete X;
   delete B;
   delete A;

	delete C;

   delete fespace;
   if (!pmesh->GetNodes())
      delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
