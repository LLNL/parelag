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

//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p ../data/square-disc.mesh
//               mpirun -np 4 ex1p ../data/star.mesh
//               mpirun -np 4 ex1p ../data/escher.mesh
//               mpirun -np 4 ex1p ../data/fichera.mesh
//               mpirun -np 4 ex1p ../data/square-disc-p2.vtk
//               mpirun -np 4 ex1p ../data/square-disc-p3.mesh
//               mpirun -np 4 ex1p ../data/square-disc-nurbs.mesh
//               mpirun -np 4 ex1p ../data/disc-nurbs.mesh
//               mpirun -np 4 ex1p ../data/pipe-nurbs.mesh
//               mpirun -np 4 ex1p ../data/ball-nurbs.mesh
//


#include <fstream>
#include <sstream>

#include "../src/topology/elag_topology.hpp"
#include "../src/partitioning/elag_partitioning.hpp"
#include "../src/amge/elag_amge.hpp"

// argv[1] - mesh file
// argv[2] - orderfe
// argv[3] - serial refinement
// argv[4] - coarsering factor

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
	int par_ref_levels = 2;
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

//-----------------------------------------------------//
	int nDimensions = pmesh->Dimension();
	int feorder = 0;
	Array<DeRhamSequence *> sequence(topo.Size() );
	int upscalingOrder = feorder;
	if(nDimensions == 2)
		sequence[0] = new DeRhamSequence2D_Hdiv_FE(topo[0], pmesh, feorder);
	else
		sequence[0] = new DeRhamSequence3D_FE(topo[0], pmesh, feorder);

	Array< MultiVector *> targets( sequence[0]->GetNumberOfForms() );

	if(nDimensions == 2)
	{
		Array<Coefficient *> H1coeff, L2coeff;
		Array<VectorCoefficient *> Hdivcoeff;

		fillCoefficientArray(nDimensions, upscalingOrder+1, H1coeff);
		fillRTVectorCoefficientArray(nDimensions, upscalingOrder,Hdivcoeff);
		fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);

		int jform(0);

		targets[jform] = dynamic_cast<DeRhamSequenceFE *>(sequence[0])->InterpolateScalarTargets(jform, H1coeff);
		++jform;

		targets[jform] = dynamic_cast<DeRhamSequenceFE *>(sequence[0])->InterpolateVectorTargets(jform, Hdivcoeff);
		++jform;

		targets[jform] = dynamic_cast<DeRhamSequenceFE *>(sequence[0])->InterpolateScalarTargets(jform, L2coeff);
		++jform;

		freeCoeffArray(H1coeff);
		freeCoeffArray(Hdivcoeff);
		freeCoeffArray(L2coeff);
	}
	else
	{
		Array<Coefficient *> H1coeff, L2coeff;
		Array<VectorCoefficient *> Hcurlcoeff, Hdivcoeff;

		fillCoefficientArray(nDimensions, upscalingOrder+1, H1coeff);
		fillVectorCoefficientArray(nDimensions, upscalingOrder, Hcurlcoeff);
		fillVectorCoefficientArray(nDimensions, upscalingOrder, Hdivcoeff);
		fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);


		int jform(0);

		targets[jform] = dynamic_cast<DeRhamSequenceFE *>(sequence[0])->InterpolateScalarTargets(jform, H1coeff);
		++jform;

		targets[jform] = dynamic_cast<DeRhamSequenceFE *>(sequence[0])->InterpolateVectorTargets(jform, Hcurlcoeff);
		++jform;

		targets[jform] = dynamic_cast<DeRhamSequenceFE *>(sequence[0])->InterpolateVectorTargets(jform, Hdivcoeff);
		++jform;

		targets[jform] = dynamic_cast<DeRhamSequenceFE *>(sequence[0])->InterpolateScalarTargets(jform, L2coeff);
		++jform;

		freeCoeffArray(H1coeff);
		freeCoeffArray(Hcurlcoeff);
		freeCoeffArray(Hdivcoeff);
		freeCoeffArray(L2coeff);
	}

	int jstart = 0;

	sequence[0]->SetjformStart( jstart );
	sequence[0]->SetTargets( targets );

	for(int i(0); i < nLevels-1; ++i)
	{
		sequence[i+1] = sequence[i]->Coarsen();
		sequence[i]->CheckInvariants();
	}
	sequence.Last()->CheckD();
	sequence.Last()->CheckTrueD();

	for(int jf(jstart); jf < nDimensions+1; ++jf)
	{
		DeRhamSequence::DeRhamSequence_os << "Interpolation Error Form " << jf << "\n";
		sequence[0]->ComputeSpaceInterpolationError(jf, *(targets[jf] ) );
	}

	SerializedOutput(comm, std::cout, DeRhamSequence::DeRhamSequence_os.str());

/*
for(int i(0); i < sequence.Size()-1; ++i)
	sequence[i]->ShowProjector(nDimensions);

for(int i(0); i < sequence.Size()-1; ++i)
{
	sequence[i]->ShowProjector(nDimensions-1);
	sequence[i]->ShowDerProjector(nDimensions-1);
}
*/
sequence[sequence.Size()-2]->ShowProjector(nDimensions-1);
sequence[sequence.Size()-2]->ShowDerProjector(nDimensions-1);

//sequence[sequence.Size()-2]->ShowProjector(nDimensions-1);

	for(int i(0); i < sequence.Size(); ++i)
		delete sequence[i];

	for(int i(0); i < targets.Size(); ++i)
		delete targets[i];

//----------------------------------------------------//
   for(int i(0); i < topo.Size(); ++i)
	   delete topo[i];

   MPI_Finalize();

   delete pmesh;

   return 0;
}
