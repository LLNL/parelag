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

#include "elag.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

// argv[1] - mesh file
// argv[2] - orderfe
// argv[3] - serial refinement
// argv[4] - coarsening factor

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
        Mesh mesh(imesh, 1, 1);
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
                mesh.UniformRefinement();
        }

        // 4. Define a parallel mesh by a partitioning of the serial
        //    mesh. Refine this mesh further in parallel to increase
        //    the resolution. Once the parallel mesh is defined, the
        //    serial mesh can be deleted.
        pmesh = make_shared<ParMesh>(comm, mesh);
    }

    const int nDimensions = pmesh->Dimension();
    const int par_ref_levels = 2;
    Array<int> level_nElements(par_ref_levels+1);
    for (int l = 0; l < par_ref_levels; l++)
    {
        level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    level_nElements[0] = pmesh->GetNE();
    const int nLevels = par_ref_levels + 1;

    std::vector<shared_ptr<AgglomeratedTopology>> topo(nLevels);

    topo[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);

    MFEMRefinedMeshPartitioner partitioner(nDimensions);
    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        Array<int> partitioning(
            topo[ilevel]->GetNumberLocalEntities(at_elem));
        partitioner.Partition(topo[ilevel]->GetNumberLocalEntities(at_elem),
                              level_nElements[ilevel+1], partitioning);
        topo[ilevel+1] =
            topo[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);
    }

    const int feorder = 0;
    std::vector<shared_ptr<DeRhamSequence>> sequence(topo.size());
    const int upscalingOrder = feorder;
    if(nDimensions == 2)
        sequence[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(
            topo[0], pmesh.get(), feorder);
    else
        sequence[0] =
            make_shared<DeRhamSequence3D_FE>(topo[0], pmesh.get(), feorder);

    DeRhamSequenceFE* DRSequence_FE = sequence[0]->FemSequence();
    std::vector<unique_ptr<MultiVector>>
        targets(sequence[0]->GetNumberOfForms());

    if(nDimensions == 2)
    {
        Array<Coefficient *> H1coeff, L2coeff;
        Array<VectorCoefficient *> Hdivcoeff;

        fillCoefficientArray(nDimensions, upscalingOrder+1, H1coeff);
        fillRTVectorCoefficientArray(nDimensions, upscalingOrder,Hdivcoeff);
        fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);

        targets[0] = DRSequence_FE->InterpolateScalarTargets(0, H1coeff);
        targets[1] = DRSequence_FE->InterpolateVectorTargets(1, Hdivcoeff);
        targets[2] = DRSequence_FE->InterpolateScalarTargets(2, L2coeff);

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

        targets[0] = DRSequence_FE->InterpolateScalarTargets(0, H1coeff);
        targets[1] = DRSequence_FE->InterpolateVectorTargets(1, Hcurlcoeff);
        targets[2] = DRSequence_FE->InterpolateVectorTargets(2, Hdivcoeff);
        targets[3] = DRSequence_FE->InterpolateScalarTargets(3, L2coeff);

        freeCoeffArray(H1coeff);
        freeCoeffArray(Hcurlcoeff);
        freeCoeffArray(Hdivcoeff);
        freeCoeffArray(L2coeff);
    }

    Array<MultiVector*> targets_in(targets.size());
    for (int ii = 0; ii < targets_in.Size(); ++ii)
        targets_in[ii] = targets[ii].get();

    sequence[0]->SetjformStart(0);
    sequence[0]->SetTargets(targets_in);

    for(int i(0); i < nLevels-1; ++i)
    {
        sequence[i+1] = sequence[i]->Coarsen();
        sequence[i]->CheckInvariants();
    }
    // FIXME: ARE THESE TESTED ANYWAY??
    //sequence.back()->CheckD();
    //sequence.back()->CheckTrueD();

    for(int jf = 0; jf < nDimensions+1; ++jf)
    {
        DeRhamSequence::DeRhamSequence_os << "Interpolation Error Form "
                                          << jf << "\n";
        sequence[0]->ComputeSpaceInterpolationError(jf, *(targets[jf]));
    }

    SerializedOutput(comm,std::cout,DeRhamSequence::DeRhamSequence_os.str());

    /*
      for(int i(0); i < sequence.Size()-1; ++i)
      sequence[i]->ShowProjector(nDimensions);

      for(int i(0); i < sequence.Size()-1; ++i)
      {
      sequence[i]->ShowProjector(nDimensions-1);
      sequence[i]->ShowDerProjector(nDimensions-1);
      }
    */
    sequence[sequence.size()-2]->ShowProjector(nDimensions-1);
    sequence[sequence.size()-2]->ShowDerProjector(nDimensions-1);

    //sequence[sequence.Size()-2]->ShowProjector(nDimensions-1);

    return EXIT_SUCCESS;
}
