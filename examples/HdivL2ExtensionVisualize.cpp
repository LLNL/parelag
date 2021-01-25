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

#include <fstream>
#include <sstream>
#include <ostream>
#include <string>
#include <vector>
#include <memory>

#include <mpi.h>

#include "elag.hpp"
#include "utilities/MPIDataTypes.hpp"

using namespace mfem;
using namespace parelag;
using namespace std;

int main(int argc, char *argv[])
{
    // Initialize MPI.
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    cout << "-- This visualizes H(div)-L2 sample coarse shape functions in 2D\n\n";

    // The file, from which to read the mesh.
    const string meshfile = "AE.mesh";

    // The number of times to refine in serial.
    int ser_ref_levels = 3;

    // The order of the finite elements on the finest level.
    const int feorder = 0;

    // The order of the polynomials to include in the coarse spaces
    // (after interpolating them onto the fine space).
    const int upscalingOrder = 2;

    // Number of levels to generate (including the finest one).
    const int nLevels = 2;

    // Which coarse facet to plot.
    const int iAF = 3;

    // Which facet basis to plot.
    const int fb = 14;

    // Which facet PV basis to plot.
    const int fPV = 12;

    // Which bubble to plot.
    const int ib = 18;

    // SVD tolerance.
    const double tolSVD = 1e-9;

    // Read the (serial) mesh from the given mesh file and uniformly refine it.
    shared_ptr<ParMesh> pmesh;
    {
        cout << "\nReading and refining serial mesh...\n";

        ifstream imesh(meshfile);
        if (!imesh)
        {
            cerr << "ERROR: Cannot open mesh file: " << meshfile << ".\n";
            return EXIT_FAILURE;
        }

        auto mesh = make_unique<Mesh>(imesh, true, true);
        imesh.close();

        for (int l = 0; l < ser_ref_levels; ++l)
            mesh->UniformRefinement();

        pmesh = make_shared<ParMesh>(comm, *mesh);
        pmesh->ReorientTetMesh();
    }

    // Refine the mesh in parallel.
    const int nDimensions = pmesh->Dimension();

    vector<int> level_nElements(nLevels);
    level_nElements[0] = pmesh->GetNE();
    for (int l = 1; l < nLevels; ++l)
        level_nElements[l] = 1;

    // Obtain the hierarchy of agglomerate topologies.
    cout << "Agglomerating topology for " << nLevels - 1
         << " coarse levels...\n";

    constexpr auto AT_elem = AgglomeratedTopology::ELEMENT;
    constexpr auto AT_facet = AgglomeratedTopology::FACET;
    vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
    for(int l = 0; l < nLevels - 1; ++l)
    {
        Array<int> partitioning(topology[l]->GetNumberLocalEntities(AT_elem));
        partitioning = 0;
        topology[l + 1] = topology[l]->CoarsenLocalPartitioning(partitioning,
                                                                false, false);
    }

//    ShowTopologyAgglomeratedElements(topology[1].get(), pmesh.get());

    // Construct the hierarchy of spaces, thus forming a hierarchy of (partial)
    // de Rham sequences.
    cout << "Building the fine-level de Rham sequence...\n";

    vector<shared_ptr<DeRhamSequence>> sequence(topology.size());

    const int jform = nDimensions - 1; // This is the H(div) form.
    MFEM_ASSERT(nDimensions == 2, "Only 2D is visualized.");
    if(nDimensions == 3)
        sequence[0] = make_shared<DeRhamSequence3D_FE>(topology[0], pmesh.get(),
                                                       feorder);
    else
        sequence[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(topology[0],
                                                            pmesh.get(),
                                                            feorder);

    // To build H(div) (form 1 in 2D), it is needed to obtain all forms and
    // spaces with larger indices.
    sequence[0]->SetjformStart(jform);

    cout << "Interpolating and setting polynomial targets...\n";

    DeRhamSequenceFE *DRSequence_FE = sequence[0]->FemSequence();
    MFEM_ASSERT(DRSequence_FE,
                "Failed to obtain the fine-level de Rham sequence.");
    DRSequence_FE->SetUpscalingTargets(nDimensions, upscalingOrder);

    cout << "Building the coarse-level de Rham sequences...\n";

    for(int l=0; l < nLevels - 1; ++l)
    {
        sequence[l]->SetSVDTol(tolSVD);
        sequence[l + 1] = sequence[l]->Coarsen();
    }

    cout << "Visualizing...\n";

    const SparseMatrix &AF_dof = sequence[0]->GetDofAgg(jform)->GetAEntityDof(AT_facet);
    DenseMatrix Ploc;
    Array<int> dof_in_AF(const_cast<int *>(AF_dof.GetRowColumns(iAF)),
                         AF_dof.RowSize(iAF));
    Full(*(sequence[0]->GetP(jform)), Ploc);
    Vector trace(Ploc.Height());
    trace = 0.0;
    for (int i=0; i < dof_in_AF.Size(); ++i)
        trace(dof_in_AF[i]) = Ploc(dof_in_AF[i], fb);
    {
        MultiVector tmp(trace.GetData(), 1, trace.Size());
        sequence[0]->show(jform, tmp);
    }
    trace = 0.0;
    for (int i=0; i < dof_in_AF.Size(); ++i)
        trace(dof_in_AF[i]) = Ploc(dof_in_AF[i], fPV);
    {
        MultiVector tmp(trace.GetData(), 1, trace.Size());
        sequence[0]->show(jform, tmp);
    }
    trace = 0.0;
    for (int i=0; i < trace.Size(); ++i)
        trace(i) = Ploc(i, fb);
    {
        MultiVector tmp(trace.GetData(), 1, trace.Size());
        sequence[0]->show(jform, tmp);
        Vector div(sequence[0]->GetD(jform)->Height());
        sequence[0]->GetD(jform)->Mult(trace, div);
        MultiVector tmp1(div.GetData(), 1, div.Size());
        sequence[0]->show(jform + 1, tmp1);
    }
    trace = 0.0;
    for (int i=0; i < trace.Size(); ++i)
        trace(i) = Ploc(i, fPV);
    {
        MultiVector tmp(trace.GetData(), 1, trace.Size());
//        sequence[0]->show(jform, tmp);
        Vector div(sequence[0]->GetD(jform)->Height());
        sequence[0]->GetD(jform)->Mult(trace, div);
        MultiVector tmp1(div.GetData(), 1, div.Size());
//        sequence[0]->show(jform + 1, tmp1);
    }

    trace = 0.0;
    for (int i=0; i < trace.Size(); ++i)
        trace(i) = Ploc(i, ib);
    {
        MultiVector tmp(trace.GetData(), 1, trace.Size());
        sequence[0]->show(jform, tmp);
        Vector div(sequence[0]->GetD(jform)->Height());
        sequence[0]->GetD(jform)->Mult(trace, div);
        MultiVector tmp1(div.GetData(), 1, div.Size());
        sequence[0]->show(jform + 1, tmp1);
    }


    cout << "\nFinished.\n";

    return EXIT_SUCCESS;
}
