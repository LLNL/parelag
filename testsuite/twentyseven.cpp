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
  Begin testing more carefully our topology checks and fixes.

  Andrew T. Barker
  atb@llnl.gov
  12 October 2016
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

constexpr int TESTMESH_NUM_ELEMENTS = 27;
constexpr double ALMOST_ZERO = 1.e-12;

void MakeSimplePartitioning(Array<int>& partitioning)
{
    PARELAG_ASSERT(partitioning.Size() == TESTMESH_NUM_ELEMENTS);
    partitioning = 1;
    for (int i=0; i<9; ++i)
        partitioning[i] = 0;
}

void MakeDisconnectedPartitioning(Array<int>& partitioning)
{
    PARELAG_ASSERT(partitioning.Size() == TESTMESH_NUM_ELEMENTS);
    partitioning = 1;
    partitioning[0] = 0;
    partitioning[26] = 0;
}

void MakeDonutPartitioning(Array<int>& partitioning)
{
    PARELAG_ASSERT(partitioning.Size() == TESTMESH_NUM_ELEMENTS);
    partitioning = 1;
    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j)
            partitioning[9*i + 3*j + 1] = 0;
    partitioning[13] = 1;
}

void MakeVoidPartitioning(Array<int>& partitioning)
{
    PARELAG_ASSERT(partitioning.Size() == TESTMESH_NUM_ELEMENTS);
    partitioning = 1;
    partitioning[13] = 0;
}

void MakeDisconnectedFacePartitioning(Array<int>& partitioning)
{
    PARELAG_ASSERT(partitioning.Size() == TESTMESH_NUM_ELEMENTS);
    partitioning = 0;
    for (int i=0; i<9; ++i)
        partitioning[i] = 1;
    for (int i=0; i<3; ++i)
        partitioning[9 + 3 + i] = 2;
}

void MakeFaceHolePartitioning(Array<int>& partitioning)
{
    PARELAG_ASSERT(partitioning.Size() == TESTMESH_NUM_ELEMENTS);
    partitioning = 2;
    for (int i=0; i<9; ++i)
        partitioning[i] = 0;
    partitioning[13] = 1;
}

void MakeDisconnectedEdgePartitioning(Array<int>& partitioning)
{
    PARELAG_ASSERT(partitioning.Size() == 36);
    partitioning = 4;
    for (int i=0; i<4; ++i)
    {
        partitioning[9+i] = 0;
        partitioning[(2*9)+i] = 1;
    }
    for (int i=5; i<9; ++i)
    {
        partitioning[9+i] = 2;
        partitioning[(2*9)+i] = 3;
    }
    for (int i=0; i<9; ++i)
    {
        partitioning[i] = 0;
        partitioning[(3*9)+i] = 3;
    }
}

void MakeTetrahedralPartitioning(Array<int>& partitioning)
{
    PARELAG_ASSERT(partitioning.Size() == 48);

    partitioning = 0;
    partitioning[8] = 1;
}

void MakeSharedEdgePartitioning(Array<int>& partitioning)
{
    PARELAG_ASSERT(partitioning.Size() == TESTMESH_NUM_ELEMENTS);
    partitioning = 0;
/*
    partitioning[0] = 1;
    partitioning[1] = 1;
    partitioning[3] = 1;
    partitioning[4] = 1;
    partitioning[9] = 1;
    partitioning[13] = 1;
*/
    partitioning[0] = 1;
    partitioning[4] = 1;
    partitioning[5] = 1;
    partitioning[9] = 1;
    partitioning[14] = 1;
    partitioning[18] = 1;
    partitioning[21] = 1;
    partitioning[22] = 1;
    partitioning[23] = 1;
}

/*
   ---   24
   xxx   21
   x--   18

   ---   15
   --x   12
   x--    9

   ---    6
   -xx    3
   ---    0
 */
void MakeSharedVertexPartitioning(Array<int>& partitioning)
{
    PARELAG_ASSERT(partitioning.Size() == TESTMESH_NUM_ELEMENTS);
    partitioning = 0;

/*
    partitioning[4] = 1;  // 4,9 is the vertex connection
    partitioning[5] = 1;
    partitioning[9] = 1;
    partitioning[14] = 1;
    partitioning[18] = 1;
    partitioning[21] = 1;
    partitioning[22] = 1;
    partitioning[23] = 1;
*/

    partitioning[4] = 1;  // 4,9 is the vertex connection
    partitioning[5] = 1;
    partitioning[9] = 2;
    partitioning[14] = 1;
    partitioning[18] = 2;
    partitioning[21] = 2;
    partitioning[22] = 2;
    partitioning[23] = 1;

    // baseline fails in all cases
    // nothing works in --no-check so far

    // partitioning[26] = 3; // fails (like baseline)
    // partitioning[2] = 3; // fails
    // partitioning[7] = 3; // fails
    // partitioning[8] = 3; // fails

    // partitioning[0] = 3; // works with check (?)
    // partitioning[1] = 3; // works with check
    // partitioning[3] = 3; // works with check
    // partitioning[6] = 3; // works with check (?)
    // partitioning[10] = 3; // works with check
    // partitioning[12] = 3; // works with check
    // partitioning[13] = 3; // works with check

    // partitioning[11] = 3; // runs with warnings (?!?!)
}

void MakeSV2Partitioning(Array<int>& partitioning)
{
    PARELAG_ASSERT(partitioning.Size() == 8);
    partitioning = 0;

    partitioning[0] = 1;
    partitioning[7] = 1;
}

int main (int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    OptionsParser args(argc, argv);
    const char* partition = "simple";
    args.AddOption(&partition, "-p", "--partition",
                   "Partition: simple, disconnected, discface, donut, void, "
                   "facehole, discedge, tet, sharededge, sharedvertex, sv2.");
    bool check = true;
    args.AddOption(&check, "-c", "--check", "-nc", "--no-check",
                   "Whether to attempt to correct bad topology.");
    bool do_visualize = true;
    args.AddOption(&do_visualize,"-v", "--do-visualize", "-nv", "--no-visualize",
                   "Whether to plot things in glvis.");
    args.Parse();

    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(std::cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(std::cout);
    }
    std::string argument(partition);

    shared_ptr<ParMesh> pmesh;
    if (argument == "discedge")
    {
        // create 39-element serial mesh
#if (MFEM_VERSION_MAJOR < 4)
        Mesh mesh(3, 3, 4, Element::HEXAHEDRON, true);
#else
        Mesh mesh(3, 3, 4, Element::HEXAHEDRON, true, 1.0, 1.0, 1.0, false);
#endif
        pmesh = make_shared<ParMesh>(comm, mesh);
    }
    else if (argument == "tet")
    {
        // create 48-element serial tetrahedral mesh
#if (MFEM_VERSION_MAJOR < 4)
        Mesh mesh(2, 2, 2, Element::TETRAHEDRON, true);
#else
        Mesh mesh(2, 2, 2, Element::TETRAHEDRON, true, 1.0, 1.0, 1.0, false);
#endif
        pmesh = make_shared<ParMesh>(comm, mesh);
    }
    else if (argument == "sv2")
    {
#if (MFEM_VERSION_MAJOR < 4)
        Mesh mesh(2, 2, 2, Element::HEXAHEDRON, true);
#else
        Mesh mesh(2, 2, 2, Element::HEXAHEDRON, true, 1.0, 1.0, 1.0, false);
#endif
        pmesh = make_shared<ParMesh>(comm, mesh);
    }
    else
    {
        // create 27-element serial mesh
#if (MFEM_VERSION_MAJOR < 4)
        Mesh mesh(3, 3, 3, Element::HEXAHEDRON, true);
#else
        Mesh mesh(3, 3, 3, Element::HEXAHEDRON, true, 1.0, 1.0, 1.0, false);
#endif
        pmesh = make_shared<ParMesh>(comm, mesh);
    }
    std::vector<shared_ptr<AgglomeratedTopology> > topology(2);
    topology[0] = make_shared<AgglomeratedTopology>(pmesh, pmesh->Dimension());

    Array<int> partitioning(topology[0]->GetNumberLocalEntities(at_elem));
    if (argument == "disconnected")
        MakeDisconnectedPartitioning(partitioning);
    else if (argument == "donut")
        MakeDonutPartitioning(partitioning);
    else if (argument == "void")
        MakeVoidPartitioning(partitioning);
    else if (argument == "discface")
        MakeDisconnectedFacePartitioning(partitioning);
    else if (argument == "facehole")
        MakeFaceHolePartitioning(partitioning);
    else if (argument == "discedge")
        MakeDisconnectedEdgePartitioning(partitioning);
    else if (argument == "tet")
        MakeTetrahedralPartitioning(partitioning);
    else if (argument == "sharededge")
        MakeSharedEdgePartitioning(partitioning);
    else if (argument == "sharedvertex")
        MakeSharedVertexPartitioning(partitioning);
    else if (argument == "sv2")
        MakeSV2Partitioning(partitioning);
    else
        MakeSimplePartitioning(partitioning);

    topology[1] = topology[0]->CoarsenLocalPartitioning(
        partitioning, check, false);

    for (int i=0; i<2; ++i)
    {
        std::stringstream msg;
        if (i==1) // for parsing in automated scripts...
        {
            topology[i]->ShowMe(msg);
            SerializedOutput(comm, std::cout, msg.str());
        }
        if (do_visualize)
        {
            // ShowAgglomeratedTopology3D(topology[i].get(), pmesh.get());
            ShowTopologyAgglomeratedElements(topology[i].get(), pmesh.get(), nullptr);
            // ShowTopologyBdrFacets(topology[0].get(), pmesh.get());
        }
    }

    // do some basic topology checks
    for (int i=0; i<2; ++i)
    {
        for (int j=0; j<pmesh->Dimension()-1; ++j)
        {
            auto BB = ToUnique(Mult(topology[i]->GetB(j), topology[i]->GetB(j+1)));
            PARELAG_ASSERT(BB->MaxNorm() < ALMOST_ZERO);
            BB.reset();

            auto& Bi = topology[i]->TrueB(j);
            auto& Bii = topology[i]->TrueB(j+1);

            PARELAG_ASSERT(hypre_ParCSRMatrixMaxNorm(Bi) > 1 - ALMOST_ZERO);
            PARELAG_ASSERT(hypre_ParCSRMatrixMaxNorm(Bii) > 1 - ALMOST_ZERO);

            auto pBB = ToUnique(ParMult(&Bi, &Bii));
            PARELAG_ASSERT(hypre_ParCSRMatrixMaxNorm(*pBB) < ALMOST_ZERO);
            PARELAG_ASSERT(hypre_ParCSRMatrixFrobeniusNorm(*pBB) < ALMOST_ZERO);
            PARELAG_ASSERT(hypre_ParCSRMatrixNorml1(*pBB) < ALMOST_ZERO);
            PARELAG_ASSERT(hypre_ParCSRMatrixNormlinf(*pBB) < ALMOST_ZERO);
        }
    }

    // coarsen the sequence itself
    const bool build_sequence = true;
    if (build_sequence)
    {
        const int feorder = 0;
        // DeRhamSequence3D_FE fine_sequence(topology[0], pmesh.get(), feorder);
        auto fine_sequence = make_shared<DeRhamSequence3D_FE>(topology[0],
                                                              pmesh.get(),
                                                              feorder);
        constexpr int jFormStart = 0;
        fine_sequence->SetjformStart(jFormStart);

        // replace mass integrators, we do form = 0 because it may be the hardest
        const int form = 0;
        ConstantCoefficient coeffSpace(1.0);
        ConstantCoefficient coeffDer(1.0);
        fine_sequence->ReplaceMassIntegrator(
            at_elem, form, make_unique<MassIntegrator>(coeffSpace), false);
        fine_sequence->ReplaceMassIntegrator(
            at_elem, form+1, make_unique<VectorFEMassIntegrator>(coeffDer), true);

        // set up coefficients / targets
        const int upscalingOrder = 0;
        fine_sequence->SetUpscalingTargets(pmesh->Dimension(), upscalingOrder);

        const double tolSVD = 1.e-9;
        fine_sequence->SetSVDTol(tolSVD);
        std::shared_ptr<DeRhamSequence> coarse_sequence = fine_sequence->Coarsen();
    }
    std::cout << "Success." << std::endl;

    // another thing to do is something in parallel (will have to adjust the
    // (fake partitioning routines)

    return EXIT_SUCCESS;
}
