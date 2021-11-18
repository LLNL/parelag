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

#include "Topology.hpp"

#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "structures/Coloring.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

void ShowTopologyAgglomeratedElements(
    AgglomeratedTopology * topo,
    ParMesh * mesh,
    std::ofstream * file)
{

    MPI_Comm comm = mesh->GetComm();

    int num_procs, myid;

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    SharingMap & elementTrueElement(topo->EntityTrueEntity(0));
    const int poff = elementTrueElement.MyGlobalOffset();

    const int nFineElements = mesh->GetNE();
    Array<int> partitioning(nFineElements),
        colors(nFineElements),
        help(nFineElements);

    const int nAgglomeratedElements
        = topo->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT);

    partitioning.SetSize(nAgglomeratedElements);
    for (int i = 0; i < nAgglomeratedElements; ++i)
        partitioning[i] = i+poff;

    colors.SetSize( nAgglomeratedElements );
    SerialCSRMatrix * el_el = topo->LocalElementElementTable();
    GetElementColoring(colors, 0, *el_el );

    AgglomeratedTopology* it = topo;
    while((it = it->FinerTopology().get()) != nullptr)
    {
        help.SetSize(it->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT));
        it->AEntityEntity(0).WedgeMultTranspose(partitioning, help);
        Swap(partitioning, help);
    }

    it = topo;
    while((it = it->FinerTopology().get()) != nullptr)
    {
        help.SetSize( it->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT) );
        it->AEntityEntity(0).WedgeMultTranspose(colors, help);
        Swap(colors, help);
    }

    auto fec = make_unique<L2_FECollection>(0, mesh->Dimension());
    auto fespace = make_unique<FiniteElementSpace>(mesh, fec.get());
    GridFunction x(fespace.get());

    for(int i = 0; i < x.Size(); ++i )
        x(i) = partitioning[i]*num_procs + myid;

    if(file == nullptr)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock (vishost, visport);
        if (num_procs > 1)
            sol_sock<<"parallel " << num_procs << " " << myid << "\n";
        sol_sock << "solution\n";
        mesh->PrintWithPartitioning(partitioning.GetData(), sol_sock, 1);
        x.Save(sol_sock);
    }
    else
    {
        mesh->PrintWithPartitioning(partitioning.GetData(), *file, 1);
        x.Save(*file);
    }

    MPI_Barrier(comm);
}

void ShowTopologyAgglomeratedFacets3D(AgglomeratedTopology * topo, ParMesh * mesh)
{
    // this method does not work and is not implemented properly
    PARELAG_ASSERT(false);

    MPI_Comm comm = mesh->GetComm();

    int num_procs, myid;

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    SharingMap & facetTrueFacet(topo->EntityTrueEntity(1));
    const int poff = facetTrueFacet.MyGlobalOffset();

    const int nAgglomeratedFacets
        = topo->GetNumberLocalEntities(AgglomeratedTopology::FACET);
    auto AF_fc = createSparseIdentityMatrix(nAgglomeratedFacets);

    AgglomeratedTopology * it = topo;
    while( (it = it->FinerTopology().get()) != nullptr )
    {
        AF_fc.reset(Mult(*AF_fc,it->AEntityEntity(1)));
    }

    unique_ptr<SerialCSRMatrix> fc_AF{Transpose(*AF_fc)};
    AF_fc.reset();

    int * j_fc_AF = fc_AF->GetJ();
    int * j_fc_AFoffset = new int[fc_AF->NumNonZeroElems()];
    for(int i = 0; i < fc_AF->NumNonZeroElems(); ++i)
        j_fc_AFoffset[i] = j_fc_AF[i] + poff;

    auto fc_AFoffset = make_unique<SerialCSRMatrix>(fc_AF->GetI(),
                                                    j_fc_AFoffset,
                                                    fc_AF->GetData(),
                                                    fc_AF->Size(),
                                                    fc_AF->Width()+poff);

    auto fec = make_unique<L2_FECollection>(0, mesh->Dimension());
    auto fespace = make_unique<FiniteElementSpace>(mesh, fec.get());
    GridFunction x(fespace.get());

    x = myid;

    char vishost[] = "localhost";
    int  visport   = 19916;
    osockstream sol_sock (visport, vishost);
    sol_sock << "parallel " << num_procs << " " << myid << "\n";
    sol_sock << "solution\n";
    {
        Table surf;
        surf.SetIJ(fc_AFoffset->GetI(), fc_AFoffset->GetJ(), fc_AFoffset->Size() );
        mesh->PrintSurfaces(surf, sol_sock);
        surf.LoseData();
    }
    x.Save(sol_sock);

    fc_AFoffset->LoseData();
    delete[] j_fc_AFoffset;
}

void ShowTopologyAgglomeratedFacets(AgglomeratedTopology * topo,
                                    ParMesh * mesh)
{
    if(mesh->Dimension() == 3)
        ShowTopologyAgglomeratedFacets3D(topo, mesh);
    else
        std::cout << "ShowTopologyAgglomeratedFacets not implemented "
                  << "for Dimension different from 3!!" << std::endl;
}

void ShowTopologyBdrFacets3D(AgglomeratedTopology * topo, ParMesh * mesh)
{
    MPI_Comm comm = mesh->GetComm();

    int num_procs, myid;

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    unique_ptr<SerialCSRMatrix> bdrAttribute_fc{
        Transpose(topo->FacetBdrAttribute()) };
    AgglomeratedTopology * it = topo;
    while( (it = it->FinerTopology().get()) != nullptr )
    {
        bdrAttribute_fc.reset(Mult(*bdrAttribute_fc, it->AEntityEntity(1)));
    }

    unique_ptr<SerialCSRMatrix> A{Transpose(*bdrAttribute_fc)};

    auto fec = make_unique<L2_FECollection>(0, mesh->Dimension());
    auto fespace = make_unique<FiniteElementSpace>(mesh, fec.get());
    GridFunction x(fespace.get());

    x = myid;

    char vishost[] = "localhost";
    int  visport   = 19916;
    osockstream sol_sock (visport, vishost);
    sol_sock<<"parallel " << num_procs << " " << myid << "\n";
    sol_sock << "solution\n";
    {
        Table surf;
        surf.SetIJ(A->GetI(), A->GetJ(), A->Size() );
        mesh->PrintSurfaces(surf, sol_sock);
        surf.LoseData();
    }
    x.Save(sol_sock);
}

void ShowTopologyBdrFacets(AgglomeratedTopology * topo, ParMesh * mesh)
{
    if(mesh->Dimension() == 3)
        ShowTopologyBdrFacets3D(topo, mesh);
    else
        std::cout << "ShowTopologyBdrFacets not implemented "
                  << "for Dimension different from 3!!" << std::endl;
}

/// I do not know what this actually shows and I cannot interpret its output
void ShowAgglomeratedTopology3D(AgglomeratedTopology * topo, ParMesh * mesh)
{
    MPI_Comm comm = mesh->GetComm();

    int num_procs, myid;

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    SharingMap & elementTrueElement(topo->EntityTrueEntity(0));
    const int poff = elementTrueElement.MyGlobalOffset();

    const int nFineElements = mesh->GetNE();
    Array<int> partitioning(nFineElements),
        colors(nFineElements),
        help(nFineElements);

    const int nAgglomeratedElements
        = topo->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT);

    partitioning.SetSize( nAgglomeratedElements );
    for( int i = 0; i < nAgglomeratedElements; ++i )
        partitioning[i] = i+poff;

    AgglomeratedTopology * it = topo;
    while ((it = it->FinerTopology().get()) != nullptr)
    {
        help.SetSize( it->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT));
        it->AEntityEntity(0).WedgeMultTranspose(partitioning, help);
        Swap(partitioning, help);
    }

    const int nAgglomeratedFacets
        = topo->GetNumberLocalEntities(AgglomeratedTopology::FACET);
    colors.SetSize(nAgglomeratedFacets);
    unique_ptr<SerialCSRMatrix> AR_AF{Transpose(topo->GetB(1))};
    unique_ptr<SerialCSRMatrix> AF_AF{Mult( topo->GetB(1), *AR_AF)};
    AR_AF.reset();
    GetElementColoring(colors, 0, *AF_AF );
    AF_AF.reset();

    Vector dcolors(nAgglomeratedFacets), dhelp;
    for(int i = 0; i < nAgglomeratedFacets; ++i)
        dcolors(i) = static_cast<double>(colors[i])+1.;

    it = topo;
    while( (it = it->FinerTopology().get()) != nullptr )
    {
        dhelp.SetSize( it->GetNumberLocalEntities(AgglomeratedTopology::FACET) );
        it->AEntityEntity(1).MultTranspose(dcolors, dhelp);
        Swap(dcolors, dhelp);
    }

    auto fec = make_unique<RT_FECollection>(0, mesh->Dimension());
    auto fespace = make_unique<FiniteElementSpace>(mesh, fec.get());
    GridFunction x;

    x.MakeRef(fespace.get(),dcolors, 0);
    for(int i = 0; i < x.Size(); ++i )
    {
        ElementTransformation *et = mesh->GetFaceTransformation(i);
        const IntegrationRule &ir
            = IntRules.Get(mesh->GetFaceBaseGeometry(i), et->OrderJ());
        double surf = 0.0, scaling = 0.0;
        for (int j = 0; j < ir.GetNPoints(); j++)
        {
            const IntegrationPoint &ip = ir.IntPoint(j);
            et->SetIntPoint(&ip);
            surf += ip.weight * et->Weight();
            scaling += ip.weight;
        }

        x(i) *= surf/scaling;
    }

    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream sol_sock (vishost, visport);
    sol_sock<<"parallel " << num_procs << " " << myid << "\n";
    sol_sock << "solution\n";
    mesh->PrintWithPartitioning(partitioning.GetData(),sol_sock);
    x.Save(sol_sock);
}
}//namespace parelag
