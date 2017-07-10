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

#include "elag_topology.hpp"

void ShowTopologyAgglomeratedElements(AgglomeratedTopology * topo, ParMesh * mesh)
{

	MPI_Comm comm = mesh->GetComm();

	int num_procs, myid;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myid);

	SharingMap & elementTrueElement(topo->EntityTrueEntity(0));
	int poff = elementTrueElement.MyGlobalOffset();

	int nFineElements = mesh->GetNE();
	Array<int> partitioning(nFineElements), colors(nFineElements), help(nFineElements);

	int nAgglomeratedElements = topo->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT);

	partitioning.SetSize( nAgglomeratedElements );
	for( int i = 0; i < nAgglomeratedElements; ++i )
		partitioning[i] = i+poff;

	colors.SetSize( nAgglomeratedElements );
	SerialCSRMatrix * el_el = topo->LocalElementElementTable();
	GetElementColoring(colors, 0, *el_el );

	AgglomeratedTopology * it = topo;
	while( (it = it->finerTopology) != static_cast<AgglomeratedTopology *>(NULL) )
	{
		help.SetSize( it->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT) );
		it->AEntityEntity(0).WedgeMultTranspose(partitioning, help);
		Swap(partitioning, help);
	}

	it = topo;
	while( (it = it->finerTopology) != static_cast<AgglomeratedTopology *>(NULL) )
	{
		help.SetSize( it->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT) );
		it->AEntityEntity(0).WedgeMultTranspose(colors, help);
		Swap(colors, help);
	}

	FiniteElementCollection *fec = new L2_FECollection(0, mesh->Dimension());
	FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
	GridFunction x(fespace);

	for(int i = 0; i < x.Size(); ++i )
		x(i) = colors[i]*num_procs + myid;

	char vishost[] = "localhost";
	int  visport   = 19916;
	socketstream sol_sock (vishost, visport);
	sol_sock<<"parallel " << num_procs << " " << myid << "\n";
	sol_sock << "solution\n";
	mesh->PrintWithPartitioning(partitioning.GetData(),sol_sock);
	x.Save(sol_sock);

	delete fespace;
	delete fec;
}

void ShowTopologyAgglomeratedFacets3D(AgglomeratedTopology * topo, ParMesh * mesh)
{
#warning THIS METHOD IS BROKEN
	MPI_Comm comm = mesh->GetComm();

	int num_procs, myid;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myid);

	SharingMap & facetTrueFacet(topo->EntityTrueEntity(1));
	int poff = facetTrueFacet.MyGlobalOffset();

	int nAgglomeratedFacets = topo->GetNumberLocalEntities(AgglomeratedTopology::FACET);
	SerialCSRMatrix * AF_fc = createSparseIdentityMatrix(nAgglomeratedFacets);

	AgglomeratedTopology * it = topo;
	while( (it = it->finerTopology) != static_cast<AgglomeratedTopology *>(NULL) )
	{
		SerialCSRMatrix * help = Mult( *AF_fc, it->AEntityEntity(1) );
		delete AF_fc;
		AF_fc = help;
	}

	SerialCSRMatrix * fc_AF = Transpose( *AF_fc );
	delete AF_fc;

	int * j_fc_AF = fc_AF->GetJ();
	int * j_fc_AFoffset = new int[fc_AF->NumNonZeroElems()];
	for(int i = 0; i < fc_AF->NumNonZeroElems(); ++i)
		j_fc_AFoffset[i] = j_fc_AF[i] + poff;

	SerialCSRMatrix * fc_AFoffset = new SerialCSRMatrix(fc_AF->GetI(), j_fc_AFoffset, fc_AF->GetData(), fc_AF->Size(), fc_AF->Width()+poff);

	FiniteElementCollection *fec = new L2_FECollection(0, mesh->Dimension());
	FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
	GridFunction x(fespace);

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
	delete fc_AFoffset;
	delete fc_AF;
	delete fespace;
	delete fec;

}

void ShowTopologyAgglomeratedFacets(AgglomeratedTopology * topo, ParMesh * mesh)
{
	if(mesh->Dimension() == 3)
		ShowTopologyAgglomeratedFacets3D(topo, mesh);
	else
		std::cout << "ShowTopologyAgglomeratedFacets not implemented for Dimension different from 3!!\n";
}

void ShowTopologyBdrFacets3D(AgglomeratedTopology * topo, ParMesh * mesh)
{
	MPI_Comm comm = mesh->GetComm();

	int num_procs, myid;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myid);

	SerialCSRMatrix * bdrAttribute_fc( Transpose( topo->FacetBdrAttribute() ) );
	AgglomeratedTopology * it = topo;
	while( (it = it->finerTopology) != static_cast<AgglomeratedTopology *>(NULL) )
	{
		SerialCSRMatrix * help = Mult( *bdrAttribute_fc, it->AEntityEntity(1) );
		delete bdrAttribute_fc;
		bdrAttribute_fc = help;
	}
	SerialCSRMatrix * A = Transpose(*bdrAttribute_fc);

	FiniteElementCollection *fec = new L2_FECollection(0, mesh->Dimension());
	FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
	GridFunction x(fespace);

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

	delete A;
	delete bdrAttribute_fc;
	delete fespace;
	delete fec;
}

void ShowTopologyBdrFacets(AgglomeratedTopology * topo, ParMesh * mesh)
{
	if(mesh->Dimension() == 3)
		ShowTopologyBdrFacets3D(topo, mesh);
	else
		std::cout << "ShowTopologyBdrFacets not implemented for Dimension different from 3!!\n";
}

void ShowAgglomeratedTopology3D(AgglomeratedTopology * topo, ParMesh * mesh)
{

	MPI_Comm comm = mesh->GetComm();

	int num_procs, myid;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myid);

	SharingMap & elementTrueElement(topo->EntityTrueEntity(0));
	int poff = elementTrueElement.MyGlobalOffset();

	int nFineElements = mesh->GetNE();
	Array<int> partitioning(nFineElements), colors(nFineElements), help(nFineElements);

	int nAgglomeratedElements = topo->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT);

	partitioning.SetSize( nAgglomeratedElements );
	for( int i = 0; i < nAgglomeratedElements; ++i )
		partitioning[i] = i+poff;

	AgglomeratedTopology * it = topo;
	while( (it = it->finerTopology) != static_cast<AgglomeratedTopology *>(NULL) )
	{
		help.SetSize( it->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT) );
		it->AEntityEntity(0).WedgeMultTranspose(partitioning, help);
		Swap(partitioning, help);
	}

	int nAgglomeratedFacets = topo->GetNumberLocalEntities(AgglomeratedTopology::FACET);
	colors.SetSize( nAgglomeratedFacets );
	SerialCSRMatrix * AR_AF = Transpose( topo->B(1) );
	SerialCSRMatrix * AF_AF = Mult( topo->B(1), *AR_AF);
	delete AR_AF;
	GetElementColoring(colors, 0, *AF_AF );

	Vector dcolors(nAgglomeratedFacets), dhelp;
	for(int i = 0; i < nAgglomeratedFacets; ++i)
		dcolors(i) = static_cast<double>(colors[i])+1.;

	it = topo;
	while( (it = it->finerTopology) != static_cast<AgglomeratedTopology *>(NULL) )
	{
		dhelp.SetSize( it->GetNumberLocalEntities(AgglomeratedTopology::FACET) );
		it->AEntityEntity(1).MultTranspose(dcolors, dhelp);
		Swap(dcolors, dhelp);
	}


	FiniteElementCollection *fec = new RT_FECollection(0, mesh->Dimension());
	FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
	GridFunction x;
	x.Update(fespace,dcolors, 0);

	for(int i = 0; i < x.Size(); ++i )
	{
		ElementTransformation *et = mesh->GetFaceTransformation(i);
		const IntegrationRule &ir = IntRules.Get(mesh->GetFaceBaseGeometry(i), et->OrderJ());
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

	delete fespace;
	delete fec;
}

