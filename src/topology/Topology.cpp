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
#include "../hypreExtension/hypreExtension.hpp"
#include "../partitioning/MFEMRefinedMeshPartitioner.hpp"
#include<algorithm>
#include<numeric>

AgglomeratedTopology::AgglomeratedTopology(MPI_Comm comm_, int ncodim_):
    finerTopology(static_cast<AgglomeratedTopology *>(NULL)),
    coarserTopology( static_cast<AgglomeratedTopology *>(NULL) ),
    comm(comm_),
    ndim(-1),
	ncodim(ncodim_),
	pmesh( static_cast<ParMesh *>(NULL) ),
	B_(ncodim),
    w(ncodim+1),
    element_attribute(0),
    facet_bdrAttribute( static_cast<TopologyTable *>(NULL) ),
    entityTrueEntity(ncodim+1),
    globalAgglomeration(-1),
    AEntity_entity(0),
    ATrueEntity_trueEntity(0),
    workspace(ncodim)
{
	B_ = static_cast<TopologyTable *>(NULL);

	for(int i = 0; i < ncodim+1; ++i)
		w[i] = new TopologyArray();

	for(int i = 0; i < ncodim + 1; ++i)
		entityTrueEntity[i] = new SharingMap(comm);
}

AgglomeratedTopology::AgglomeratedTopology(ParMesh * pmesh_, int ncodim_):
	finerTopology(static_cast<AgglomeratedTopology *>(NULL)),
	coarserTopology( static_cast<AgglomeratedTopology *>(NULL) ),
	comm(pmesh_->GetComm()),
	ndim(pmesh_->Dimension()),
	ncodim(ncodim_),
	pmesh(pmesh_),
	B_(ncodim),
	w(ndim),
	element_attribute(0),
	facet_bdrAttribute( static_cast<TopologyTable *>(NULL) ),
	entityTrueEntity(ncodim+1),
	globalAgglomeration(-1),
	AEntity_entity(0),
	ATrueEntity_trueEntity(0),
	workspace(ncodim)
{
	elag_trace_enter_block("AgglomeratedTopology::AgglomeratedTopology( pmesh_ = " << pmesh_ << ", ncodim_ = " << ncodim_ << ")" );
	B_ = static_cast<TopologyTable *>(NULL);

	for(int i = 0; i < w.Size(); ++i)
		w[i] = new TopologyArray();

	for(int i = 0; i < ncodim + 1; ++i)
		entityTrueEntity[i] = new SharingMap(comm);

	generateTopology(pmesh);
	elag_trace_leave_block("AgglomeratedTopology::AgglomeratedTopology( pmesh_ = " << pmesh_ << ", ncodim_ = " << ncodim_ << ")" );
}

AgglomeratedTopology::~AgglomeratedTopology()
{
	for(int i = 0; i < AEntity_entity.Size(); ++i)
		delete AEntity_entity[i];

	for(int i = 0; i < ATrueEntity_trueEntity.Size(); ++i)
		delete ATrueEntity_trueEntity[i];

	for(int i = 0; i < entityTrueEntity.Size(); ++i)
		delete entityTrueEntity[i];

	for(int i = 0; i < ncodim; ++i)
		delete B_[i];

	for(int i = 0; i < w.Size(); ++i)
		delete w[i];


	for(int j = 0; j < conn.NumCols(); ++j)
		for(int i = 0; i < conn.NumRows(); ++i)
			delete conn(i,j);

	delete facet_bdrAttribute;
}

void AgglomeratedTopology::generateTopology(ParMesh * pmesh)
{
	elag_trace_enter_block("AgglomeratedTopology::generateTopology( pmesh_ = " << pmesh << ")" );
	Array<FiniteElementCollection *> fecColl(ndim+1);

	fecColl[0] = new L2_FECollection(0, ndim );
	fecColl[1] = new RT_FECollection(0, ndim );
	if(ndim == 2)
	{
		fecColl[2] = new H1_FECollection(1, ndim );
	}
	else
	{
		fecColl[2] = new ND_FECollection(1, ndim );
		fecColl[3] = new H1_FECollection(1, ndim );
	}

	Array<FiniteElementSpace *> feSpace( ncodim+1);
	for(int i(0); i < ncodim+1; ++ i)
		feSpace[i] = new FiniteElementSpace(pmesh, fecColl[i]);

	// Define bilinear integrators
	Array<DiscreteLinearOperator *> Bvarf(ncodim);
	Array<DiscreteInterpolator *> Bop(ncodim);

	// NOTE THIS DISCRETEINTERPOLATOR will be deallocated by the destructor of Bvarf
	Bop[0] = new DivergenceInterpolator();
	if(ncodim > 1)
	{
		if(ndim == 2)
		{
			Bop[1] = new GradientInterpolator();
		}
		else
		{
			Bop[1] = new CurlInterpolator();
			if(ncodim == 3)
				Bop[2] = new GradientInterpolator();
		}
	}

	for(int i(0); i<ncodim; ++i)
	{
		Bvarf[i] = new DiscreteLinearOperator(feSpace[i+1], feSpace[i]);
		Bvarf[i]->AddDomainInterpolator(Bop[i]);
		Bvarf[i]->Assemble();
		Bvarf[i]->Finalize();

		B_[i] = new TopologyTable( Bvarf[i]->LoseMat() );
		B_[i]->OrientationTransform();
	}

	// Delete the BilinearForms
	for(int i(0); i < ncodim; ++ i)
		delete Bvarf[i];

	// Build the SharingMap for each codimension.
	elag_trace("Build the SharingMap for each codimension");
	for(int i(0); i < ncodim+1; ++i)
		entityTrueEntity[i]->SetUp(pmesh, i);

	// Delete the finite element collections
	for( int i(0); i < ncodim+1; ++i)
		delete feSpace[i];

	for( int i(0); i < ndim+1; ++i)
		delete fecColl[i];

	initializeWeights(pmesh);

	element_attribute.SetSize( pmesh->GetNE() );
	for( int i = 0; i < pmesh->GetNE(); ++i)
		element_attribute[i] = static_cast<TopologyDataType>( pmesh->GetAttribute(i) );

	if(ncodim >= 1)
		generateFacetBdrAttributeTable(pmesh);

	BuildConnectivity();
	elag_trace_leave_block("AgglomeratedTopology::generateTopology( pmesh_ = " << pmesh << ")" );
}

void AgglomeratedTopology::initializeWeights(Mesh *mesh)
{
	w[ELEMENT]->SetSize( mesh->GetNE() );

	if(ncodim > 0)
	{
		if(ndim == 3)
		{
			w[FACET]->SetSize(mesh->GetNFaces());
			if(ncodim > 1)
				w[RIDGE]->SetSize(mesh->GetNEdges());
		}
		else
		{
			w[FACET]->SetSize(mesh->GetNEdges());
		}
	}

	for(int i = 0; i < std::min(ncodim+1, ndim); ++i)
		(*w[i]) = static_cast<TopologyDataType>(1);

}

void AgglomeratedTopology::generateFacetBdrAttributeTable(Mesh *mesh)
{
	int nFacets = w[FACET]->Size();
	int nnz = mesh->GetNBE();
	int * i_mat = new int[nFacets+1];
	std::fill( i_mat, i_mat+nFacets+1, 0 );

	if(mesh->bdr_attributes.Size() == 0)
	{
		int * j_mat = new int[1];
		TopologyDataType * a_mat = new TopologyDataType[1];
		facet_bdrAttribute = new TopologyTable(i_mat, j_mat, a_mat, nFacets, 0);
		return;
	}

	int * j_mat = new int[nnz];
	TopologyDataType * a_mat = new TopologyDataType[nnz];
	int nbdrattributes = mesh->bdr_attributes.Max();

	// NOTE in 2D MFEM does not orient boundary facets in a consistent way.
	// Therefore we need to use the sign of facet_element to make sure that
	// the orientation is correct on each boundary surface.
	SparseMatrix * facet_element = Transpose(*(B_[0]));

	int * i_facet_element = facet_element->GetI();
	double * a_facet_element = facet_element->GetData();

	int index;
	//mark which facet is a boundary facet.
	for(int i = 0; i < nnz; ++i)
	{
		index = mesh->GetBdrElementEdgeIndex(i);
		(i_mat[index+1])++;
	}

	// do a partial sum to compute the i array of the CSRmatrix
	std::partial_sum(i_mat, i_mat+nFacets+1, i_mat);

	// Fill in the CSR matrix
	for(int i = 0; i < nnz; ++i)
	{
		// Get the facet index corresponding to boundary element i
		index = mesh->GetBdrElementEdgeIndex(i);
		// fill the j array corresponding to facet index to the respective boundary attribute
		j_mat[ i_mat[index] ] = mesh->GetBdrAttribute(i)-1;
		// This is a redundant check. If index is a bdr facets that it is adjacent to only 1 element.
		elag_assert(i_facet_element[index+1] - i_facet_element[index] == 1);
		// We orient the bdr facet as the opposite of its local orientation in the adjacent element.
		a_mat[ i_mat[index] ] = -a_facet_element[ i_facet_element[index] ];
	}

	delete facet_element;
	facet_bdrAttribute = new TopologyTable(i_mat, j_mat, a_mat, nFacets, nbdrattributes);
}

void AgglomeratedTopology::BuildConnectivity()
{
	conn.SetSize(ncodim+1, ncodim+1);
	conn =  static_cast<BooleanMatrix *>(NULL);

	for(int codim(0); codim < ncodim; ++codim)
		conn(codim, codim+1) = new BooleanMatrix( B(codim) );

	if(ncodim > 1)
		conn(ELEMENT, RIDGE) = BoolMult(B(ELEMENT), B(FACET));
	if(ncodim > 2)
	{
		conn(FACET, PEAK)   = BoolMult(B(FACET), B(RIDGE));
		conn(ELEMENT, PEAK) = BoolMult(*conn(ELEMENT, RIDGE), B(RIDGE) );
	}

}

const BooleanMatrix & AgglomeratedTopology::GetConnectivity(int range, int domain) const
{
	elag_assert(conn(range, domain) != NULL);

	return *conn(range, domain);
}

void AgglomeratedTopology::GetBoundaryOfEntity(EntityByCodim type, EntityByCodim type_bdr, int ientity, Array<int> & bdr_entity) const
{
	elag_assert(conn(type, type_bdr) != NULL);
	conn(type, type_bdr)->GetRow(ientity, bdr_entity);
}

SerialCSRMatrix * AgglomeratedTopology::LocalElementElementTable()
{
	if(workspace.elem_elem_local)
		return workspace.elem_elem_local;

	TopologyTable * Bt = B(0).Transpose( );

	Bt->ScaleRows( Weight(1) );

	workspace.elem_elem_local = Mult( B(0), *Bt);

	delete Bt;

	return workspace.elem_elem_local;
}

AgglomeratedTopology::TopologyParTable * AgglomeratedTopology::GlobalElementElementTable()
{
	if(workspace.elem_elem_global)
		return workspace.elem_elem_global;

	TopologyParTable * Bt = TrueB(0).Transpose( );

	TopologyArray * wT = TrueWeight(1);
	Bt->ScaleRows( *wT );
	delete wT;


	workspace.elem_elem_global = ParMult( &TrueB(0), Bt);

	delete Bt;

	return workspace.elem_elem_global;

}

void AgglomeratedTopology::ShowMe(std::ostream & os)
{
	os << "N_elements = " << std::setw(10) <<  GetNumberLocalEntities(ELEMENT)
		                  << std::setw(10) <<  GetNumberGlobalTrueEntities(ELEMENT) << "\n";
	os << "N_facets   = " << std::setw(10) <<  GetNumberLocalEntities(FACET)
				          << std::setw(10) <<  GetNumberGlobalTrueEntities(FACET) << "\n";
	os << "N_ridges   = " << std::setw(10) <<  GetNumberLocalEntities(RIDGE)
				          << std::setw(10) <<  GetNumberGlobalTrueEntities(RIDGE) << "\n";
	if(ndim == 3)
		os << "N_peaks    = " << std::setw(10) <<  GetNumberLocalEntities(PEAK)
		                      << std::setw(10) <<  GetNumberGlobalTrueEntities(PEAK) << "\n";

	if(ndim == ncodim)
	{
		int eulerNumberLocal  = 0;
		int eulerNumberGlobal = 0;
		for(int codim(ndim); codim >= 0; codim-=2)
		{
			eulerNumberLocal  += GetNumberLocalEntities(static_cast<EntityByCodim>(codim));
			eulerNumberGlobal += GetNumberGlobalTrueEntities(static_cast<EntityByCodim>(codim));
		}
		for(int codim(ndim-1); codim >= 0; codim-=2)
		{
			eulerNumberLocal  -= GetNumberLocalEntities(static_cast<EntityByCodim>(codim));
			eulerNumberGlobal -= GetNumberGlobalTrueEntities(static_cast<EntityByCodim>(codim));
		}

		os << "Euler Number = " << std::setw(10) << eulerNumberLocal
			                << std::setw(10) << eulerNumberGlobal << "\n";
	}

	Array<std::string *> entityName(ncodim);
	entityName[0] = new std::string("Element");
	entityName[1] = new std::string("Facet");
	if(ncodim == 3)
		entityName[2] = new std::string("Ridge");

	Array<int> minW(ncodim), maxW(ncodim);
	Array<double> meanW(ncodim);

	for(int codim(0); codim < ncodim; ++codim)
	{
		TopologyArray & wc(*w[codim]);
		TopologyDataType minw( wc[0] );
		TopologyDataType maxw( wc[0] );
		TopologyDataType sumw( static_cast<TopologyDataType>(0) );

		for(TopologyDataType * it = wc.GetData(); it != wc.GetData()+wc.Size(); ++it)
		{
			minw = (minw < *it) ? minw : *it;
			maxw = (maxw > *it) ? maxw : *it;
			sumw += *it;
		}

		minW[codim] = minw;
		maxW[codim] = maxw;
		meanW[codim] = static_cast<double>(sumw) / static_cast<double>( wc.Size() );
	}

	os << "Entity   MIN  MAX AVERAGE (weight) \n";
	for(int codim(0); codim < ncodim; ++codim)
		os << *(entityName[codim]) << " " << minW[codim] << " " << maxW[codim] << " " << meanW[codim] << "\n";

	for(int codim(0); codim < ncodim; ++codim)
		delete entityName[codim];

}

AgglomeratedTopology * AgglomeratedTopology::FinestTopology()
{
	AgglomeratedTopology * topo = this;

	while( topo->finerTopology )
		topo = topo->finerTopology;

	return topo;
}

AgglomeratedTopology::TopologyParTable & AgglomeratedTopology::TrueB(int i)
{
	if(workspace.trueB_[i] == 0)
		workspace.trueB_[i] = IgnoreNonLocalRange( *(entityTrueEntity[i]), *(B_[i]), *(entityTrueEntity[i+1]));

	return *(workspace.trueB_[i]);
}

AgglomeratedTopology::TopologyArray * AgglomeratedTopology::TrueWeight(int i)
{
	TopologyArray * truew = new TopologyArray( entityTrueEntity[i]->GetTrueLocalSize() );
	entityTrueEntity[i]->IgnoreNonLocal( *(w[i]), *truew );

	return truew;
}

AgglomeratedTopology * AgglomeratedTopology::CoarsenLocalPartitioning(Array<int> & partitioning_, int checkTopology, int preserveMaterialInterfaces)
{
	partitioning_.Copy(partitioning);

	globalAgglomeration = 0;
	coarserTopology = new AgglomeratedTopology(comm, ncodim);
	coarserTopology->finerTopology = this;
	coarserTopology->ndim = ndim;

	int nAE;
	if(preserveMaterialInterfaces == 0)
		nAE = connectedComponents(partitioning, *LocalElementElementTable());
	else
		nAE = connectedComponents(partitioning, *LocalElementElementTable(), element_attribute);

	elag_assert(AEntity_entity.Size() == 0);
	AEntity_entity.SetSize(ncodim+1);
	AEntity_entity[0] = TransposeOrientation(partitioning, nAE);

	if(checkTopology)
	{
		std::stringstream os;
		AgglomeratedTopologyCheck::ShowBadAgglomeratedEntities(0,*this, os);
		SerializedOutput(comm, std::cout, os.str());
	}

	coarserTopology->entityTrueEntity[0]->SetUp(nAE);

	for(int icodim = 0; icodim < ncodim; ++icodim)
	{
		TopologyTable * AE_fc = MultOrientation( *(AEntity_entity[icodim]), *(B_[icodim]) );
		TopologyTable * fc_AE = AE_fc->Transpose( );
		SerialCSRMatrix * fc_AE_fc = Mult(*fc_AE, *AE_fc);
		delete fc_AE;

		if(icodim == 0 && facet_bdrAttribute)
		{
			TopologyTable * bdrAttribute_facet = facet_bdrAttribute->Transpose();
			SerialCSRMatrix * fc_bdrAttr_fc = Mult(*facet_bdrAttribute, * bdrAttribute_facet);
			SerialCSRMatrix * tmp = fc_AE_fc;
			fc_AE_fc = Add( *tmp, *fc_bdrAttr_fc);
			delete tmp;
			delete bdrAttribute_facet;
			delete fc_bdrAttr_fc;
		}
		SerialCSRMatrix * Z = AssembleNonLocal( *(entityTrueEntity[icodim+1]), *fc_AE_fc, *(entityTrueEntity[icodim+1]) );
		TopologyTable * fc_AF = new TopologyTable( findMinimalIntersectionSets(*Z, .5) );
		AEntity_entity[icodim+1] = fc_AF->Transpose( );

		if(checkTopology)
		{
			std::stringstream os;
			AgglomeratedTopologyCheck::ShowBadAgglomeratedEntities(icodim+1,*this, os);
			SerializedOutput(comm, std::cout, os.str());
		}

		coarserTopology->B_[icodim] = MultOrientation( *AE_fc, *fc_AF);
		coarserTopology->entityTrueEntity[icodim+1]->SetUp( *fc_AF, *(entityTrueEntity[icodim+1]) );

		delete fc_AF;
		delete Z;
		delete fc_AE_fc;
		delete AE_fc;
	}

	if(facet_bdrAttribute)
		coarserTopology->facet_bdrAttribute = MultOrientation(*(AEntity_entity[1]), *facet_bdrAttribute);

	for(int icodim = 0; icodim < std::min(ncodim+1,ndim); ++icodim)
	{
		coarserTopology->w[icodim]->SetSize( AEntity_entity[icodim]->Size() );
		AEntity_entity[icodim]->WedgeMult( *(w[icodim]), *(coarserTopology->w[icodim]) );
	}

	coarserTopology->BuildConnectivity();

	return coarserTopology;
}

AgglomeratedTopology * AgglomeratedTopology::UniformRefinement()
{
	elag_assert(pmesh);
	elag_assert( this == FinestTopology() );
	int nAE = pmesh->GetNE();
	pmesh->UniformRefinement();
	finerTopology = new AgglomeratedTopology(pmesh, ncodim);
	finerTopology->coarserTopology = this;
	finerTopology->globalAgglomeration = 0;
	finerTopology->partitioning.SetSize( pmesh->GetNE() );

	MFEMRefinedMeshPartitioner partitioner(ndim);
	partitioner.Partition(pmesh->GetNE(), nAE, finerTopology->partitioning);

	elag_assert(finerTopology->AEntity_entity.Size() == 0);
	finerTopology->AEntity_entity.SetSize(ncodim+1);
	finerTopology->AEntity_entity = NULL;
	finerTopology->AEntity_entity[0] = TransposeOrientation(finerTopology->partitioning, nAE);

	switch(ncodim)
	{
	case 0:
		break;
	case 1:
		finerTopology->AEntity_entity[1] = generate_AF_f_ForUniformRefinement();
		break;
	case 2:
		finerTopology->AEntity_entity[1] = generate_AF_f_ForUniformRefinement();
		finerTopology->AEntity_entity[2] = generate_AR_r_ForUniformRefinement();
		break;
	case 3:
		finerTopology->AEntity_entity[1] = generate_AF_f_ForUniformRefinement();
		finerTopology->AEntity_entity[2] = generate_AR_r_ForUniformRefinement();
		finerTopology->AEntity_entity[3] = generate_AP_p_ForUniformRefinement();
		break;
	default:
		elag_error(1);
	}

	pmesh = NULL;
	return finerTopology;

}

TopologyTable * FindIntersectionsAF(SerialCSRMatrix * Z)
{
	int nAF = Z->Size();
	int nf = Z->Width();

	int * const i_Z = Z->GetI();
	int * const j_Z = Z->GetJ();
	double * const a_Z = Z->GetData();

	int nnz = 0;
	int * i_AF_f = new int[nAF+1];

	double tol = 1e-9;
	double * it = a_Z, *end;
	for(int i(0); i < nAF; ++i)
	{
		i_AF_f[i] = nnz;
		for(end = a_Z + i_Z[i+1]; it != end; ++it)
		{
			elag_assert(fabs(*it) < 2.+tol);
			if( fabs(*it) > 2. - tol )
				++nnz;
		}
	}
	i_AF_f[nAF] = nnz;

	int * j_AF_f = new int[nnz];
	double * o_AF_f = new double[nnz];
	it = a_Z;
	int * j_it = j_Z;
	nnz = 0;
	for(int i(0); i < nAF; ++i)
	{
		for(end = a_Z + i_Z[i+1]; it != end; ++it, ++j_it)
		{
			elag_assert(fabs(*it) < 2.+tol);
			if( fabs(*it) > 2. - tol )
			{
				j_AF_f[nnz] = *j_it;
				o_AF_f[nnz] = (*it > 0.) ? 1.0:-1.0;
				++nnz;
			}
		}
	}

	return new TopologyTable(i_AF_f, j_AF_f, o_AF_f, nAF, nf);
}

TopologyTable * AgglomeratedTopology::generate_AF_f_ForUniformRefinement()
{
	TopologyTable * AF_AE = B_[0]->Transpose();
	TopologyTable * AF_bdr = facet_bdrAttribute;
	TopologyTable * AE_f  = MultOrientation( *(finerTopology->AEntity_entity[0]), *(finerTopology->B_[0]) );
	TopologyTable * bdr_f = finerTopology->FacetBdrAttribute().Transpose();
	SerialCSRMatrix * Z1 = Mult(*AF_AE, *AE_f);
	SerialCSRMatrix * Z2 = Mult(*AF_bdr, *bdr_f);
	SerialCSRMatrix * Z3 = AssembleNonLocal(*(entityTrueEntity[FACET]), *Z1, *(finerTopology->entityTrueEntity[FACET]) );
	SerialCSRMatrix * Z = Add(*Z2, *Z3);
	delete Z3;
	delete Z2;
	delete Z1;
	delete bdr_f;
	delete AE_f;
	delete AF_AE;
	TopologyTable * out = FindIntersectionsAF(Z);
	delete Z;
	return out;
}

TopologyTable * FindIntersectionsAR(SerialCSRMatrix * AR_AF, SerialCSRMatrix * Z)
{
	int nAR = Z->Size();
	int nr = Z->Width();

	int * const i_Z = Z->GetI();
	int * const j_Z = Z->GetJ();
	double * const a_Z = Z->GetData();

	int * const i_AR_AF = AR_AF->GetI();

	int nnz = 0;
	int * i_AR_r = new int[nAR+1];

	double tol = 1e-9;
	double * it = a_Z, *end;
	for(int i(0); i < nAR; ++i)
	{
		i_AR_r[i] = nnz;
		int nAdiacentFaces = i_AR_AF[i+1] - i_AR_AF[i];
		for(end = a_Z + i_Z[i+1]; it != end; ++it)
		{
			elag_assert(fabs(*it) < static_cast<double>(nAdiacentFaces)+tol);
			if( fabs(*it) > static_cast<double>(nAdiacentFaces) - tol )
				++nnz;
		}
	}
	i_AR_r[nAR] = nnz;

	int * j_AR_r = new int[nnz];
	double * o_AR_r = new double[nnz];
	it = a_Z;
	int * j_it = j_Z;
	nnz = 0;
	for(int i(0); i < nAR; ++i)
	{
		int nAdiacentFaces = i_AR_AF[i+1] - i_AR_AF[i];
		for(end = a_Z + i_Z[i+1]; it != end; ++it, ++j_it)
		{
			elag_assert(fabs(*it) < static_cast<double>(nAdiacentFaces)+tol);
			if( fabs(*it) > static_cast<double>(nAdiacentFaces) - tol )
			{
				j_AR_r[nnz] = *j_it;
				o_AR_r[nnz] = (*it > 0.) ? 1.0:-1.0;
				++nnz;
			}
		}
	}

	return new TopologyTable(i_AR_r, j_AR_r, o_AR_r, nAR, nr);
}

TopologyTable * AgglomeratedTopology::generate_AR_r_ForUniformRefinement()
{
	TopologyTable * AR_AF = B_[1]->Transpose();
	TopologyTable * AF_r  = MultOrientation( *(finerTopology->AEntity_entity[1]), *(finerTopology->B_[1]) );

	SerialCSRMatrix * Z = Mult(*AR_AF, *AF_r);
	delete AF_r;
	TopologyTable * out = FindIntersectionsAR(AR_AF, Z);
	delete AR_AF;
	delete Z;
	return out;
}

TopologyTable * AgglomeratedTopology::generate_AP_p_ForUniformRefinement()
{
	TopologyTable * AR_AF = B_[2]->Transpose();
	TopologyTable * AF_r  = MultOrientation( *(finerTopology->AEntity_entity[2]), *(finerTopology->B_[2]) );

	SerialCSRMatrix * Z = Mult(*AR_AF, *AF_r);
	delete AF_r;

	TopologyTable * out = FindIntersectionsAR(AR_AF, Z);
	delete AR_AF;
	delete Z;

	return out;
}

AgglomeratedTopology::ExtraTopologyTables::ExtraTopologyTables(int ncodim):
			elem_elem_local(0),
			owns_elem_elem_local(1),
			Bt_(ncodim),
			owns_Bt(ncodim),
			elem_elem_global(0),
			owns_elem_elem_global(1),
			trueB_(ncodim),
			owns_trueB(ncodim),
			trueBt_(ncodim),
			owns_trueBt(ncodim)
{
	Bt_ = NULL;
	owns_Bt = 1;
	trueB_ = NULL;
	owns_trueB = 1;
	trueBt_ = NULL;
	owns_trueBt = 1;
}

AgglomeratedTopology::ExtraTopologyTables::~ExtraTopologyTables()
{
	Clean();
}

void AgglomeratedTopology::ExtraTopologyTables::Clean()
{
	if(owns_elem_elem_local)
		delete elem_elem_local;

	for(int icodim = 0; icodim < Bt_.Size(); ++icodim)
		if( owns_Bt[icodim] )
			delete Bt_[icodim];

	if(owns_elem_elem_global)
		delete elem_elem_global;

	for(int icodim = 0; icodim < trueB_.Size(); ++icodim)
		if( owns_trueB[icodim] )
			delete trueB_[icodim];

}
