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

#include "elag_amge.hpp"

#include <algorithm>
#include <numeric>
#include "../linalg/MatrixUtils.hpp"
#include "../linalg/SubMatrixExtraction.hpp"
#include <fstream>

DofHandler::DofHandler(MPI_Comm comm, int maxCodimensionBaseForDof_, int nDim_):
    maxCodimensionBaseForDof(maxCodimensionBaseForDof_),
    nDim(nDim_),
    entity_dof(maxCodimensionBaseForDof_+1),
    rDof_dof(maxCodimensionBaseForDof_+1),
    entity_rdof(maxCodimensionBaseForDof_+1),
    dofTrueDof(comm)
{
	for(int i(0); i < 4; ++i)
		finalized[i] = false;

	entity_dof = static_cast<SparseMatrix *>(NULL);
	rDof_dof = static_cast<SparseMatrix *>(NULL);
	entity_rdof = static_cast<SparseMatrix *>(NULL);
}

DofHandler::~DofHandler()
{
	for(int i(0); i < maxCodimensionBaseForDof+1; ++i)
		delete entity_dof[i];

	for(int i(0); i < maxCodimensionBaseForDof+1; ++i)
	{
		if(rDof_dof[i])
		{
			delete[] rDof_dof[i]->GetI();
			rDof_dof[i]->LoseData();
			delete rDof_dof[i];
		}
	}

	for(int i(0); i < maxCodimensionBaseForDof+1; ++i)
	{
		if(entity_rdof[i])
		{
			delete[] entity_rdof[i]->GetJ();
			delete[] entity_rdof[i]->GetData();
			entity_rdof[i]->LoseData();
			delete entity_rdof[i];
		}
	}

}

const SparseMatrix & DofHandler::GetEntityDofTable(entity type) const
{

	if(!finalized[type])
		mfem_error("DofHandler::GetEntityDofTable Please call BuildEntityDofTables first \n");

	if( type > maxCodimensionBaseForDof)
		mfem_error("DofHandler::GetEntityDofTable Not a valid entity type \n");

	return *entity_dof[type];

}

const SparseMatrix & DofHandler::GetEntityRDofTable(entity type)
{
	if(!entity_rdof[type])
	{
		int nrows = entity_dof[type]->Size();
		int ncols = entity_dof[type]->NumNonZeroElems();
		int nnz   = entity_dof[type]->NumNonZeroElems();
		int * i = entity_dof[type]->GetI();
		int * j = new int[nnz];
		double * a = new double[nnz];

		std::fill(a, a+nnz, 1.0);
		Array<int> dofs;
		for(int irow(0); irow < nrows; ++irow)
		{
			dofs.MakeRef(j+i[irow], i[irow+1] - i[irow]);
			GetrDof(type, irow, dofs);
		}

		entity_rdof[type] = new SparseMatrix(i,j,a, nrows, ncols);
		CheckMatrix(*entity_rdof[type]);
	}

	return *entity_rdof[type];
}

const SparseMatrix & DofHandler::GetrDofDofTable(entity type)
{
	if(!rDof_dof[type])
	{
		const SparseMatrix & ent_dof(GetEntityDofTable(type));
		int nnz = ent_dof.NumNonZeroElems();
		int * i_Rdof_dof = new int[nnz+1];
		for(int kk(0); kk < nnz+1; ++kk)
			i_Rdof_dof[kk] = kk;

		rDof_dof[type] = new SparseMatrix(i_Rdof_dof, ent_dof.GetJ(), ent_dof.GetData(), nnz, ent_dof.Width());
		CheckMatrix(*rDof_dof[type]);
	}
	return *rDof_dof[type];
}

void DofHandler::GetrDof(entity type, int ientity, Array<int> & dofs) const
{
	const SparseMatrix & entityDof = GetEntityDofTable(type);
	int * I = entityDof.GetI();

	int start = I[ientity];
	int stop  = I[ientity+1];
	dofs.SetSize(stop - start);
	int * d = dofs.GetData();

	for(int i(start); i < stop; ++i)
		*(d++) = i;
}

int DofHandler::GetNumberEntities(entity type) const
{
	if(finalized[type])
		return GetEntityDofTable(type).Size();
	else
		return getNumberOf( static_cast<int>(type) );
}

int DofHandler::GetMaxCodimensionBaseForDof() const
{
	return maxCodimensionBaseForDof;
}

int DofHandler::GetNumberInteriorDofs(entity type)
{
	int n(0);
	Array<int> dofs;

	for(int ientity(0); ientity < getNumberOf(type); ++ientity)
	{
		GetInteriorDofs(type, ientity, dofs);
		n += dofs.Size();
	}

	return n;

}

void DofHandler::GetDofs(entity type, int ientity, Array<int> & dofs)
{
	Vector discard;
	entity_dof[type]->GetRow(ientity, dofs, discard);
}


SparseMatrix * DofHandler::AllocGraphElementMass(entity type)
{
	const SparseMatrix entity_rdof = GetEntityRDofTable(type);
	SparseMatrix * rdof_entity = Transpose(const_cast<SparseMatrix&>(entity_rdof) );
	SparseMatrix * G = Mult( *rdof_entity, const_cast<SparseMatrix&>(entity_rdof) );
	delete rdof_entity;
	return G;
}

void DofHandler::Average(entity entity_type, const MultiVector & repV, MultiVector & globV)
{
	const SparseMatrix & rDof_Dof = GetrDofDofTable(entity_type);
	Vector eta( rDof_Dof.Width() );
	eta = 0.0;
	const int * J = rDof_Dof.GetJ();
	int nnz = rDof_Dof.NumNonZeroElems();
	const int * Jend = J+nnz;

	for( ; J < Jend; ++J)
		++eta(*J);

	MatrixTTimesMultiVector(rDof_Dof, repV, globV);
	globV.InverseScale(eta);
}

void DofHandler::AssembleGlobalMultiVector(entity type, const MultiVector & local, MultiVector & global)
{
	Array<int> dofs;
	entity itype(type);
	for(int i(type); i < maxCodimensionBaseForDof; ++i)
	{
		itype = static_cast<entity>(i);
		for(int ientity(0); ientity < GetNumberEntities(itype); ++ientity)
		{
			GetInteriorDofs(itype, ientity,dofs);
			global.AddSubMultiVector(dofs, local);
		}
	}
}

void DofHandler::AssembleGlobalMultiVectorFromInteriorDofs(entity type, const MultiVector & local, MultiVector & global)
{
	Array<int> dofs;
	Vector local_view;
	double * data = local.GetData();
	for(int ientity(0); ientity < GetNumberEntities(type); ++ientity)
	{
		GetInteriorDofs(type, ientity,dofs);
		local_view.SetDataAndSize( data, dofs.Size() );
		global.AddSubMultiVector(dofs, local_view);
		data += dofs.Size();
	}
}

void DofHandler::CheckInvariants() const
{
	for(int codim = maxCodimensionBaseForDof; codim >= 0; --codim)
		if(!finalized[codim])
			mfem_error("DofHandler::CheckInvariants. DofHandler not finalized");

	for(int codim = maxCodimensionBaseForDof; codim >= 0; --codim)
	{
		if(entity_dof[codim]->Size() != getNumberOf(codim))
			mfem_error("DofHandler::CheckInvariants. Not Matching sizes entity_dof \n");

		if(entity_rdof[codim]->Size() != getNumberOf(codim))
			mfem_error("DofHandler::CheckInvariants. Not Matching sizes entity_rdof \n");
	}

	checkMyInvariants();
}

// RAP
SparseMatrix * Assemble(DofHandler::entity entity_type, SparseMatrix & M_e, DofHandler & range, DofHandler & domain)
{
	range.GetrDofDofTable(entity_type);
	domain.GetrDofDofTable(entity_type);

	SparseMatrix * range_dof_rdof = Transpose( *range.rDof_dof[entity_type] );
	SparseMatrix * rt_Me = Mult(*range_dof_rdof, M_e);
	SparseMatrix * domain_rdof_dof = domain.rDof_dof[entity_type];
	SparseMatrix * rt_Me_r = Mult(*rt_Me, *domain_rdof_dof);

	delete rt_Me;
	delete range_dof_rdof;
	return rt_Me_r;
}

//RAP other way around
SparseMatrix * Distribute(DofHandler::entity entity_type, SparseMatrix & M_g, DofHandler & range, DofHandler & domain)
{
	mfem_error("this method shold not be called!");
	SparseMatrix * range_rdof_dof = range.rDof_dof[entity_type];
	SparseMatrix * r_Mg = Mult(*range_rdof_dof, M_g);
	SparseMatrix * domain_dof_rdof = Transpose(*domain.rDof_dof[entity_type]);
	SparseMatrix * r_Mg_rt = Mult(*r_Mg, *domain_dof_rdof);

	delete r_Mg;
	delete domain_dof_rdof;
	return r_Mg_rt;
}



//--------------------------------------------------------------------------------------
DofHandlerFE::DofHandlerFE(MPI_Comm comm, FiniteElementSpace * fespace_, int maxCodimensionBaseForDof):
	DofHandler(comm, maxCodimensionBaseForDof, fespace_->GetMesh()->Dimension()),
	fespace(fespace_)
{
	ParMesh * pmesh = dynamic_cast<ParMesh *>(fespace->GetMesh());
	elag_assert(pmesh);
	ParFiniteElementSpace * pfes = new ParFiniteElementSpace(pmesh, fespace->FEColl());
	dofTrueDof.SetUp(pfes, 1);
	delete pfes;
}

int DofHandlerFE::MarkDofsOnSelectedBndr(const Array<int> & bndrAttributesMarker, Array<int> & dofMarker) const
{
	dofMarker = 0;
	fespace->GetEssentialVDofs(bndrAttributesMarker, dofMarker);

	int nmarked(0);
	int * end = dofMarker.GetData() + dofMarker.Size();

	for(int * it = dofMarker.GetData(); it != end; ++it)
		if(*it)
		{
			*it = 1;
			++nmarked;
		}

	return nmarked;
}

void DofHandlerFE::GetInteriorDofs(entity type, int ientity, Array<int> & dofs)
{
	if(type == AgglomeratedTopology::ELEMENT)
	{
		fespace->GetElementInteriorDofs(ientity, dofs);
		return;
	}
	else
		mfem_error("NIY: DofHandlerFE::GetInteriorDofs(entity type, int ientity, Array<int> & dofs)");

}


DofHandlerFE::~DofHandlerFE()
{

}

void DofHandlerFE::BuildEntityDofTables()
{
	//element_dof facet_dof and ridge_dof peak_dof
	for(int entity_type=0; entity_type < maxCodimensionBaseForDof+1; ++entity_type)
		BuildEntityDofTable(static_cast<entity>(entity_type) );
}

void DofHandlerFE::BuildEntityDofTable(entity entity_type)
{
	if(finalized[entity_type])
		return;

//		std::cout<<"entity_dof for entity " << entity_type << std::endl;
	int nEntities = getNumberOf(entity_type);
	int * i = new int[nEntities+1]; i[0] = 0;

	int * i_it = i+1;
	int nnz(0);
	for(int ientity(0); ientity < nEntities; ++ientity, ++i_it)
	{
		nnz += getNumberOfDofForEntity(entity_type, ientity);
		*i_it = nnz;
	}

	int * j = new int[nnz];
	double * val = new double[nnz];

	int * j_it = j;
	double * val_it = val;
	int offset(0);

	for(int ientity(0); ientity < nEntities; ++ientity)
	{
		offset = i[ientity];
		getDofForEntity(static_cast<entity>(entity_type), ientity, j_it + offset, val_it + offset);
	}

	entity_dof[entity_type] = new SparseMatrix(i,j, val, nEntities, fespace->GetNDofs() );
	CheckMatrix(*entity_dof[entity_type]);

	finalized[entity_type] = true;

}

void DofHandlerFE::getElementDof(int entity_id, int * dofs, double * orientation) const
{
	Array<int> sdofs;
	fespace->GetElementDofs(entity_id, sdofs);

	int ind;
	for(int i(0); i < sdofs.Size(); ++i)
	{
		ind = sdofs[i];
		if(ind < 0)
		{
			dofs[i] = -1-ind;
			orientation[i] = -1.;
		}
		else
		{
			dofs[i] = ind;
			orientation[i] = 1.;
		}
	}

}

void DofHandlerFE::getFacetDof(int entity_id, int * dofs, double * orientation) const
{
	Array<int> sdofs;
	switch(nDim)
	{
	case 1:
		mfem_error("NOT SUPPORTED \n");
		break;
	case 2:
		fespace->GetEdgeDofs(entity_id, sdofs);
		break;
	case 3:
		fespace->GetFaceDofs(entity_id, sdofs);
		break;
	default:
		mfem_error("NOT SUPPORTED \n");
	}

	int ind;
	for(int i(0); i < sdofs.Size(); ++i)
	{
		ind = sdofs[i];
		if(ind < 0)
		{
			dofs[i] = -1-ind;
			orientation[i] = -1.;
		}
		else
		{
			dofs[i] = ind;
			orientation[i] = 1.;
		}
	}

}

void DofHandlerFE::getRidgeDof(int entity_id, int * dofs, double * orientation) const
{
	Array<int> sdofs;
	switch(nDim)
	{
	case 1:
		mfem_error("NOT SUPPORTED \n");
		break;
	case 2:
		fespace->GetVertexDofs(entity_id, sdofs);
		break;
	case 3:
		fespace->GetEdgeDofs(entity_id, sdofs);
		break;
	default:
		mfem_error("NOT SUPPORTED \n");
	}

	int ind;
	for(int i(0); i < sdofs.Size(); ++i)
	{
		ind = sdofs[i];
		if(ind < 0)
		{
			dofs[i] = -1-ind;
			orientation[i] = -1.;
		}
		else
		{
			dofs[i] = ind;
			orientation[i] = 1.;
		}
	}
}

void DofHandlerFE::getPeakDof(int entity_id, int * dofs, double * orientation) const
{
	Array<int> sdofs;
	if(nDim < 3)
		mfem_error("NOT SUPPORTED \n");
	else
		fespace->GetVertexDofs(entity_id, sdofs);
	int ind;
	for(int i(0); i < sdofs.Size(); ++i)
	{
		ind = sdofs[i];
		if(ind < 0)
		{
			dofs[i] = -1-ind;
			orientation[i] = -1.;
		}
		else
		{
			dofs[i] = ind;
			orientation[i] = 1.;
		}
	}
}


void DofHandlerFE::getDofForEntity(entity type, int entity_id, int * dofs, double * orientation) const
{
	switch(type)
	{
	case AgglomeratedTopology::ELEMENT:
		getElementDof(entity_id, dofs, orientation);
		break;
	case AgglomeratedTopology::FACET:
		getFacetDof(entity_id, dofs, orientation);
		break;
	case AgglomeratedTopology::RIDGE:
		getRidgeDof(entity_id, dofs, orientation);
		break;
	case AgglomeratedTopology::PEAK:
		getPeakDof(entity_id, dofs, orientation);
		break;
	default:
		mfem_error("Wrong entity type \n");
	}
}

int DofHandlerFE::getNumberOfElements() const
{
	return fespace->GetMesh()->GetNE();
}

int DofHandlerFE::getNumberOfFacets() const
{
	switch(nDim)
	{
	case 1:
		return fespace->GetMesh()->GetNV();
	case 2:
		return fespace->GetMesh()->GetNEdges();
	case 3:
		return fespace->GetMesh()->GetNFaces();
	default:
		return -1;
	}
}
int DofHandlerFE::getNumberOfRidges() const
{
	switch(nDim)
	{
	case 1:
		return -1;
	case 2:
		return fespace->GetMesh()->GetNV();
		break;
	case 3:
		return fespace->GetMesh()->GetNEdges();
		break;
	default:
		return -1;
	}
}

int DofHandlerFE::getNumberOfPeaks() const
{
	if(nDim == 3)
		return fespace->GetMesh()->GetNV();
	else
		return -1;
}

int DofHandlerFE::getNumberOf(int type) const
{
	switch(type)
	{
	case AgglomeratedTopology::ELEMENT:
		return getNumberOfElements();
	case AgglomeratedTopology::FACET:
		return getNumberOfFacets();
	case AgglomeratedTopology::RIDGE:
		return getNumberOfRidges();
	case AgglomeratedTopology::PEAK:
		return getNumberOfPeaks();
	default:
		return -1;
	}
}

void DofHandlerFE::checkMyInvariants() const
{

}

int DofHandlerFE::getNumberOfDofForElement(int entity_id)
{
	return fespace->FEColl()->FiniteElementForGeometry( fespace->GetMesh()->GetElementBaseGeometry(entity_id) )->GetDof();
}

int DofHandlerFE::getNumberOfDofForFacet(int entity_id)
{
	switch(nDim)
	{
	case 1:
		return fespace->FEColl()->FiniteElementForGeometry(Geometry::POINT)->GetDof();
	case 2:
		return fespace->FEColl()->FiniteElementForGeometry(Geometry::SEGMENT)->GetDof();
	case 3:
		return fespace->FEColl()->FiniteElementForGeometry( fespace->GetMesh()->GetFaceBaseGeometry(entity_id) )->GetDof();
	default:
		mfem_error("WRONG DIMENSION");
		return -1;
	}
}
int DofHandlerFE::getNumberOfDofForRidge(int /*entity_id*/)
{
	switch(nDim)
	{
	case 1:
		mfem_error("WRONG DIMENSION");
		return -1;
	case 2:
		return fespace->FEColl()->FiniteElementForGeometry(Geometry::POINT)->GetDof();
	case 3:
		return fespace->FEColl()->FiniteElementForGeometry(Geometry::SEGMENT)->GetDof();
	default:
		mfem_error("WRONG DIMENSION");
		return -1;
	}
}

int DofHandlerFE::getNumberOfDofForPeak(int /*entity_id*/)
{
	if(nDim == 3)
		return fespace->FEColl()->FiniteElementForGeometry(Geometry::POINT)->GetDof();
	else
	{
		mfem_error("WRONG DIMENSION");
		return -1;
	}
}

int DofHandlerFE::getNumberOfDofForEntity(int entity_type, int entity_id)
{
	switch(entity_type)
	{
	case AgglomeratedTopology::ELEMENT:
		return getNumberOfDofForElement(entity_id);
	case AgglomeratedTopology::FACET:
		return getNumberOfDofForFacet(entity_id);
	case AgglomeratedTopology::RIDGE:
		return getNumberOfDofForRidge(entity_id);
	case AgglomeratedTopology::PEAK:
		return getNumberOfDofForPeak(entity_id);
	default:
		return -1;
	}
}



//============================================================================

DofHandlerALG::DofHandlerALG(int maxCodimensionBaseForDof, const AgglomeratedTopology * topo_):
					DofHandler(topo_->GetComm(), maxCodimensionBaseForDof, topo_->Dimensions()),
					topo(*topo_),
					nDofs(0),
					DofType(0)
{


	int codim(0);
	for( ; codim < maxCodimensionBaseForDof; ++codim)
	{
		entity_hasInteriorDofs[codim] = NullSpace | RangeTSpace;
		entity_NumberOfInteriorDofsRangeTSpace[codim]  = new Array<int>(topo.GetNumberLocalEntities(static_cast<entity>(codim) ) );
		entity_NumberOfInteriorDofsNullSpace[codim] = new Array<int>(topo.GetNumberLocalEntities(static_cast<entity>(codim) ) );
		entity_InteriorDofOffsets[codim] =  new Array<int>(topo.GetNumberLocalEntities(static_cast<entity>(codim) ) + 1);
	}

	if(codim == topo.Dimensions() )
	{
		entity_hasInteriorDofs[codim] = RangeTSpace;
		entity_NumberOfInteriorDofsRangeTSpace[codim] = new Array<int>(topo.GetNumberLocalEntities(static_cast<entity>(codim) ) );
		entity_NumberOfInteriorDofsNullSpace[codim] = NULL;
		entity_InteriorDofOffsets[codim] =  new Array<int>(topo.GetNumberLocalEntities(static_cast<entity>(codim) ) + 1);
	}
	else
	{
		entity_hasInteriorDofs[codim] = NullSpace | RangeTSpace;
		entity_NumberOfInteriorDofsNullSpace[codim] = new Array<int>(topo.GetNumberLocalEntities(static_cast<entity>(codim) ) );
		entity_NumberOfInteriorDofsRangeTSpace[codim]  = new Array<int>(topo.GetNumberLocalEntities(static_cast<entity>(codim) ) );
		entity_InteriorDofOffsets[codim] =  new Array<int>(topo.GetNumberLocalEntities(static_cast<entity>(codim) ) + 1);
	}

	for( codim = maxCodimensionBaseForDof+1; codim < 4; ++codim)
	{
		entity_hasInteriorDofs[codim] = Empty;
		entity_NumberOfInteriorDofsNullSpace[codim] = NULL;
		entity_NumberOfInteriorDofsRangeTSpace[codim]  = NULL;
		entity_InteriorDofOffsets[codim] =  NULL;
	}

	std::fill(entityType_nDofs, entityType_nDofs+4, 0);

}

DofHandlerALG::DofHandlerALG(int * entity_HasInteriorDofs_,  int maxCodimensionBaseForDof, const AgglomeratedTopology & topo_):
	DofHandler(topo_.GetComm(), maxCodimensionBaseForDof, topo_.Dimensions()),
	topo(topo_),
	nDofs(0),
	DofType(0)
{

	if( !(entity_HasInteriorDofs_[maxCodimensionBaseForDof] & RangeTSpace) )
		mfem_error("entity_HasInteriorDofs_[maxCodimensionBaseForDof] must contain at least the Pasciak Vassilevski Space");

	for(int codim(0); codim < 4; ++codim)
	{
		entity_hasInteriorDofs[codim] = entity_HasInteriorDofs_[codim];

		if(entity_hasInteriorDofs[codim] & NullSpace)
			entity_NumberOfInteriorDofsNullSpace[codim] = new Array<int>(topo.GetNumberLocalEntities(static_cast<entity>(codim) ) );
		else
			entity_NumberOfInteriorDofsNullSpace[codim] = NULL;

		if(entity_hasInteriorDofs[codim] & RangeTSpace)
			entity_NumberOfInteriorDofsRangeTSpace[codim]  = new Array<int>(topo.GetNumberLocalEntities(static_cast<entity>(codim) ) );
		else
			mfem_error("entity_HasInteriorDofs_[codim] must always be true");

		entity_InteriorDofOffsets[codim] =  new Array<int>(topo.GetNumberLocalEntities(static_cast<entity>(codim) ) + 1);
	}

	std::fill(entityType_nDofs, entityType_nDofs+4, 0);

}

DofHandlerALG::~DofHandlerALG()
{
	for(int codim(0); codim < 4; ++codim)
	{
		delete entity_NumberOfInteriorDofsNullSpace[codim];
		delete entity_NumberOfInteriorDofsRangeTSpace[codim];
		delete entity_InteriorDofOffsets[codim];
	}
}

int DofHandlerALG::MarkDofsOnSelectedBndr(const Array<int> & bndrAttributesMarker, Array<int> & dofMarker) const
{

	if(dofMarker.Size() != GetNDofs() )
		mfem_error("int DofHandlerALG::MarkDofsOnSelectedBndr #1");

	dofMarker = 0;

	const TopologyTable & fc_bdnr( topo.FacetBdrAttribute() );

	int n_fc = fc_bdnr.Size();
	int * i_fc_bndr = fc_bdnr.GetI();
	int * j_fc_bndr = fc_bdnr.GetJ();

	int * i_facet_dof = entity_dof[AgglomeratedTopology::FACET]->GetI();
	int * j_facet_dof = entity_dof[AgglomeratedTopology::FACET]->GetJ();


	int start(0), end(0);
	for(int ifc = 0; ifc < n_fc; ++ifc)
	{
		end = i_fc_bndr[ifc+1];
		elag_assert( ( (end-start) == 0) || ( (end-start)  == 1) );
		if( (end-start) == 1 && bndrAttributesMarker[j_fc_bndr[start] ])
			for(int * it = j_facet_dof + i_facet_dof[ifc]; it != j_facet_dof + i_facet_dof[ifc+1]; ++it)
				dofMarker[*it] = 1;

		start = end;
	}

	int nMarked(0);

	for(int * it = dofMarker.GetData(); it != dofMarker.GetData()+ dofMarker.Size(); ++it)
		if(*it)
			++nMarked;

	return nMarked;
}

int DofHandlerALG::GetNumberInteriorDofs(entity type)
{

	if(type == maxCodimensionBaseForDof)
		return entity_InteriorDofOffsets[type]->Last();
	else
		return entity_InteriorDofOffsets[type]->Last() - entity_InteriorDofOffsets[type+1]->Last();
}

int DofHandlerALG::GetNumberInteriorDofs(entity type, int entity_id)
{
	return entity_InteriorDofOffsets[type]->operator[](entity_id+1) - entity_InteriorDofOffsets[type]->operator[](entity_id);
}

void DofHandlerALG::BuildEntityDofTables()
{
	buildPeakDofTable();
	buildRidgeDofTable();
	buildFacetDofTable();
	buildElementDofTable();
}

void DofHandlerALG::BuildEntityDofTable(entity type)
{
	switch(type)
	{
	case AgglomeratedTopology::PEAK:
		buildPeakDofTable();
		break;
	case AgglomeratedTopology::RIDGE:
		buildRidgeDofTable();
		break;
	case AgglomeratedTopology::FACET:
		buildFacetDofTable();
		break;
	case AgglomeratedTopology::ELEMENT:
		buildElementDofTable();
		break;
	default:
		mfem_error("void DofHandlerALG::BuildEntityDofTable(entity type)");
	}
}


void DofHandlerALG::AllocateDofTypeArray(int maxSize)
{
	DofType.SetSize(maxSize, Empty);
}

void DofHandlerALG::SetDofType(int dof, dof_type type)
{
	if(DofType[dof] != Empty)
	{
		std::cout<< "DofType[" << dof <<"] is already set to the value " << DofType[dof] << "\n";
		mfem_error("You can't change the type of Dof \n");
	}
	elag_assert( type != Empty );
	DofType[dof] = type;
}


SparseMatrix * DofHandlerALG::GetEntityNullSpaceDofTable(entity type) const
{
	elag_assert( DofType.Find(Empty) == -1 || DofType.Find(Empty) >= nDofs );
	elag_assert( DofType.Size() >= nDofs);

	int colStart(0);
	if(type < maxCodimensionBaseForDof)
		colStart = entityType_nDofs[type+1];

	DropEntryAccordingToColumnMarkerAndId drop(DofType, NullSpace, colStart);
	return DropEntriesFromSparseMatrix(*entity_dof[type], drop);
}


SparseMatrix * DofHandlerALG::GetEntityRangeTSpaceDofTable(entity type) const
{
	elag_assert( DofType.Find(Empty) == -1 || DofType.Find(Empty) >= nDofs );
	elag_assert( DofType.Size() >= nDofs);

	int colStart(0);
	if(type < maxCodimensionBaseForDof)
		colStart = entityType_nDofs[type+1];

	DropEntryAccordingToColumnMarkerAndId drop(DofType, RangeTSpace, colStart);
	return DropEntriesFromSparseMatrix(*entity_dof[type], drop);
}

SparseMatrix * DofHandlerALG::GetEntityInternalDofTable(entity type) const
{
	elag_assert(finalized[type]);


	int colStart(0);
	if(type < maxCodimensionBaseForDof)
		colStart = entityType_nDofs[type+1];

	DropEntryAccordingToId drop(colStart);
	return DropEntriesFromSparseMatrix(*entity_dof[type], drop);
}

void DofHandlerALG::SetNumberOfInteriorDofsNullSpace(entity type, int entity_id, int nLocDof)
{
	if( entity_hasInteriorDofs[type] & NullSpace )
		 (*entity_NumberOfInteriorDofsNullSpace[type])[entity_id] = nLocDof;
	else
		mfem_error("DofHandlerALG::SetNumberOfInteriorDofsNullSpace");
}

void DofHandlerALG::SetNumberOfInteriorDofsRangeTSpace(entity type, int entity_id, int nLocDof)
{
	if( entity_hasInteriorDofs[type] & RangeTSpace )
		 (*entity_NumberOfInteriorDofsRangeTSpace[type])[entity_id] = nLocDof;
	else
		mfem_error("DofHandlerALG::SetNumberOfInteriorDofsRangeTSpace");
}

void DofHandlerALG::GetInteriorDofs(entity type, int ientity, Array<int> & dofs)
{
	Vector dummy;
	if(type == static_cast<entity>(maxCodimensionBaseForDof))	// all dofs are internal
		entity_dof[type]->GetRow(ientity, dofs, dummy);
	else
	{
		Array<int> allDofs;
		int startIndex = entityType_nDofs[type+1];
		entity_dof[type]->GetRow(ientity, allDofs, dummy);
//		if( !dofs.IsSorted() )
//			mfem_error("entity_dof[type]->GetRow(ientity, allDofs) is not sorted :(");

		int size(allDofs.Size());
		dofs.SetSize(size);
		dofs.SetSize(0);

		int * data = allDofs.GetData();
		for(int * end = data+size; data != end; ++data )
			if( *data >= startIndex)
				dofs.Append(*data);
	}
}


void DofHandlerALG::GetDofsOnBdr(entity type, int ientity, Array<int> & dofs)
{
	int ndofs(0);
	Array<int> bdr_entity;
	Array<int> ibdr_entity_dofs;

	for(int i(maxCodimensionBaseForDof); i > type ;--i)
	{
		entity type_bdr = static_cast<entity>(i);
		topo.GetBoundaryOfEntity(type, type_bdr, ientity, bdr_entity);

		for(int * ibdr_entity = bdr_entity.GetData(); ibdr_entity != bdr_entity.GetData()+bdr_entity.Size(); ++ibdr_entity )
			ndofs += GetNumberInteriorDofs(type_bdr, *ibdr_entity);

	}

	dofs.SetSize(ndofs);
	int * it = dofs.GetData();

	for(int i(maxCodimensionBaseForDof); i > type; --i)
	{
		entity type_bdr = static_cast<entity>(i);
		topo.GetBoundaryOfEntity(type, type_bdr, ientity, bdr_entity);

		for(int * ibdr_entity = bdr_entity.GetData(); ibdr_entity != bdr_entity.GetData()+bdr_entity.Size(); ++ibdr_entity )
		{
			GetInteriorDofs(type_bdr, *ibdr_entity, ibdr_entity_dofs);
			it = std::copy(ibdr_entity_dofs.GetData(), ibdr_entity_dofs.GetData()+ibdr_entity_dofs.Size(), it);
		}

	}
}


int DofHandlerALG::getNumberOf(int type) const
{
	return topo.GetNumberLocalEntities(static_cast<entity>(type));
}

void DofHandlerALG::computeOffset(entity type)
{
	if(finalized[type])
		return;

	int * e_offset, * e_offsetEnd;
	int * nullSpace_ndof, * rangeTSpace_ndof;
	int size(-1);

	size = topo.GetNumberLocalEntities(type);

#ifdef ELAG_DEBUG
	if( type > maxCodimensionBaseForDof)
		mfem_error("Invalid type \n");

	if( type ==  maxCodimensionBaseForDof)
	{
		if(nDofs != 0 || entityType_nDofs[maxCodimensionBaseForDof] != 0)
			mfem_error("Something strange is happening. nDofs should be 0");
	}
	else
	{
		if(nDofs < 0 )
			mfem_error("Invalid nDofs");

		if( !finalized[type+1] )
			mfem_error("Before calling computedOffset(codim) you need to call BuildEntityDofTable(codim+1) \n");

		if( nDofs != entityType_nDofs[type+1] )
			mfem_error("Something strange is happening. nDofs and entityType_nDofs do not agree");
	}

	if( entity_hasInteriorDofs[type] & RangeTSpace )
	{
		int * it = entity_NumberOfInteriorDofsRangeTSpace[type]->GetData();
		int * end = it + entity_NumberOfInteriorDofsRangeTSpace[type]->Size();
		for( ; it != end; ++it)
		{
			if(*it < 0)
				mfem_error("negative number! ");

			if(*it > 100)
				mfem_error("so many dofs on a coarse entity are impossible!");
		}
	}

	if( entity_hasInteriorDofs[type] & NullSpace )
	{
		int * it = entity_NumberOfInteriorDofsNullSpace[type]->GetData();
		int * end = it + entity_NumberOfInteriorDofsNullSpace[type]->Size();
		for( ; it != end; ++it)
		{
			if(*it < 0)
				mfem_error("negative number! ");

			if(*it > 100)
				mfem_error("so many dofs on a coarse entity are impossible!");
		}
	}

#endif

	switch(entity_hasInteriorDofs[type])
	{
	case Empty:
		break;

	case RangeTSpace:
		e_offset = entity_InteriorDofOffsets[type]->GetData();
		rangeTSpace_ndof = entity_NumberOfInteriorDofsRangeTSpace[type]->GetData();
		for(e_offsetEnd = e_offset + size; e_offset != e_offsetEnd; ++e_offset, ++rangeTSpace_ndof)
		{
			*e_offset = nDofs;
			nDofs += *rangeTSpace_ndof;
		}
		*e_offset = nDofs;
		break;

	case RangeTSpace | NullSpace:
		e_offset = entity_InteriorDofOffsets[type]->GetData();
		nullSpace_ndof = entity_NumberOfInteriorDofsNullSpace[type]->GetData();
		rangeTSpace_ndof = entity_NumberOfInteriorDofsRangeTSpace[type]->GetData();
		for(e_offsetEnd = e_offset + size; e_offset != e_offsetEnd; ++e_offset, ++nullSpace_ndof, ++rangeTSpace_ndof)
		{
			*e_offset = nDofs;
			nDofs += *nullSpace_ndof+*rangeTSpace_ndof;
		}
		*e_offset = nDofs;
		break;

	default:
		mfem_error("Invalid entity_hasInteriorDofs[type]");
	}

	entityType_nDofs[type] = nDofs;
}

void DofHandlerALG::buildPeakDofTable()
{

	if(AgglomeratedTopology::PEAK > maxCodimensionBaseForDof)
		return;

	computeOffset(AgglomeratedTopology::PEAK);

	int nrow = topo.GetNumberLocalEntities(AgglomeratedTopology::PEAK);
//	int ncol = entity_InteriorDofOffsets[Topology::PEAK]->GetData()[nrow];
	int ncol = entityType_nDofs[AgglomeratedTopology::PEAK];

	int * i = new int[nrow+1];
	int * j = new int[ncol];
	double * val = new double[ncol];

	std::copy(entity_InteriorDofOffsets[AgglomeratedTopology::PEAK]->GetData(), entity_InteriorDofOffsets[AgglomeratedTopology::PEAK]->GetData()+nrow+1, i);
	for(int k(0), * it = j; k < ncol; ++k, ++it)
		*it = k;
	std::fill(val, val+ncol, 1.);

	entity_dof[AgglomeratedTopology::PEAK] = new SparseMatrix(i, j, val, nrow, ncol);
	CheckMatrix(*entity_dof[AgglomeratedTopology::PEAK]);
	finalized[AgglomeratedTopology::PEAK] = true;
}


void DofHandlerALG::buildRidgeDofTable()
{

	if(AgglomeratedTopology::RIDGE > maxCodimensionBaseForDof)
		return;

	computeOffset(AgglomeratedTopology::RIDGE);

	int nrow = topo.GetNumberLocalEntities(AgglomeratedTopology::RIDGE);
//	int ncol = entity_InteriorDofOffsets[entity_hasInteriorDofs[Topology::RIDGE] ? Topology::RIDGE : Topology::PEAK]->Last();
	int ncol = entityType_nDofs[AgglomeratedTopology::RIDGE];

	int * i = new int[nrow+2];
	std::fill(i, i+nrow+2, 0);
	int * i_assembly = i+2;

	if(entity_hasInteriorDofs[AgglomeratedTopology::PEAK])
	{
		const BooleanMatrix & ridge_peak = topo.GetConnectivity(static_cast<int>(AgglomeratedTopology::RIDGE), static_cast<int>(AgglomeratedTopology::PEAK) );
		const int * i_ridge_peak = ridge_peak.GetI();
		const int * j_ridge_peak = ridge_peak.GetJ();
		int * p_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::PEAK]->GetData();

		for(int iridge(0); iridge < nrow; ++iridge)
			for(const int * j_peak = j_ridge_peak+i_ridge_peak[iridge]; j_peak != j_ridge_peak+i_ridge_peak[iridge+1]; ++ j_peak)
				i_assembly[iridge] += p_offsets[*j_peak+1] - p_offsets[*j_peak];
	}

	if(entity_hasInteriorDofs[AgglomeratedTopology::RIDGE])
	{
		int * r_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::RIDGE]->GetData();
		for(int iridge(0); iridge < nrow; ++iridge)
			i_assembly[iridge] += r_offsets[iridge+1] - r_offsets[iridge];
	}

	std::partial_sum(i_assembly, i_assembly+nrow, i_assembly);

	//Shift back
	i_assembly = i+1;
	int nnz = i_assembly[nrow];
	int * j = new int[nnz];
	double * val = new double[nnz];

	std::fill(val, val + nnz, 1.);

	if(entity_hasInteriorDofs[AgglomeratedTopology::PEAK])
	{
		const BooleanMatrix & ridge_peak = topo.GetConnectivity(static_cast<int>(AgglomeratedTopology::RIDGE), static_cast<int>(AgglomeratedTopology::PEAK) );
		const int * i_ridge_peak = ridge_peak.GetI();
		const int * j_ridge_peak = ridge_peak.GetJ();
		int * p_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::PEAK]->GetData();

		for(int iridge(0); iridge < nrow; ++iridge)
			for(const int * j_peak = j_ridge_peak+i_ridge_peak[iridge]; j_peak != j_ridge_peak+i_ridge_peak[iridge+1]; ++ j_peak)
			{
				int * j_it = j+i_assembly[iridge];
				for(int jdof = p_offsets[*j_peak]; jdof < p_offsets[*j_peak+1]; ++jdof, ++j_it)
					*j_it = jdof;
				i_assembly[iridge] += p_offsets[*j_peak+1] - p_offsets[*j_peak];
			}
	}

	if(entity_hasInteriorDofs[AgglomeratedTopology::RIDGE])
	{
		int * r_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::RIDGE]->GetData();
		for(int iridge(0); iridge < nrow; ++iridge)
		{
			int * j_it = j+i_assembly[iridge];
			for(int jdof = r_offsets[iridge]; jdof < r_offsets[iridge+1]; ++jdof, ++j_it)
				*j_it = jdof;
			i_assembly[iridge] += r_offsets[iridge+1] - r_offsets[iridge];
		}
	}


	entity_dof[AgglomeratedTopology::RIDGE] = new SparseMatrix(i, j, val, nrow, ncol);
	CheckMatrix(*entity_dof[AgglomeratedTopology::RIDGE]);
	finalized[AgglomeratedTopology::RIDGE] = true;

}

void DofHandlerALG::buildFacetDofTable()
{

	if(AgglomeratedTopology::FACET > maxCodimensionBaseForDof)
		return;

	computeOffset(AgglomeratedTopology::FACET);

	int nrow = topo.GetNumberLocalEntities(AgglomeratedTopology::FACET);
	int ncol = entityType_nDofs[AgglomeratedTopology::FACET];

/*
	if(entity_hasInteriorDofs[Topology::FACET])
		ncol = entity_InteriorDofOffsets[Topology::FACET]->GetData()[nrow];
	else
		ncol = entity_InteriorDofOffsets[entity_hasInteriorDofs[Topology::RIDGE] ? Topology::RIDGE : Topology::PEAK]->Last();
*/

	int * i = new int[nrow+2];
	std::fill(i, i+nrow+2, 0);
	int * i_assembly = i+2;

	if(entity_hasInteriorDofs[AgglomeratedTopology::PEAK])
	{
		const BooleanMatrix & facet_peak = topo.GetConnectivity(static_cast<int>(AgglomeratedTopology::FACET), static_cast<int>(AgglomeratedTopology::PEAK) );
		const int * i_facet_peak = facet_peak.GetI();
		const int * j_facet_peak = facet_peak.GetJ();
		int * p_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::PEAK]->GetData();

		for(int ifacet(0); ifacet < nrow; ++ifacet)
			for(const int * j_peak = j_facet_peak+i_facet_peak[ifacet]; j_peak != j_facet_peak+i_facet_peak[ifacet+1]; ++ j_peak)
				i_assembly[ifacet] += p_offsets[*j_peak+1] - p_offsets[*j_peak];
	}

	if(entity_hasInteriorDofs[AgglomeratedTopology::RIDGE])
	{
		const BooleanMatrix & facet_ridge = topo.GetConnectivity(static_cast<int>(AgglomeratedTopology::FACET), static_cast<int>(AgglomeratedTopology::RIDGE) );

		const int * i_facet_ridge = facet_ridge.GetI();
		const int * j_facet_ridge = facet_ridge.GetJ();

		int * r_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::RIDGE]->GetData();

		for(int ifacet(0); ifacet < nrow; ++ifacet)
			for(const int * j_ridge = j_facet_ridge+i_facet_ridge[ifacet]; j_ridge != j_facet_ridge+i_facet_ridge[ifacet+1]; ++ j_ridge)
				i_assembly[ifacet] += r_offsets[*j_ridge+1] - r_offsets[*j_ridge];
	}

	if(entity_hasInteriorDofs[AgglomeratedTopology::FACET])
	{
		int * f_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::FACET]->GetData();
		for(int ifacet(0); ifacet < nrow; ++ifacet)
			i_assembly[ifacet] += f_offsets[ifacet+1] - f_offsets[ifacet];
	}

	std::partial_sum(i_assembly, i_assembly+nrow, i_assembly);

	//Shift back
	i_assembly = i+1;
	int nnz = i_assembly[nrow];
	int * j = new int[nnz];
	double * val = new double[nnz];
	std::fill(val, val + nnz, 1.);

	if(entity_hasInteriorDofs[AgglomeratedTopology::PEAK])
	{
		const BooleanMatrix & facet_peak = topo.GetConnectivity(static_cast<int>(AgglomeratedTopology::FACET), static_cast<int>(AgglomeratedTopology::PEAK) );
		const int * i_facet_peak = facet_peak.GetI();
		const int * j_facet_peak = facet_peak.GetJ();
		int * p_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::PEAK]->GetData();

		for(int ifacet(0); ifacet < nrow; ++ifacet)
			for(const int * j_peak = j_facet_peak+i_facet_peak[ifacet]; j_peak != j_facet_peak+i_facet_peak[ifacet+1]; ++ j_peak)
			{
				int * j_it = j+i_assembly[ifacet];
				for(int jdof = p_offsets[*j_peak]; jdof < p_offsets[*j_peak+1]; ++jdof, ++j_it)
					*j_it = jdof;
				i_assembly[ifacet] += p_offsets[*j_peak+1] - p_offsets[*j_peak];
			}
	}

	if(entity_hasInteriorDofs[AgglomeratedTopology::RIDGE])
	{
		const BooleanMatrix & facet_ridge = topo.GetConnectivity(static_cast<int>(AgglomeratedTopology::FACET), static_cast<int>(AgglomeratedTopology::RIDGE) );
		const int * i_facet_ridge = facet_ridge.GetI();
		const int * j_facet_ridge = facet_ridge.GetJ();
		int * r_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::RIDGE]->GetData();

		for(int ifacet(0); ifacet < nrow; ++ifacet)
			for(const int * j_ridge = j_facet_ridge+i_facet_ridge[ifacet]; j_ridge != j_facet_ridge + i_facet_ridge[ifacet+1]; ++j_ridge)
			{
				int * j_it = j+i_assembly[ifacet];
				for(int jdof = r_offsets[*j_ridge]; jdof < r_offsets[*j_ridge+1]; ++jdof, ++j_it)
					*j_it = jdof;

				i_assembly[ifacet] += r_offsets[*j_ridge+1] - r_offsets[*j_ridge];
			}
	}


	if(entity_hasInteriorDofs[AgglomeratedTopology::FACET])
	{
		int * f_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::FACET]->GetData();
		for(int ifacet(0); ifacet < nrow; ++ifacet)
		{
			int * j_it = j+i_assembly[ifacet];
			for(int jdof = f_offsets[ifacet]; jdof < f_offsets[ifacet+1]; ++jdof, ++j_it)
				*j_it = jdof;
			i_assembly[ifacet] += f_offsets[ifacet+1] - f_offsets[ifacet];
		}
	}


	entity_dof[AgglomeratedTopology::FACET] = new SparseMatrix(i, j, val, nrow, ncol);
	CheckMatrix(*entity_dof[AgglomeratedTopology::FACET]);
	finalized[AgglomeratedTopology::FACET] = true;

}

void DofHandlerALG::buildElementDofTable()
{

	computeOffset(AgglomeratedTopology::ELEMENT);

	int nrow = topo.GetNumberLocalEntities(AgglomeratedTopology::ELEMENT);
	int ncol = entityType_nDofs[AgglomeratedTopology::ELEMENT];

/*
	if(entity_hasInteriorDofs[Topology::ELEMENT])
		ncol = entity_InteriorDofOffsets[Topology::ELEMENT]->GetData()[nrow];
	else if(entity_hasInteriorDofs[Topology::FACET])
		ncol = entity_InteriorDofOffsets[Topology::FACET]->Last();
	else
		ncol = entity_InteriorDofOffsets[entity_hasInteriorDofs[Topology::RIDGE] ? Topology::RIDGE : Topology::PEAK]->Last();
*/

	int * i = new int[nrow+2];
	std::fill(i, i+nrow+2, 0);
	int * i_assembly = i+2;

	if(entity_hasInteriorDofs[AgglomeratedTopology::PEAK])
	{
		const BooleanMatrix & element_peak = topo.GetConnectivity(static_cast<int>(AgglomeratedTopology::ELEMENT), static_cast<int>(AgglomeratedTopology::PEAK) );
		const int * i_element_peak = element_peak.GetI();
		const int * j_element_peak = element_peak.GetJ();
		int * p_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::PEAK]->GetData();

		for(int iel(0); iel < nrow; ++iel)
			for(const int * j_peak = j_element_peak+i_element_peak[iel]; j_peak != j_element_peak+i_element_peak[iel+1]; ++ j_peak)
				i_assembly[iel] += p_offsets[*j_peak+1] - p_offsets[*j_peak];
	}

	if(entity_hasInteriorDofs[AgglomeratedTopology::RIDGE])
	{
		const BooleanMatrix & element_ridge = topo.GetConnectivity(static_cast<int>(AgglomeratedTopology::ELEMENT), static_cast<int>(AgglomeratedTopology::RIDGE) );

		const int * i_element_ridge = element_ridge.GetI();
		const int * j_element_ridge = element_ridge.GetJ();

		int * r_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::RIDGE]->GetData();

		for(int iel(0); iel < nrow; ++iel)
			for(const int * j_ridge = j_element_ridge+i_element_ridge[iel]; j_ridge != j_element_ridge+i_element_ridge[iel+1]; ++ j_ridge)
				i_assembly[iel] += r_offsets[*j_ridge+1] - r_offsets[*j_ridge];
	}

	if(entity_hasInteriorDofs[AgglomeratedTopology::FACET])
	{
		const BooleanMatrix & element_facet = topo.GetConnectivity(static_cast<int>(AgglomeratedTopology::ELEMENT), static_cast<int>(AgglomeratedTopology::FACET) );

		const int * i_element_facet = element_facet.GetI();
		const int * j_element_facet = element_facet.GetJ();

		int * r_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::FACET]->GetData();

		for(int iel(0); iel < nrow; ++iel)
			for(const int * j_facet = j_element_facet+i_element_facet[iel]; j_facet != j_element_facet+i_element_facet[iel+1]; ++ j_facet)
				i_assembly[iel] += r_offsets[*j_facet+1] - r_offsets[*j_facet];
	}

	if(entity_hasInteriorDofs[AgglomeratedTopology::ELEMENT])
	{
		int * e_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::ELEMENT]->GetData();
		for(int iel(0); iel < nrow; ++iel)
			i_assembly[iel] += e_offsets[iel+1] - e_offsets[iel];
	}

	std::partial_sum(i_assembly, i_assembly+nrow, i_assembly);

	//Shift back
	i_assembly = i+1;
	int nnz = i_assembly[nrow];
	int * j = new int[nnz];
	double * val = new double[nnz];
	std::fill(val, val + nnz, 1.);

	if(entity_hasInteriorDofs[AgglomeratedTopology::PEAK])
	{
		const BooleanMatrix & element_peak = topo.GetConnectivity(static_cast<int>(AgglomeratedTopology::ELEMENT), static_cast<int>(AgglomeratedTopology::PEAK) );
		const int * i_element_peak = element_peak.GetI();
		const int * j_element_peak = element_peak.GetJ();
		int * p_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::PEAK]->GetData();

		for(int ielement(0); ielement < nrow; ++ielement)
			for(const int * j_peak = j_element_peak+i_element_peak[ielement]; j_peak != j_element_peak+i_element_peak[ielement+1]; ++ j_peak)
			{
				int * j_it = j+i_assembly[ielement];
				for(int jdof = p_offsets[*j_peak]; jdof < p_offsets[*j_peak+1]; ++jdof, ++j_it)
					*j_it = jdof;
				i_assembly[ielement] += p_offsets[*j_peak+1] - p_offsets[*j_peak];
			}
	}

	if(entity_hasInteriorDofs[AgglomeratedTopology::RIDGE])
	{
		const BooleanMatrix & element_ridge = topo.GetConnectivity(static_cast<int>(AgglomeratedTopology::ELEMENT), static_cast<int>(AgglomeratedTopology::RIDGE) );
		const int * i_element_ridge = element_ridge.GetI();
		const int * j_element_ridge = element_ridge.GetJ();
		int * r_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::RIDGE]->GetData();

		for(int iel(0); iel < nrow; ++iel)
			for(const int * j_ridge = j_element_ridge+i_element_ridge[iel]; j_ridge != j_element_ridge + i_element_ridge[iel+1]; ++j_ridge)
			{
				int * j_it = j+i_assembly[iel];
				for(int jdof = r_offsets[*j_ridge]; jdof < r_offsets[*j_ridge+1]; ++jdof, ++j_it)
					*j_it = jdof;
				i_assembly[iel] += r_offsets[*j_ridge+1] - r_offsets[*j_ridge];
			}
	}


	if(entity_hasInteriorDofs[AgglomeratedTopology::FACET])
	{
		const BooleanMatrix & element_facet = topo.GetConnectivity(static_cast<int>(AgglomeratedTopology::ELEMENT), static_cast<int>(AgglomeratedTopology::FACET) );
		const int * i_element_facet = element_facet.GetI();
		const int * j_element_facet = element_facet.GetJ();
		int * f_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::FACET]->GetData();

		for(int iel(0); iel < nrow; ++iel)
			for(const int * j_facet = j_element_facet+i_element_facet[iel]; j_facet != j_element_facet + i_element_facet[iel+1]; ++j_facet)
			{
				int * j_it = j+i_assembly[iel];
				for(int jdof = f_offsets[*j_facet]; jdof < f_offsets[*j_facet+1]; ++jdof, ++j_it)
					*j_it = jdof;
				i_assembly[iel] += f_offsets[*j_facet+1] - f_offsets[*j_facet];
			}
	}


	if(entity_hasInteriorDofs[AgglomeratedTopology::ELEMENT])
	{
		int * e_offsets = entity_InteriorDofOffsets[AgglomeratedTopology::ELEMENT]->GetData();
		for(int iel(0); iel < nrow; ++iel)
		{
			int * j_it = j+i_assembly[iel];
			for(int jdof = e_offsets[iel]; jdof < e_offsets[iel+1]; ++jdof, ++j_it)
				*j_it = jdof;
			i_assembly[iel] += e_offsets[iel+1] - e_offsets[iel];
		}
	}


	entity_dof[AgglomeratedTopology::ELEMENT] = new SparseMatrix(i, j, val, nrow, ncol);
	CheckMatrix(*entity_dof[AgglomeratedTopology::ELEMENT]);
	finalized[AgglomeratedTopology::ELEMENT] = true;
}

void DofHandlerALG::checkMyInvariants() const
{

}
