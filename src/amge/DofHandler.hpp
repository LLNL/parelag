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

#ifndef DOFHANDLER_HPP_
#define DOFHANDLER_HPP_

//! @class
/*!
 * @brief Handles parallel numbering, including entityTrueEntity, dofTrueDof, etc.
 *
 */
class DofHandler
{
public:

	typedef AgglomeratedTopology::EntityByCodim entity;
	enum {SCALARSPACE = 0, VECTORSPACE = 1};

	DofHandler(MPI_Comm comm, int maxCodimensionBaseForDof, int nDim);
	virtual ~DofHandler();

	int SpaceType(){ return (maxCodimensionBaseForDof == 0 || maxCodimensionBaseForDof == nDim) ? SCALARSPACE : VECTORSPACE; }

	virtual void BuildEntityDofTables() = 0;
	virtual void BuildEntityDofTable(entity type) = 0;
	// bndrAttributesMarker boolean array [input]
	// dofMarker: boolean array (size GetNDofs): selected dofs are marked with ones.
	// return how many dofs were marked.
	virtual int MarkDofsOnSelectedBndr(const Array<int> & bndrAttributesMarker, Array<int> & dofMarker) const = 0;


	const SparseMatrix & GetEntityDofTable(entity type) const;
	int GetEntityNDof(entity type, int ientity) const { return GetEntityDofTable(type).RowSize(ientity); }
	const SparseMatrix & GetEntityRDofTable(entity type);
	// NOTE this routine assumes that rdof relative to the same entity are contiguous.
	const SparseMatrix & GetrDofDofTable(entity type);
	// NOTE this routine returns an array of rdof that is contiguous.
	virtual void GetrDof(entity type, int ientity, Array<int> & dofs) const;
	virtual void GetDofs(entity type, int ientity, Array<int> & dofs);
	int GetNumberEntities(entity type) const;
	int GetMaxCodimensionBaseForDof() const;
	virtual int GetNumberInteriorDofs(entity type);
	virtual void GetInteriorDofs(entity type, int ientity, Array<int> & dofs) = 0;
	virtual void GetDofsOnBdr(entity type, int ientity, Array<int> & dofs) = 0;

	SparseMatrix * AllocGraphElementMass(entity type);

	inline int GetNDofs() const {return entity_dof[0]->Width(); }
	inline int GetNrDofs(entity entity_type) const { return entity_dof[entity_type]->NumNonZeroElems();/*rDof_dof[entity_type]->Size();*/ }

	void Average(entity entity_type, const MultiVector & repV, MultiVector & globV);
	void AssembleGlobalMultiVector(entity type, const MultiVector & local, MultiVector & global);
	void AssembleGlobalMultiVectorFromInteriorDofs(entity type, const MultiVector & local, MultiVector & global);

	friend SparseMatrix * Assemble(entity entity_type, SparseMatrix & M_e, DofHandler & range, DofHandler & domain);
	friend SparseMatrix * Distribute(entity entity_type, SparseMatrix & M_g, DofHandler & range, DofHandler & domain);

	int Finalized(){ return finalized[0]; }

	virtual const SharingMap & GetEntityTrueEntity(int codim) const = 0;
	SharingMap & GetDofTrueDof(){return dofTrueDof;}
	const SharingMap & GetDofTrueDof() const {return dofTrueDof;}


	void CheckInvariants() const;

protected:

	virtual int getNumberOf(int type) const = 0;
	virtual void checkMyInvariants() const = 0;

	int finalized[4];
    int maxCodimensionBaseForDof;
    int nDim;

	Array<SparseMatrix *> entity_dof;
	Array<SparseMatrix *> rDof_dof;
	Array<SparseMatrix *> entity_rdof;

	SharingMap dofTrueDof;
};

class DofHandlerFE : public DofHandler
{
public:

	typedef DofHandler super;
	typedef super::entity entity;

	DofHandlerFE(MPI_Comm comm, FiniteElementSpace * fespace_, int maxCodimensionBaseForDof_);
	virtual ~DofHandlerFE();

	virtual void BuildEntityDofTables();
	virtual void BuildEntityDofTable(entity type);
	virtual int MarkDofsOnSelectedBndr(const Array<int> & bndrAttributesMarker, Array<int> & dofMarker) const;
	virtual void GetInteriorDofs(entity type, int ientity, Array<int> & dofs);
	virtual void GetrDof(entity type, int ientity, Array<int> & dofs) const{ super::GetrDof(type, ientity, dofs); }
	virtual void GetDofsOnBdr(entity type, int ientity, Array<int> & dofs){ mfem_error("DofHandlerFE::GetDofOnBdr not implemented! ");}
	virtual const SharingMap & GetEntityTrueEntity(int codim) const {elag_error_msg(1,"Not Implemented"); SharingMap * m = NULL; return *m; }

protected:
	virtual int getNumberOf(int type) const;
	virtual void checkMyInvariants() const;

private:


	void getElementDof(int entity_id, int * dofs, double * orientation) const;
	void getFacetDof(int entity_id, int * dofs, double * orientation) const;
	void getRidgeDof(int entity_id, int * dofs, double * orientation) const;
	void getPeakDof(int entity_id, int * dofs, double * orientation) const;
	void getDofForEntity(entity entity_type, int entity_id, int * dofs, double * orientation) const;
 
    int getNumberOfElements() const;
	int getNumberOfFacets() const;
	int getNumberOfRidges() const;
	int getNumberOfPeaks() const;


	int getNumberOfDofForElement(int entity_id);
	int getNumberOfDofForFacet(int entity_id);
	int getNumberOfDofForRidge(int entity_id);
	int getNumberOfDofForPeak(int entity_id);
	int getNumberOfDofForEntity(int entity_type, int entity_id);


	FiniteElementSpace * fespace;
};


class DofHandlerALG : public DofHandler
{
public:

	enum dof_type{ Empty = 0x0, RangeTSpace = 0x1, NullSpace = 0x2};

	DofHandlerALG(int maxCodimensionBaseForDof, const AgglomeratedTopology * topo);
	DofHandlerALG(int * entity_HasInteriorDofs, int maxCodimensionBaseForDof, const AgglomeratedTopology & topo);
	virtual ~DofHandlerALG();


	virtual int GetNumberInteriorDofs(entity type);
	int GetNumberInteriorDofs(entity type, int entity_id);
	virtual void BuildEntityDofTables();
	virtual void BuildEntityDofTable(entity type);

	void AllocateDofTypeArray(int maxSize);
	void SetDofType(int dof, dof_type type);

	virtual int MarkDofsOnSelectedBndr(const Array<int> & bndrAttributesMarker, Array<int> & dofMarker) const;

	SparseMatrix * GetEntityNullSpaceDofTable(entity type) const;
	SparseMatrix * GetEntityRangeTSpaceDofTable(entity type) const;
	SparseMatrix * GetEntityInternalDofTable(entity type) const;

	void SetNumberOfInteriorDofsNullSpace(entity type, int entity_id, int nLocDof);
	void SetNumberOfInteriorDofsRangeTSpace(entity type, int entity_id, int nLocDof);

	virtual void GetInteriorDofs(entity type, int ientity, Array<int> & dofs);
	virtual void GetDofsOnBdr(entity type, int ientity, Array<int> & dofs);
	virtual const SharingMap & GetEntityTrueEntity(int codim) const { return topo.EntityTrueEntity(codim); }

protected:
	virtual int getNumberOf(int type) const;
	virtual void checkMyInvariants() const;

private:
	void computeOffset(entity type);
	void buildPeakDofTable();
	void buildRidgeDofTable();
	void buildFacetDofTable();
	void buildElementDofTable();

	int entity_hasInteriorDofs[4];
	int entityType_nDofs[4];

	const AgglomeratedTopology & topo;

	Array<int> * entity_NumberOfInteriorDofsNullSpace[4];
	Array<int> * entity_NumberOfInteriorDofsRangeTSpace[4];
	Array<int> * entity_InteriorDofOffsets[4];

	// This variable will contain nDofs when finalized[ELEMENT] = true.
	int nDofs;

	// Type of Global Dof:  RangeTSpace NullSpace.
	Array<int> DofType;
};


#endif /* DOFHANDLER_HPP_ */
