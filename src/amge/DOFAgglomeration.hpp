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

#ifndef DOFAGGLOMERATION_HPP_
#define DOFAGGLOMERATION_HPP_

//! @class DofAgglomeration
/**
	@brief keeps track of how Dofs are aggregated, used in particular to make local agglomerate matrices for solving local problems 
*/
class DofAgglomeration
{
public:
	DofAgglomeration(AgglomeratedTopology * topo, DofHandler * dof );

	virtual ~DofAgglomeration();

	void CheckAdof();

	inline int GetNumberFineEntities(AgglomeratedTopology::EntityByCodim entity_type)
	{
		return fineTopo->GetNumberLocalEntities(entity_type);
	}

	inline int GetNumberCoarseEntities(AgglomeratedTopology::EntityByCodim entity_type)
	{
		return coarseTopo->GetNumberLocalEntities(entity_type);
	}

	inline int GetNumberAgglomerateInternalDofs(AgglomeratedTopology::EntityByCodim entity_type, int entity_id)
	{
		int begin, end;
		GetAgglomerateInternalDofRange(entity_type, entity_id, begin, end);
		return end-begin;
	}

	DofHandler * GetDofHandler(){ return dof; }
	SparseMatrix * GetAEntityDof(AgglomeratedTopology::EntityByCodim entity_type){ return AEntity_dof[entity_type]; }
	const SparseMatrix & GetAEntityDofTable(AgglomeratedTopology::EntityByCodim entity_type){ return *(AEntity_dof[entity_type]); }
	void GetAgglomerateInternalDofRange(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, int & begin, int & end);
	void GetAgglomerateBdrDofRange(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, int & begin, int & end);
	void GetAgglomerateDofRange(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, int & begin, int & end);
	void GetViewAgglomerateInternalDofGlobalNumering(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, Array<int> & gdofs);
	void GetViewAgglomerateBdrDofGlobalNumering(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, Array<int> & gdofs);
	void GetViewAgglomerateDofGlobalNumering(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, Array<int> & gdofs);
	int * GetAgglomerateInternalDofGlobalNumering(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, int * gdofs);
	MultiVector * DistributeGlobalMultiVector(AgglomeratedTopology::EntityByCodim entity_type, const MultiVector & vg);
	Vector * DistributeGlobalVector(AgglomeratedTopology::EntityByCodim entity_type, Vector & vg);

	// Each row represents an entity of the Boundary: We first use PEAK (if any), the RIDGE (if any) and Finally FACETS.
//	void MapDofFromLocalNumberingAEtoLocalNumberingBoundaryAE(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, Table & map);


	/*
	 * input:
	 * -- entity_type: 0 -> Element, 1 -> Facet, 2 -> Ridge, 3 -> Peak
	 * -- M_e: entity matrix of size nrDof_range x nrDof_domain
	 * -- rangeSpace:  index in the dof array corresponding to the range of the operator
	 * -- domainSpace: index in the dof array corresponding to the domain of the operator
	 * output:
	 * -- Agglomerated Entity matrix of size nADof_range x nADof_domain
	 *
	 *  Description:
	 *  Given M_e = diag( m_ei ) where:
	 *  -- m_ei is the local matrix on entity i
	 *  computes M_E =  diag( m_EJ ) where:
	 *  -- m_EJ is the local matrix on agglomerated Entity J.
	 *
	 *  Implementation: M_E = RAP(rDof_ADof_range, M_e, rDof_ADof_domain)
	 *  where rDof_RDof is the +1,-1 matrix that maps the set of repeated dof for each entity in fineTopo
	 *  into the set Agglomerated dof for each Agglomerated entity in coarseTopo.
	 *
	 */
	friend SparseMatrix * AssembleAgglomerateMatrix(AgglomeratedTopology::EntityByCodim entity_type, SparseMatrix & M_e, DofAgglomeration * range, DofAgglomeration * domain);

	friend SparseMatrix * AssembleAgglomerateRowsGlobalColsMatrix(AgglomeratedTopology::EntityByCodim entity_type, SparseMatrix & M_e, DofAgglomeration * range, DofHandler * domain);

	friend SparseMatrix * DistributeProjector(AgglomeratedTopology::EntityByCodim entity_type, SparseMatrix & P_t, DofHandler * range, DofAgglomeration * domain);

	/*
	 * input:
	 * -- entity_type: 0 -> Element, 1 -> Facet, 2 -> Ridge, 3 -> Peak
	 * -- M_a: Agglomerated Entity matrix of size nADof_range x nADof_domain
	 * -- rangeSpace:  index in the dof array corresponding to the range of the operator
	 * -- domainSpace: index in the dof array corresponding to the domain of the operator
	 * output:
	 * -- M_g global matrix
	 *
	 *  Description:
	 *  Given M_a = diag( m_ai ) where:
	 *  -- m_ai is the local sparse matrix on agglomerate i
	 *  computes M_g.
	 *
	 *  Implementation: M_g = Dof_ADof_range * M_a, ADof_Dof_domain
	 *
	 */
	friend SparseMatrix * Assemble(AgglomeratedTopology::EntityByCodim entity_type, SparseMatrix & M_a, DofAgglomeration * range, DofAgglomeration * domain);

	/*
	 * input:
	 * -- entity_type: 0 -> Element, 1 -> Facet, 2 -> Ridge, 3 -> Peak
	 * -- M_g: entity matrix of size nDof_range x nDof_domain
	 * -- rangeSpace:  index in the dof array corresponding to the range of the operator
	 * -- domainSpace: index in the dof array corresponding to the domain of the operator
	 * output:
	 * -- Agglomerated Entity matrix of size nADof_range x nADof_domain
	 *
	 *  Description:
	 *  Given D_g the fully assembled finite element matrix
	 *  computes D_E =  diag( d_EJ ) where:
	 *  -- d_EJ is the local matrix on agglomerated Entity J.
	 *
	 *  Implementation: D_E = RAP(Dof_ADof_range', D_G, Dof_ADof_domain)
	 *  where Dof_ADof is the +1,-1 matrix that maps the set of global dof
	 *  into the set Agglomerate dof for each Agglomerated entity in coarseTopo.
	 *
	 */
	friend SparseMatrix * DistributeAgglomerateMatrix(AgglomeratedTopology::EntityByCodim entity_type, SparseMatrix & D_g, DofAgglomeration * range, DofAgglomeration * domain);

#if 0
	/*
	 * Computes the rectangular matrix M_BC of size nAdof x nDof
	 * such that the rows are the same as in the Agglomerate Entity matrix M
	 * and the columns are the same as the global matrix M.
	 * That is
	 * M_BC = RAP(rDof_ADof_range, M_e, rDof_dof_domain) of size nAdof x nDof,
	 */
	friend SparseMatrix * AssembleAgglomerateMatrixBC(Topology::entity entity_type, SparseMatrix & M_e, DofAgglomeration * range, DofAgglomeration * domain);



	/*
	 * input:
	 * -- entity_type: 0 -> Element, 1 -> Facet, 2 -> Ridge, 3 -> Peak
	 * -- M_e: entity matrix of size nrDof_range x nrDof_domain
	 * -- rangeSpace:  index in the dof array corresponding to the range of the operator
	 * -- domainSpace: index in the dof array corresponding to the domain of the operator
	 * output:
	 * -- Agglomerated Entity matrix of size nRDof_range x nRDof_domain
	 *
	 *  Description:
	 *  Given M_e = diag( m_ei ) where:
	 *  -- m_ei is the local matrix on entity i
	 *  computes M_E = PERM_range^T diag( m_EJ ) PERM_domain where:
	 *  -- m_EJ is the local matrix on agglomerated Entity J.
	 *  -- and PERM_range, PERM_domain are the permutation matrices so that
	 *     the degree of freedom (of the domain and range space) that are shared among entities are last.
	 *
	 *  Implementation: M_E = RAP(rDof_RDof_range, M_e, rDof_RDof_domain)
	 *  where rDof_RDof is the +1,-1 matrix that maps the set of repeated dof for each entity in fineTopo
	 *  into the set Repeated dof for each Agglomerated entity in coarseTopo.
	 *
	 */
	friend SparseMatrix * AssembleAgglomerateMatrix(Topology::entity entity_type, SparseMatrix & M_e, DofAgglomeration * range, DofAgglomeration * domain);

	/*
	 * input:
	 * -- entity_type: 0 -> Element, 1 -> Facet, 2 -> Ridge, 3 -> Peak
	 * -- M_e: entity matrix of size nrDof_range x nrDof_domain
	 * -- rangeSpace:  index in the dof array corresponding to the range of the operator
	 * -- domainSpace: index in the dof array corresponding to the domain of the operator
	 * output:
	 * -- Agglomerated Entity matrix of size nRDof_range x nRDof_domain
	 *
	 *  Description:
	 *  Given M_e = diag( m_ei ) where:
	 *  -- m_ei is the local matrix on entity i
	 *  computes M_E = PERM_range^T diag( m_EJ ) PERM_domain where:
	 *  -- m_EJ is the local matrix on agglomerated Entity J.
	 *  -- and PERM_range, PERM_domain are the permutation matrices so that
	 *     the degree of freedom (of the domain and range space) that are shared among entities are last.
	 *
	 *  Implementation: M_E = RAP(rDof_RDof_range, M_e, rDof_RDof_domain)
	 *  where rDof_RDof is the +1,-1 matrix that maps the set of repeated dof for each entity in fineTopo
	 *  into the set Repeated dof for each Agglomerated entity in coarseTopo.
	 *
	 */
	friend SparseMatrix * AssembleAgglomerateMatrix(Topology::entity entity_type, SparseMatrix & M_e, DofAgglomeration * range, DofAgglomeration * domain);


	/*
	 * Returns the principal submatrix of M_E of size nIdof x nIdof
	 * consisting of only the dofs that belong to one and only Agglomerated Entity
	 * That is:
	 * M_ii = RAP(rDof_RDof_range(:, internalDOFS), M_e, rDof_RDof_domain(:, internalDOFS))
	 */
	friend SparseMatrix * AssembleAgglomerateMatrixInternalDof(Topology::entity entity_type, SparseMatrix & M_e, DofAgglomeration * range, DofAgglomeration * domain);

	/*
	 * Computes the rectangular matrix M_BC of size nIdof x nDof
	 * such that the rows are the same as in the Agglomerate Entity matrix M_ii
	 * and the columns are the same as the global matrix M.
	 * That is
	 * M_BC = RAP(rDof_RDof_range(:, internalDOFS), M_e, rDof_dof) of size nIdof x nDof,
	 */
	friend SparseMatrix * AssembleAgglomerateMatrixBC(Topology::entity entity_type, SparseMatrix & M_e, DofAgglomeration * range, DofAgglomeration * domain);
#endif

private:
	// Reorder the j and val arrays of a so that the entries are ordered increasignly with respect to weights.
	// It returns for each row the number of occurrencies of minWeight
	int reorderRow(int nentries, int * j, double * a, int minWeight, int maxWeight, const Array<int> & weights);

	AgglomeratedTopology * fineTopo;
	AgglomeratedTopology * coarseTopo;
	DofHandler * dof;
	//! The index in this array represents the type of entity (0 = ELEMENT, 1 = FACET, 2 = RIDGE, 3 = PEAK).
	// Each matrix has for each row the ids and the orientation of the dofs in AgglomeratedEntity i.
	Array<SparseMatrix *> AEntity_dof;
	Array<SparseMatrix *> ADof_Dof;
	Array<SparseMatrix *> ADof_rDof;
	Array<Array<int>* > AE_nInternalDof;
	/*!
	 * dof_separatorType[i] = 0 --> i belongs to the interior of an agglomerated element
	 * dof_separatorType[i] = 1 --> i belongs to the interior of an agglomerated facet
	 * dof_separatorType[i] = 2 --> i belongs to the interior of an agglomerated ridge
	 * dof_separatorType[i] = 3 --> i belongs to an agglomerated peak
	 *
	 */
	Array<int> dof_separatorType;

	mutable Array<int> dofMapper;

};

#endif /* DOFAGGLOMERATION_HPP_ */
