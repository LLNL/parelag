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

#ifndef COCHAINPROJECTOR_HPP_
#define COCHAINPROJECTOR_HPP_

//! @class CochainProjector
/**
	\brief Projects from fine space to coarse space, eg from S_h to S_H
*/
class CochainProjector
{
public:
	CochainProjector(AgglomeratedTopology * cTopo, DofHandler * cDof, DofAgglomeration * dofAgg, SparseMatrix * P);
	virtual ~CochainProjector();

	void CreateDofFunctional(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, const MultiVector & localProjector, const SparseMatrix & M_ii);
	void SetDofFunctional(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, DenseMatrix * dof_lin_func );

	//@{ Local Projectors
	// Methods to Perform local projections from the Bdr to the interior of an agglomerated element.
	// VFine and res have the dimension of the degree of freedom on the agglomerated entity (including the one on the boundary)
	// vCoarse has the dimension of the Coarse Degree of Freedom that lives on the boundary of the agglomerated entity

	// res: input the MultiVector we want to interpolate on the boundary. Output: The residual of the interpolation
//	void LocalResidualBdr(Topology::entity entity_type, int entity_id, MultiVector & res);
//	void ProjectLocalBdr(Topology::entity entity_type, int entity_id, const MultiVector & vFine, MultiVector & vCoarse, MultiVector & res);
//	void ProjectLocalBdr(Topology::entity entity_type, int entity_id, const MultiVector & vFine, MultiVector & vCoarse);

	//@}

	//@{ Global Projectors
	// Note res should not point to the same memory area as vFine
	void Project(const MultiVector & vFine, MultiVector & vCoarse);
	void Project(const MultiVector & vFine, MultiVector & vCoarse, MultiVector & res);
	//@}

	void ComputeProjector();
	const SparseMatrix & GetProjectorMatrix();
	SparseMatrix * GetIncompleteProjector();

	//@{ Integrity Check
	void Check();
	void LongCheck();
	void CheckInvariants();
	//@}

	int Finalized();


private:

	SparseMatrix * assembleInternalProjector(int codim);

	AgglomeratedTopology * cTopo;
	DofHandlerALG * cDof;
	DofAgglomeration * dofAgg;
	SparseMatrix * P;

	// Each entry of the array corresponds to a codimensions. Therefore the size is 1 for L2, 2 for Hdiv, etc...
	// If M_agg is the assembled mass matrix on the agglomerated entity (only for interior dofs) and
	// Let localProjector the piece of the interpolation matrix
	// The localCoarseM is given by: localProjector' * M_agg * localProjector
	// The dofLinearFunctional is then: localCoarseM^-1 * localProjector' * M_agg
	Array<ElementalMatricesContainer *> dofLinearFunctional;

	SparseMatrix * Pi;


};

#endif /* COCHAINPROJECTOR_HPP_ */
