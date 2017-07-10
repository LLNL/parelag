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

#ifndef ELEMENTALMATRICESCONTAINER_HPP_
#define ELEMENTALMATRICESCONTAINER_HPP_

//! @class
/*!
 * @brief a way of storing a matrix in un-assembled form, so we can assemble over subdomains at will
 *
 */
class ElementalMatricesContainer
{
public:
	ElementalMatricesContainer(int nEntities);
    // Deep copy
    ElementalMatricesContainer(const ElementalMatricesContainer & orig);
	virtual ~ElementalMatricesContainer();
	int Finalized();

	// To be implemented in the future :)

	SparseMatrix * GetAsSparseMatrix();
/*
	SparseMatrix * AssembleGlobal(DofHandler & domain, DofHandler & range);
	SparseMatrix * AssembleAgglomerate(Topology::entity entity_type, DofAgglomeration & domain, DofAgglomeration & range);
	SparseMatrix * AssembleAgglomerate(Topology::entity entity_type, int entity_id, DofAgglomeration & domain, DofAgglomeration & range);
	SparseMatrix * AssembleAgglomerateII(Topology::entity entity_type, int entity_id, DofAgglomeration & domain, DofAgglomeration & range);
	SparseMatrix * AssembleAgglomerateIB(Topology::entity entity_type, int entity_id, DofAgglomeration & domain, DofAgglomeration & range);
	SparseMatrix * AssembleAgglomerateBI(Topology::entity entity_type, int entity_id, DofAgglomeration & domain, DofAgglomeration & range);
	SparseMatrix * AssembleAgglomerateBB(Topology::entity entity_type, int entity_id, DofAgglomeration & domain, DofAgglomeration & range);
*/

	void SetElementalMatrix(int i, DenseMatrix * smat);
	void ResetElementalMatrix(int i, DenseMatrix * smat);
	void SetElementalMatrix(int i, const double & val);
	DenseMatrix & GetElementalMatrix(int i);

private:
	Array<DenseMatrix *> emat;
};

#endif /* ELEMENTALMATRICESCONTAINIER_HPP_ */
