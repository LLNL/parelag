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

#ifndef DERHAMSEQUENCEFE2_HPP_
#define DERHAMSEQUENCEFE2_HPP_

class DeRhamSequenceFE : public DeRhamSequence
{
public:
	DeRhamSequenceFE(AgglomeratedTopology * topo, int nSpaces);
	void Build(ParMesh * mesh_, Array<FiniteElementSpace *> & feSpace, Array<DiscreteInterpolator *> & di, Array2D<BilinearFormIntegrator *> & mi);
	void ReplaceMassIntegrator(AgglomeratedTopology::EntityByCodim ientity, int jform, BilinearFormIntegrator * m, bool recompute = true);
	virtual SparseMatrix * ComputeProjectorFromH1ConformingSpace(int jform);
	virtual ~DeRhamSequenceFE();
	virtual void showP(int jform, SparseMatrix & P, Array<int> & parts);
	virtual void show(int jform, MultiVector & v);
	FiniteElementSpace * GetFeSpace(int jform){ return feSpace[jform]; }
	MultiVector * InterpolateScalarTargets(int jform, const Array<Coefficient *> & scalars);
	MultiVector * InterpolateVectorTargets(int jform, const Array<VectorCoefficient *> & vectors);
	virtual void ProjectCoefficient(int jform, Coefficient & c, Vector & v);
	virtual void ProjectVectorCoefficient(int jform, VectorCoefficient & c, Vector & v);
	virtual DeRhamSequenceFE * FemSequence(){ return this;}


	void DEBUG_CheckNewLocalMassAssembly();

protected:

	void buildDof();
	void assembleLocalMass();
	void assembleLocalMass_ser();
	void assembleLocalMass_old();
	void assembleLocalMass_omp();
	void assembleDerivative();
	virtual void computePVTraces(AgglomeratedTopology::EntityByCodim icodim, Vector & PVinAgg) = 0;

	int owns_data;
	Array<DiscreteInterpolator *> di;
	Array2D<BilinearFormIntegrator *> mi;
	Array<FiniteElementCollection *> fecColl;
	Array<FiniteElementSpace *> feSpace;
	ParMesh * mesh;

private:

	const FiniteElement * GetFE(int ispace, int ientity_type, int ientity) const;
	ElementTransformation * GetTransformation(int ientity_type, int ientity) const;
	void GetTransformation(int ientity_type, int ientity, IsoparametricTransformation & tr) const;
};

class DeRhamSequence3D_FE : public DeRhamSequenceFE
{
public:
	DeRhamSequence3D_FE(AgglomeratedTopology * topo, ParMesh * mesh, int order);
	virtual ~DeRhamSequence3D_FE();
protected:
	virtual void computePVTraces(AgglomeratedTopology::EntityByCodim icodim, Vector & pv);
};

class DeRhamSequence2D_Hdiv_FE : public DeRhamSequenceFE
{
public:
	DeRhamSequence2D_Hdiv_FE(AgglomeratedTopology * topo, ParMesh * mesh, int order);
	virtual ~DeRhamSequence2D_Hdiv_FE();
protected:
	virtual void computePVTraces(AgglomeratedTopology::EntityByCodim icodim, Vector & pv);
};

void InterpolatePV_L2(const FiniteElementSpace * fespace, const SparseMatrix & AE_element, Vector & AE_Interpolant);
void InterpolatePV_HdivTraces(const FiniteElementSpace * fespace, const SparseMatrix & AF_facet, Vector & AF_Interpolant);
void InterpolatePV_HcurlTraces(const FiniteElementSpace * fespace, const SparseMatrix & AR_ridge, Vector & AR_Interpolant);
void InterpolatePV_H1Traces(const FiniteElementSpace * fespace, const SparseMatrix & AP_peak, Vector & AP_Interpolant);

#endif /* DERHAMSEQUENCEFE_HPP_ */
