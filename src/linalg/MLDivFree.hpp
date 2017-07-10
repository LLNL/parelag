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

#ifndef MLDIVFREE_HPP_
#define MLDIVFREE_HPP_

class MLDivFree : public Solver
{
public:
	enum CycleType {FULL_SPACE, DIVFREE_SPACE};
	MLDivFree(BlockMatrix * A, Array<DeRhamSequence *> & seqs, Array<int> & label_ess);
	void SetTolForPressureRank(double tol){numerical_zero = tol;}
	void SetOperator(const Operator &op);
	void SetBlockMatrix(BlockMatrix * A);
	void Build(const Vector & ess_nullspace_p, const SparseMatrix & mass_p);

	void SetCycle(CycleType c);
	void Mult(const Vector & x, Vector & y) const;
	void Mult(const MultiVector & x, MultiVector & y) const;

	virtual ~MLDivFree();

protected:

	enum { VELOCITY=0, PRESSURE=1};
	void MGVcycle(const Vector & x, Vector & y) const;
	void MGVcycleOnlyU(const Vector & x, Vector & y) const;

	//subdomainSmoother will always use the value of sol
	void subdomainSmoother(int i, const Vector & rhs, Vector & sol) const;
	//nullSpaceSmoother will always use the value of sol
	void nullSpaceSmoother(int i, const Vector & rhs, Vector & sol) const;
    //coarse solver
	void coarseSolver(const Vector & rhs, Vector & sol) const;

	void computeSharedDof( int ilevel, Array<int> & is_shared);
	int getLocalInternalDofs(int ilevel, int iAE, Array<int> & loc_dof) const;
	int getLocalInternalDofs(int comp, int ilevel, int iAE, Array<int> & loc_dof) const;
	int getLocalDofs(int comp, int ilevel, int iAE, Array<int> & loc_dof) const;
	int isRankDeficient(const SparseMatrix & Bt, const Vector & x) const;
	BlockMatrix * PtAP(BlockMatrix * A, BlockMatrix * P) const;

	//Sequence
	Array<DeRhamSequence *> sequence;
	int l2form;
	int hdivform;
	int hcurlform;
	//Number of levels (set by P.Size()+1)
	int nLevels;
	//Arithmetic TrueDof Complexitity
	int arithmeticTrueComplexity;
	//Arithmetic Dof Complexitity
	int arithmeticComplexity;
	// fine level saddle point matrix
	BlockMatrix * A;
	Array<BlockMatrix * > Al;
	Array<BlockOperator * > trueAl;
	Array<HypreParMatrix * > Cl;
	Array<Solver *> Maux;
	// Interpolators
	Array<BlockMatrix  * > P;
	Array<BlockOperator *> trueP;
	Array<SparseMatrix *> P00_to_be_deleted;
	// Cochain Projectors
	Array<BlockMatrix  * > Pi;

	//For each level, excluded the coarsest
	Array<BlockMatrix *> AE_dof;

	//We assume that in *_data the dofs for all levels are strided one after the other (from fine to coarse);
	// levelTrueStart is an array of length nLevels+1 such that
	// levelTrueStart[i] is such that *_data[levelTrueStart[i]] point to the first dofs at level i.
	Array<int> levelTrueStart;
	// Multiplier Stuff at level i goes from levelTrueStartMultiplier[i] to levelTrueStart[i+1].
	// Primal variable stuff at level i goes from levelTrueStart[i] to levelTrueStartMultiplier[i].
	Array<int> levelTrueStartMultiplier;
	//We assume that in *_data the dofs for all levels are strided one after the other (from fine to coarse);
	// levelStart is an array of length nLevels+1 such that
	// levelStart[i] is such that *_data[levelStart[i]] point to the first dofs at level i.
	Array<int> levelStart;
	// Multiplier Stuff at level i goes from levelStartMultiplier[i] to levelStart[i+1].
	// Primal variable stuff at level i goes from levelStart[i] to levelTrueMultiplier[i].
	Array<int> levelStartMultiplier;
	// 0 if the dof belongs to only one AE, 1 if it belongs to multiple arrays. Uses LevelStart
	Array<int> dof_is_shared_among_AE_data;
	// rhs (both u and p) for the multilevel reconstruction of \hat{u} (solve div equation) and p or just the auxiliary variable
	// Uses LevelTrueStart
	mutable Array<double> trueRhs_data;
	// sol (both u and p) for the multilevel reconstruction of \hat{u} (solve div equation) and p or just the auxiliary variable
	// Uses LevelTrueStart
	mutable Array<double> trueSol_data;
	// For the subdomain smoother
	mutable Array<double> rhs_data;
	mutable Array<double> sol_data;
	// Auxiliary workspace of local size
	// Uses levelStart and levelTrueStart
	Array<double> essnullspace_data;
	// Uses levelStart and levelTrueStart
	Array<double> t_data;
	//! If \| Bt_loc * ess_nullspace_loc \|_inf < numerical_zero then we consider A_loc singular
	double numerical_zero;

	CycleType my_cycle;

};


#endif /* MLDIVFREE_HPP_ */
