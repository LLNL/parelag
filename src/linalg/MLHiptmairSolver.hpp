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

/*
 * MLHiptmairSolver.hpp
 *
 *  Created on: Apr 17, 2015
 *      Author: uvilla
 */

#ifndef SRC_LINALG_MLHIPTMAIRSOLVER_HPP_
#define SRC_LINALG_MLHIPTMAIRSOLVER_HPP_

class HdivProblem3D
{
public:
	typedef HypreExtension::HypreADS CoarseSolver;
	typedef HypreExtension::HypreADSData CoarseSolverData;
	static int form;
};

class HdivProblem2D
{
public:
	typedef HypreExtension::HypreAMS CoarseSolver;
	typedef HypreExtension::HypreAMSData CoarseSolverData;
	static int form;
};

class HcurlProblem
{
public:
	typedef HypreExtension::HypreAMS CoarseSolver;
	typedef HypreExtension::HypreAMSData CoarseSolverData;
	static int form;
};

class HypreSmootherData
{
public:
	HypreSmootherData();
	int type;
	int relax_times;
	double relax_weight;
	double omega;
	int poly_order;
	double poly_fraction;
};

template<class PROBLEM>
class MLHiptmairSolver: public Solver
{
public:
	typedef PROBLEM Problem;
	typedef typename Problem::CoarseSolver CoarseSolver;
	typedef typename Problem::CoarseSolverData CoarseSolverData;

	MLHiptmairSolver(Array<DeRhamSequence *> & seqs, Array<int> & label_ess);
	void SetOperator(const Operator & A);
	void SetMatrix(HypreParMatrix * A);

	void SetHypreSmootherProperties(HypreSmootherData & data);
	void SetCoarseSolverProperties(CoarseSolverData & data);

	/// Operator application
	virtual void Mult (const Vector & x, Vector & y) const;

	/// Action of the transpose operator
	virtual void MultTranspose (const Vector & x, Vector & y) const
	{
	    mfem_error("MultTranspose of MLPreconditioner not implemented");
	}

	virtual ~MLHiptmairSolver();

private:
	//! compute the hierarchy
	void compute();
	//! Recursive multigrid algorithm
	void MGCycle(int level) const;
	//! clean up data
	void cleanUp();

	void presmoothing(int level, const Vector & res, Vector & sol) const;
	void postsmoothing(int level, const Vector & res, Vector & sol) const;

	Array<DeRhamSequence *> & seqs;
	Array<int> label_ess;

	//! number of levels
	int nLevels;

	//! A at each level (nLevels)
	Array<HypreParMatrix *> A;
	//! Auxiliary A at each level, but the coarsest (nLevels-1)
	Array<HypreParMatrix *> Aaux;

	//! Derivative operators at each levels, but the coarset (nLevels-1)
	// Aaux = Dt A D
	Array<HypreParMatrix *> D;

	//! Prolongator operators between levels (nLevels-1)
	Array<HypreParMatrix *> P;

	//! A-Smoother at each level (nLevels-1)
	Array<HypreSmoother *> SA;
	//! Auxiliary-Smoother at each level (nLevels-1)
	Array<HypreSmoother *> Saux;
	HypreSmootherData Sdata;

	//! coarse solver
	CoarseSolver * coarseSolver;
	CoarseSolverData coarseSolverData;

	mutable Array<Vector*> v;   // Approximate solution on all coarse levels
	mutable Array<Vector*> d;   // Defects at all levels
	mutable Array<Vector*> t;

	mutable Vector res1;
	mutable Vector sol1;
	mutable Vector resaux;
	mutable Vector solaux;

	double arithmeticComplexity;
	double operatorComplexity;
};

#endif /* SRC_LINALG_MLHIPTMAIRSOLVER_HPP_ */
