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

#ifndef MA57BLOCKOPERATOR_HPP_
#define MA57BLOCKOPERATOR_HPP_

#include <vector>

class MA57BlockOperator : public Solver
{
public:

	enum ordering{METIS_ORDERING, AMD_ORDERING};

	MA57BlockOperator(const int nBlocks_);

	void SetOrdering(ordering opt){opt_ordering = opt;}
	void SetBlock(int i, int j, const SparseMatrix & Aij_);
	void SetOperator(const Operator &op){mfem_error("MA57BlockOperator::SetOperator");}
	int Compute();


	virtual void Mult (const Vector & x, Vector & y) const;
	virtual void Mult (const MultiVector & x, MultiVector & y) const;
	virtual void Mult (const DenseMatrix & x, DenseMatrix & y) const;

	virtual ~MA57BlockOperator(){ };

private:

	void computeOffsets();
	// Assumes that all the diagonal blocks have non-zero diagonal
	int computeNnz();
	void matrix2lowerTriple();
	int symbolicFactorization();
	int numericFactorization();
	int solve(const double * x, double * y, int nrhs) const;

	int nBlocks;
	bool isComputed;
	ordering opt_ordering;
	Array2D<const SparseMatrix *> Aij;
	Array<int> offsets;

	std::vector<int>    iRow;
	std::vector<int>    jCol;
	std::vector<double> val;

	mutable Array<int>     info;
	        Array<double> rinfo;

	        Array<double> cntl;
	mutable Array<int> icntl;

	        Array<int>    keep;
	mutable Array<double> fact;
	mutable Array<int>   ifact;

	mutable Array<int>   iwork;
};



#endif /* MA57BLOCKOPERATOR_HPP_ */
