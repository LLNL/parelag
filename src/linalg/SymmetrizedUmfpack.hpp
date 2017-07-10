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

#ifndef SYMMETRIZEDUMFPACK_HPP_
#define SYMMETRIZEDUMFPACK_HPP_

class SymmetrizedUmfpack : public Solver
{
public:
	SymmetrizedUmfpack();
	SymmetrizedUmfpack(SparseMatrix &A);
	SymmetrizedUmfpack(BlockMatrix &A);

	double * Control(){ return solver.Control; }
	double * Info(){ return solver.Info; }

	// Works on sparse matrices only; calls SparseMatrix::SortColumnIndices().
	virtual void SetOperator(const Operator &op);

	void SetPrintLevel(int print_lvl) { solver.Control[UMFPACK_PRL] = print_lvl; }

	virtual void Mult(const Vector &b, Vector &x) const;
	virtual void MultTranspose(const Vector &b, Vector &x) const;

	void Mult(const MultiVector &b, MultiVector &x) const;
	void MultTranspose(const MultiVector &b, MultiVector &x) const;

        void Mult(const DenseMatrix &b, DenseMatrix &x) const;
        void MultTranspose(const DenseMatrix &b, DenseMatrix &x) const;

	virtual ~SymmetrizedUmfpack();

private:
	UMFPackSolver solver;
	SparseMatrix * Amono;
	mutable Vector help;
};

#endif /* SYMMETRIZEDUMFPACK_HPP_ */
