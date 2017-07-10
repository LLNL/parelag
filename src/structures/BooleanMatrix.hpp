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

#ifndef BOOLEANMATRIX_HPP_
#define BOOLEANMATRIX_HPP_

class MultiVector;

class BooleanMatrix
{
public:
	BooleanMatrix(SparseMatrix & A);
	BooleanMatrix(int * i, int * j, int size, int width);
	int Size() const;
	int Width() const;
	BooleanMatrix * Transpose() const;
	void SetOwnerShip(int owns);
	inline int NumNonZeroElems() const { return I[size]; }
	int MaxRowSize() const;
	inline int RowSize(int i) const { return I[i+1] - I[i]; }
	inline const int * GetI() const {return I;}
	inline const int * GetJ() const {return J;}
	inline int * GetI() {return I;}
	inline int * GetJ() {return J;}
	void GetRow(int i, Array<int> & cols);
	void GetRow(int i, Array<int> & cols) const;
	void GetRowCopy(int i, Array<int> & cols) const;
	SerialCSRMatrix * AsCSRMatrix() const;
	virtual ~BooleanMatrix();

	void Mult(const Vector & x, Vector & y) const;
	void MultTraspose(const Vector & x, Vector & y) const;
	void Mult(const MultiVector & x, MultiVector & y) const;
	void MultTraspose(const MultiVector & x, MultiVector & y) const;
	void Mult(const DenseMatrix & x, DenseMatrix & y) const;
	void MultTraspose(const DenseMatrix & x, DenseMatrix & y) const;


private:
	int owner;
	int size;
	int width;
	int * I;
	int * J;
};

BooleanMatrix * BoolMult(const BooleanMatrix & A, const BooleanMatrix & B);
BooleanMatrix * BoolMult(const SerialCSRMatrix & A, const SerialCSRMatrix & B);
BooleanMatrix * BoolMult(const BooleanMatrix & A, const SerialCSRMatrix & B);
BooleanMatrix * BoolMult(const SerialCSRMatrix & A, const BooleanMatrix & B);

SerialCSRMatrix * Mult(const BooleanMatrix & A, const SerialCSRMatrix & B);
SerialCSRMatrix * Mult(const SerialCSRMatrix & A, const BooleanMatrix & B);

#endif /* BOOLEANMATRIX_HPP_ */
