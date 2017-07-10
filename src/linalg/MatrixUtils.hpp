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

#ifndef MATRIXUTILS_HPP_
#define MATRIXUTILS_HPP_

bool AreAlmostEqual(const SparseMatrix & A, const SparseMatrix & B, const std::string & Aname, const std::string & Bname, double tol, bool verbose = false, std::ostream & os = std::cout);
bool AreAlmostEqual(const SparseMatrix & A, const SparseMatrix & B, const SparseMatrix & G, const std::string & Aname, const std::string & Bname,const std::string & Gname, double tol, bool verbose = false, std::ostream & os = std::cout);
bool IsAlmostIdentity(const SparseMatrix & A, double tol, bool verbose = false);
bool IsDiagonal(const SparseMatrix & A);

void fillSparseIdentity(int * I, int * J, double * A, int size);

SparseMatrix * createSparseIdentityMatrix(int size);

// Returns a matrix 1 by width, with entries A(0,i) = data[i] for i \in 0 ... width-1
SparseMatrix * createSparseMatrixRepresentationOfScalarProduct(double * data, int width);
void          destroySparseMatrixRepresentationOfScalarProduct(SparseMatrix *&);

SparseMatrix * diagonalMatrix(double * data, int size);
SparseMatrix * diagonalMatrix(int size);
SparseMatrix * spzeros(int nrows, int ncols);

void dropSmallEntry(SparseMatrix & A, double tol);
void signumTransformation(SparseMatrix & A);

SparseMatrix * DeepCopy(SparseMatrix & A);

void CheckMatrix(SparseMatrix & A);

//SparseMatrix Addition
SparseMatrix * Add(double a, const SparseMatrix & A, double b, const SparseMatrix & B, double c, const SparseMatrix & C);

void Full(const SparseMatrix & A, DenseMatrix & Adense);
void AddMatrix(const DenseMatrix &A, const SparseMatrix & B, DenseMatrix & C);
void AddMatrix(double sA, const DenseMatrix &A, double sB, const SparseMatrix & B, DenseMatrix & C);
//A += B
void AddOpenFormat(SparseMatrix & A, SparseMatrix & B);

SparseMatrix * Kron(SparseMatrix & A, SparseMatrix & B);

void Mult(const SparseMatrix & A, const DenseMatrix & B, DenseMatrix & out);

SparseMatrix * PtAP(SparseMatrix & A, SparseMatrix & P);

/// Compute \| A(irow, :) \|_1
double RowNormL1(const SparseMatrix & A, int irow);

#endif /* MATRIXUTILS_HPP_ */
