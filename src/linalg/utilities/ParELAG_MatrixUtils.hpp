/*
  Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-745557. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#ifndef MATRIXUTILS_HPP_
#define MATRIXUTILS_HPP_

#include <memory>

#include <mfem.hpp>

namespace parelag
{
bool AreAlmostEqual(
    const mfem::SparseMatrix & A, const mfem::SparseMatrix & B,
    const std::string & Aname, const std::string & Bname,
    double tol, bool verbose = false, std::ostream & os = std::cout);

bool AreAlmostEqual(
    const mfem::SparseMatrix & A, const mfem::SparseMatrix & B,
    const mfem::SparseMatrix & G,
    const std::string & Aname, const std::string & Bname,
    const std::string & Gname,
    double tol, bool verbose = false, std::ostream & os = std::cout);

bool IsAlmostIdentity(
    const mfem::SparseMatrix & A, double tol, bool verbose = false);

bool IsDiagonal(const mfem::SparseMatrix & A);

void fillSparseIdentity(int * I, int * J, double * A, int size);

std::unique_ptr<mfem::SparseMatrix> createSparseIdentityMatrix(int size);

// Returns a matrix 1 by width, with entries A(0,i) = data[i] for
// i \in 0 ... width-1
std::unique_ptr<mfem::SparseMatrix>
createSparseMatrixRepresentationOfScalarProduct(double * data, int width);

void destroySparseMatrixRepresentationOfScalarProduct(mfem::SparseMatrix *&);

std::unique_ptr<mfem::SparseMatrix> diagonalMatrix(double * data, int size);

std::unique_ptr<mfem::SparseMatrix> diagonalMatrix(int size);

std::unique_ptr<mfem::SparseMatrix> spzeros(int nrows, int ncols);

void dropSmallEntry(mfem::SparseMatrix & A, double tol);

void signumTransformation(mfem::SparseMatrix & A);

std::unique_ptr<mfem::SparseMatrix> DeepCopy(mfem::SparseMatrix & A);

void CheckMatrix(const mfem::SparseMatrix & A);

//SparseMatrix Addition
std::unique_ptr<mfem::SparseMatrix> Add(
    double a, const mfem::SparseMatrix & A,
    double b, const mfem::SparseMatrix & B,
    double c, const mfem::SparseMatrix & C);

void Full(const mfem::SparseMatrix & A, mfem::DenseMatrix & Adense);

void Full(const mfem::SparseMatrix & Asparse, mfem::DenseMatrix & Adense,
          int i_offset, int j_offset);

void AddMatrix(
    const mfem::DenseMatrix &A, const mfem::SparseMatrix & B,
    mfem::DenseMatrix & C);
void AddMatrix(
    double sA, const mfem::DenseMatrix &A,
    double sB, const mfem::SparseMatrix & B, mfem::DenseMatrix & C);

/// A += B
void AddOpenFormat(mfem::SparseMatrix & A, mfem::SparseMatrix & B);

std::unique_ptr<mfem::SparseMatrix> Kron(
    mfem::SparseMatrix & A, mfem::SparseMatrix & B);

void Mult(const mfem::SparseMatrix & A, const mfem::DenseMatrix & B,
          mfem::DenseMatrix & out);

std::unique_ptr<mfem::SparseMatrix> MultAbs(const mfem::SparseMatrix &A, const mfem::SparseMatrix &B);

/// Computes an RAP where the offd block of RT is ignored in the computation.
std::unique_ptr<mfem::HypreParMatrix>
IgnoreNonLocalRange(
    hypre_ParCSRMatrix* RT, hypre_ParCSRMatrix* A, hypre_ParCSRMatrix* P);

/// Does a "real" RAP (i.e. you give it R, A, and P, and this just
/// multiplies them together, unlike MFEM, where you give Rt and
/// then it compute R=Transpose(Rt) before multiplying...)
template <class MatrixType>
std::unique_ptr<MatrixType> RAP(
    const MatrixType & R, const MatrixType & A, const MatrixType & P);

/// Computes P^T*A*P given matrices A and P of the same type. The
/// type must have "Transpose" and "Mult" defined for them.
template <class MatrixType>
std::unique_ptr<MatrixType> PtAP(
    const MatrixType & A, const MatrixType & P);

/// Compute \| A(irow, :) \|_1
double RowNormL1(const mfem::SparseMatrix & A, int irow);

/*! \brief A function generating a matrix from another matrix.

  It generates a diagonal matrix from another sparse matrix.
  If we consider the generalized eigenvalue problem:
  \f$ A\mathbf{x} = \lambda B \mathbf{x}\f$,
  then this function generates sparse B from given sparse \a A.
  Here B is the weighted l1-smoother i.e. \f$ B = \text{diag}(d_i) \f$ with
  entries \f$ d_i = \sum\limits_j |a_{ij}| \sqrt{\frac{a_{ii}}{a_{jj}}} \f$.

  \param A (IN) The first matrix.
  \param diagonal_matrix (OUT) The second (generated) matrix as a vector,
  since it is diagonal.
*/
void Weightedl1Smoother(
    const mfem::SparseMatrix& A, mfem::Vector& diagonal_matrix);

void Block2by2(
    mfem::DenseMatrix &A00, mfem::DenseMatrix &A01,
    mfem::DenseMatrix &A10, mfem::DenseMatrix &A11, mfem::DenseMatrix &A);

void BlockDiag2by2(
    mfem::DenseMatrix &A00, mfem::DenseMatrix &A11, mfem::DenseMatrix &A);

void SplitMatrixHorizontally(
    const mfem::DenseMatrix &A, int middle_row,
    mfem::DenseMatrix &top, mfem::DenseMatrix &bottom);

std::unique_ptr<mfem::HypreParMatrix>
Mult(const mfem::HypreParMatrix& A, const mfem::HypreParMatrix& B,
     bool own_starts = true);
}//namespace parelag
#endif /* MATRIXUTILS_HPP_ */
