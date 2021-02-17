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

#ifndef ELEMAGG_MULTIVECTOR_HPP_
#define ELEMAGG_MULTIVECTOR_HPP_

#include <memory>

#include <mfem.hpp>

namespace parelag
{
class MultiVector : public mfem::Vector
{
public:

    /// Default constructor for Vector. Sets size = 0 and data = NULL
    MultiVector();

    /// Creates vector of size s. (lda == s)
    MultiVector(int _number_of_vector, int s);

    // if lda < 0 then lda = size
    MultiVector(double *_data, int _number_of_vector, int _size, int _lda = -1);

    /// Copy constructor
    MultiVector(const MultiVector &);

    ~MultiVector() = default;

    /// lda = s
    void SetSizeAndNumberOfVectors(int size, int _number_of_vectors);

    void GetSubMultiVector(
        const mfem::Array<int> &dofs, MultiVector &elemvect) const;

    void GetSubMultiVector(
        const mfem::Array<int> &dofs, double *elem_data) const;

    void SetSubMultiVector(
        const mfem::Array<int> &dofs, const MultiVector &elemvect);

    void SetSubMultiVector(const mfem::Array<int> &dofs, double *elem_data);

    void AddSubMultiVector(
        const mfem::Array<int> &dofs, const MultiVector &elemvect);

    void AddSubMultiVector(const mfem::Array<int> &dofs, double *elem_data);

    // m = this
    void CopyToDenseMatrix(
        int ivectStart, int ivectEnd, int entryStart, int entryEnd,
        mfem::DenseMatrix & m);

    // m = this^T
    void CopyToDenseMatrixT(
        int ivectStart, int ivectEnd, int entryStart, int entryEnd,
        mfem::DenseMatrix & m);

    void GetVectorView(int ivect, mfem::Vector & view);

    void GetRangeView(int start, int end, MultiVector & view);

    double * GetDataFromVector(int ivect)
    {return GetData()+ivect*LDA_;}

    const double * GetDataFromVector(int ivect) const
    {return GetData()+ivect*LDA_;}

    int Size() const { return LocalSize_;}
    int LeadingDimension() const { return LDA_; }
    int NumberOfVectors() const {return NumVectors_;}

    // out[i] = this[p[i]];
    std::unique_ptr<MultiVector>
    Permutation(const mfem::Array<int> & p) const;

    std::unique_ptr<MultiVector>
    Permutation(const int * p, int size_p = -1) const;

    // out[p] = this;
    std::unique_ptr<MultiVector>
    InversePermutation(const mfem::Array<int> & p) const;

    std::unique_ptr<MultiVector>
    InversePermutation(const int * p, int size_p = -1, int out_size = -1) const;

    void Scale(const mfem::Vector & s);
    void InverseScale(const mfem::Vector & s);

    void RoundSmallEntriesToZero(double smallValue);

    MultiVector & operator=(const double *v);

    /// Redefine '=' for vector = vector.
    MultiVector & operator=(const MultiVector &v);

    /// Redefine '=' for vector = constant.
    MultiVector & operator=(double value);

    void Copy(MultiVector & copy) const;

    // y = M*x
    friend void MatrixTimesMultiVector(
        const mfem::SparseMatrix & M, const MultiVector & x, MultiVector & y);

    // y = M^T*x
    friend void MatrixTTimesMultiVector(
        const mfem::SparseMatrix & M, const MultiVector & x, MultiVector & y);

    // y = y + scaling*M*x
    friend void MatrixTimesMultiVector(
        double scaling, const mfem::SparseMatrix & M, const MultiVector & x,
        MultiVector & y);

    // y = y + scaling*M^T*x
    friend void MatrixTTimesMultiVector(
        double scaling, const mfem::SparseMatrix & M, const MultiVector & x,
        MultiVector & y);

private:
    int NumVectors_;
    int LDA_;
    int LocalSize_;
};

//y = M * x;
void Mult(const mfem::DenseMatrix & M, const MultiVector & x, MultiVector & y);

//y = M^T * x;
void MultTranspose(const mfem::DenseMatrix & M, const MultiVector & x, MultiVector & y);

// y = y + scaling M*x
void Mult(double scaling, mfem::DenseMatrix & M, const MultiVector & x,
          MultiVector & y);

void add(double a, const MultiVector &x,
         double b, const MultiVector &y, MultiVector &z);

//y = A * x;
void Mult(mfem::BlockMatrix & A, const MultiVector & x, MultiVector & y);

//y = A * x;
void Mult(const mfem::HypreParMatrix & A, const MultiVector & x, MultiVector & y);

// out = M*x;
std::unique_ptr<MultiVector> MatrixTimesMultiVector(
    const mfem::SparseMatrix & M, const MultiVector & x);

// out = M^T*x
std::unique_ptr<MultiVector> MatrixTTimesMultiVector(
    const mfem::SparseMatrix & M, const MultiVector & x);
}//namespace parelag
#endif /* MULTIVECTOR_HPP_ */
