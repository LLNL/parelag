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

#ifndef BOOLEANMATRIX_HPP_
#define BOOLEANMATRIX_HPP_

#include <memory>

#include "elag_typedefs.hpp"
#include "linalg/dense/ParELAG_MultiVector.hpp"

namespace parelag
{

// Can't we just use an mfem::Table?
class BooleanMatrix
{
public:
    BooleanMatrix() = delete;
    BooleanMatrix(BooleanMatrix & other) = delete;
    BooleanMatrix(const BooleanMatrix & other) = delete;
    BooleanMatrix(BooleanMatrix && other) = delete;
    BooleanMatrix& operator=(BooleanMatrix & other) = delete;
    BooleanMatrix& operator=(const BooleanMatrix & other) = delete;
    BooleanMatrix& operator=(BooleanMatrix && other) = delete;

    BooleanMatrix(mfem::SparseMatrix & A);
    BooleanMatrix(int * i, int * j, int size, int width);
    int Size() const noexcept;
    int Width() const noexcept;
    std::unique_ptr<BooleanMatrix> Transpose() const;
    void SetOwnerShip(bool owns) noexcept;
    inline int NumNonZeroElems() const noexcept { return I_[Size_]; }
    bool OwnsIJ() const noexcept { return OwnIJ_; }
    int MaxRowSize() const;
    inline int RowSize(int i) const noexcept { return I_[i+1] - I_[i]; }
    inline const int * GetI() const noexcept {return I_;}
    inline const int * GetJ() const noexcept {return J_;}
    inline int * GetI() noexcept {return I_;}
    inline int * GetJ() noexcept {return J_;}
    void GetRow(int i, mfem::Array<int> & cols);
    void GetRow(int i, mfem::Array<int> & cols) const;
    void GetRowCopy(int i, mfem::Array<int> & cols) const;
    std::unique_ptr<SerialCSRMatrix> AsCSRMatrix() const;
    virtual ~BooleanMatrix();

    void Mult(const mfem::Vector & x, mfem::Vector & y) const;
    void MultTranspose(const mfem::Vector & x, mfem::Vector & y) const;
    void Mult(const MultiVector & x, MultiVector & y) const;
    void MultTranspose(const MultiVector & x, MultiVector & y) const;
    void Mult(const mfem::DenseMatrix & x, mfem::DenseMatrix & y) const;
    void MultTranspose(const mfem::DenseMatrix & x, mfem::DenseMatrix & y) const;


private:
    bool OwnIJ_ = false;
    int Size_;
    int Width_;
    int * I_;
    int * J_;
};

std::unique_ptr<BooleanMatrix> BoolMult(const BooleanMatrix & A,
                                        const BooleanMatrix & B);
std::unique_ptr<BooleanMatrix> BoolMult(const SerialCSRMatrix & A,
                                        const SerialCSRMatrix & B);
std::unique_ptr<BooleanMatrix> BoolMult(const BooleanMatrix & A,
                                        const SerialCSRMatrix & B);
std::unique_ptr<BooleanMatrix> BoolMult(const SerialCSRMatrix & A,
                                        const BooleanMatrix & B);

std::unique_ptr<SerialCSRMatrix> Mult(const BooleanMatrix & A,
                                      const SerialCSRMatrix & B);
std::unique_ptr<SerialCSRMatrix> Mult(const SerialCSRMatrix & A,
                                      const BooleanMatrix & B);
}//namespace parelag
#endif /* BOOLEANMATRIX_HPP_ */
