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

#ifndef SYMMETRIZEDUMFPACK_HPP_
#define SYMMETRIZEDUMFPACK_HPP_

#include <mfem.hpp>

#include "linalg/dense/ParELAG_MultiVector.hpp"

namespace parelag
{
class SymmetrizedUmfpack : public mfem::Solver
{
public:
    SymmetrizedUmfpack() = default;
    SymmetrizedUmfpack(mfem::SparseMatrix &A);
    SymmetrizedUmfpack(mfem::BlockMatrix &A);

    double * Control(){ return solver_.Control; }
    double * Info(){ return solver_.Info; }

    // Works on sparse matrices only; calls SparseMatrix::SortColumnIndices().
    virtual void SetOperator(const mfem::Operator &op);

    void SetPrintLevel(int print_lvl) { solver_.Control[UMFPACK_PRL] = print_lvl; }

    virtual void Mult(const mfem::Vector &b, mfem::Vector &x) const;
    virtual void MultTranspose(const mfem::Vector &b, mfem::Vector &x) const;

    void Mult(const MultiVector &b, MultiVector &x) const;
    void MultTranspose(const MultiVector &b, MultiVector &x) const;

    void Mult(const mfem::DenseMatrix &b, mfem::DenseMatrix &x) const;
    void MultTranspose(const mfem::DenseMatrix &b, mfem::DenseMatrix &x) const;

    ~SymmetrizedUmfpack() = default;

private:
    mfem::UMFPackSolver solver_;
    std::unique_ptr<mfem::SparseMatrix> Amono_;
    mutable mfem::Vector help_;
};

}//namespace parelag
#endif // SYMMETRIZEDUMFPACK_HPP_
