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

#include <memory>

#include "ElementalMatricesContainer.hpp"

#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "utilities/elagError.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

ElementalMatricesContainer::ElementalMatricesContainer(int nEntities):
    emat_(nEntities)
{
}

ElementalMatricesContainer::ElementalMatricesContainer(
    const ElementalMatricesContainer & orig)
    : emat_(orig.emat_.size())
{
    const auto n_mats = orig.emat_.size();
    for (auto ii = decltype(n_mats){0}; ii < n_mats; ++ii)
        emat_[ii] = make_unique<DenseMatrix>(*(orig.emat_[ii]));
}

ElementalMatricesContainer::~ElementalMatricesContainer()
{
}

bool ElementalMatricesContainer::Finalized()
{
    const auto n_mats = emat_.size();
    for (auto ii = decltype(n_mats){0}; ii < n_mats; ++ii)
        if (!emat_[ii])
            return false;

    return true;
}

unique_ptr<SparseMatrix> ElementalMatricesContainer::GetAsSparseMatrix()
{

    //double smallEntry(1e-16);
    double smallEntry(0.);

    PARELAG_TEST_FOR_EXCEPTION(
        not this->Finalized(),
        std::runtime_error,
        "ElementalMatricesContainer::GetAsSparseMatrix():\n"
        "Container not finalized; cannot build sparse matrix!");

    int localNRows = 0, localNCols = 0, nrows = 0, ncols = 0,
        maxLocalNRows = 0, maxLocalNCols = 0;

    const auto n_mats = emat_.size();
    for (auto ii = decltype(n_mats){0}; ii < n_mats; ++ii)
    {
        localNRows = emat_[ii]->Height();
        localNCols = emat_[ii]->Width();

        nrows += localNRows;
        ncols += localNCols;

        if (localNRows > maxLocalNRows)
            maxLocalNRows = localNRows;

        if (localNCols > maxLocalNCols)
            maxLocalNCols = localNCols;
    }

    auto out = make_unique<SparseMatrix>(nrows, ncols);

    Array<int> row_ind(maxLocalNRows), col_ind(maxLocalNCols);

    int row_ind_start(0), col_ind_start(0);

    for (auto ii = decltype(n_mats){0}; ii < n_mats; ++ii)
    {
        localNRows = emat_[ii]->Height();
        localNCols = emat_[ii]->Width();

        row_ind.SetSize(localNRows);
        for (int irow(0); irow < localNRows; ++irow)
            row_ind[irow] = (row_ind_start++);

        col_ind.SetSize(localNCols);
        for (int icol(0); icol < localNCols; ++icol)
            col_ind[icol] = (col_ind_start++);

        double * data = emat_[ii]->Data();
        for (double * end = data+(localNRows*localNCols); data != end; ++data)
        {
            if (fabs(*data) < smallEntry)
                *data = 0.0;
        }

        out->SetSubMatrix(row_ind, col_ind, *(emat_[ii]));
    }

    out->Finalize();
    CheckMatrix(*out);

    return out;
}
/*
  void ElementalMatricesContainer::Mult(const Vector & x, Vector & y) const
  {
  int row_start(0), col_start(0), row_end(0), col_end(0), nLocalRows(0), nLocalCols(0);

  double * xx(x.GetData()), *yy(y.GetData());

  for (int i(0); i <  emat.Size(); ++i)
  {
  nLocalRows = emat[i]->Height();
  nLocalCols = emat[i]->Width();
  row_end = row_start+nLocalRows;
  col_end = col_start+nLocalCols;

  emat[i]->Mult(xx+col_start, yy+row_start);

  //Update
  row_start = row_end;
  col_start = col_end;

  }

  if (row_end != y.Size())
  mfem_error("ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const #1");

  if (col_end != x.Size())
  mfem_error("ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const #2");
  }

  extern "C"
  {
  void dgemm_(char *, char *, int *, int *, int *, double *, double *,
  int *, double *, int *, double *, double *, int *);
  }
*/
/*
  void ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const
  {

  if (x.NumberOfVectors() != y.NumberOfVectors())
  mfem_error("ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const #1");

  if (nrows != y.Size())
  {
  std::cout << "y.Size() " << y.Size() << " this->Height() " << nrows << "\n";
  mfem_error("ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const #2");
  }

  if (ncols != x.Size())
  {
  std::cout << "x.Size() " << x.Size() << " this->Width() " << ncols << "\n";
  mfem_error("ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const #3");
  }

  int row_start(0), col_start(0), row_end(0), col_end(0), nLocalRows(0), nLocalCols(0);

  double * xx(x.GetData()), *yy(y.GetData());
  int ldx = x.LeadingDimension();
  int ldy = y.LeadingDimension();

  int nv = x.NumberOfVectors();

  static char transa = 'N', transb = 'N';
  static double alpha = 1.0, beta = 0.0;

  for (int i(0); i <  emat.Size(); ++i)
  {
  nLocalRows = emat[i]->Height();
  nLocalCols = emat[i]->Width();
  row_end = row_start+nLocalRows;
  col_end = col_start+nLocalCols;

  dgemm_(&transa, &transb, &nLocalRows, &nv, &nLocalCols,
  &alpha, emat[i]-> Data(), &nLocalRows, xx+col_start, &ldx,
  &beta, yy+row_start, &ldy);

  //Update
  row_start = row_end;
  col_start = col_end;
  }

  if (row_end != nrows)
  {
  std::cout << "row_end " << row_end << " nrows " << nrows << "\n";
  mfem_error("ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const #4");
  }

  if (col_end != ncols)
  {
  std::cout << "col_end " << col_end << " ncols " << ncols << "\n";
  mfem_error("ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const #4");
  }
  }
*/
void ElementalMatricesContainer::SetElementalMatrix(
    int i, unique_ptr<DenseMatrix> smat)
{
    PARELAG_TEST_FOR_EXCEPTION(
        (bool) emat_[i],
        std::runtime_error,
        "ElementalMatricesContainer::SetElementalMatrix(...):\n"
        "Matrix " << i << " already set. Use ResetElementalMatrix() instead.");

    emat_[i] = std::move(smat);
}

void ElementalMatricesContainer::ResetElementalMatrix(
    int i, unique_ptr<DenseMatrix> smat)
{
    PARELAG_TEST_FOR_EXCEPTION(
        not emat_[i],
        std::runtime_error,
        "ElementalMatricesContainer::ResetElementalMatrix(...):\n"
        "Matrix " << i << " not yet set. Use SetElementalMatrix() instead.");

    emat_[i] = std::move(smat);
}

void ElementalMatricesContainer::SetElementalMatrix(int i, const double & val)
{
    PARELAG_TEST_FOR_EXCEPTION(
        (bool) emat_[i],
        std::runtime_error,
        "ElementalMatricesContainer::SetElementalMatrix(...):\n"
        "Matrix " << i << " already set.");

    emat_[i] = make_unique<DenseMatrix>(1);
    *(emat_[i]) = val;
}

DenseMatrix & ElementalMatricesContainer::GetElementalMatrix(int i)
{
    return *emat_[i];
}
}//namespace parelag
