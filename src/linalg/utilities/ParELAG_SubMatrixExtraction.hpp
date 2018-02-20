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

#ifndef SUBMATRIXEXTRACTION_HPP_
#define SUBMATRIXEXTRACTION_HPP_

#include <memory>
#include <vector>

#include <mfem.hpp>

#include "linalg/dense/ParELAG_MultiVector.hpp"

namespace parelag
{

std::unique_ptr<mfem::SparseMatrix> ExtractRowAndColumns(
    const mfem::SparseMatrix & A,
    const mfem::Array<int> & rows,
    const mfem::Array<int> & cols,
    mfem::Array<int> & colMapper );

std::unique_ptr<mfem::BlockMatrix> ExtractRowAndColumns(
    const mfem::BlockMatrix * A,
    const mfem::Array<int> & gRows,
    const mfem::Array<int> & gCols,
    mfem::Array<int> & colMapper );

std::unique_ptr<mfem::SparseMatrix> ExtractColumns(
    const mfem::SparseMatrix & A,
    const mfem::Array<int> & cols,
    mfem::Array<int> & colMapper );

std::unique_ptr<mfem::SparseMatrix> ExtractRowAndColumns(
    const mfem::SparseMatrix & A,
    const mfem::Array<int> & rows,
    const mfem::Array<int> & cols );

std::unique_ptr<mfem::SparseMatrix> ExtractSubMatrix(
    const mfem::SparseMatrix & A,
    const int rowStart,
    const int rowEnd,
    const int colStart,
    const int colEnd );

void ExtractSubMatrices(
    const mfem::SparseMatrix & A,
    const mfem::Table & subMat_row,
    const mfem::Table & subMat_col,
    std::vector<std::unique_ptr<mfem::SparseMatrix>> & subMats );

void ExtractComponents(const mfem::SparseMatrix & A,
                       mfem::Array2D<mfem::SparseMatrix *> & Comp,
                       int ordering);

void ExtractComponentsByVDIM(const mfem::SparseMatrix & A,
                             mfem::Array2D<mfem::SparseMatrix *> & Comp);

void ExtractComponentsByNODES(const mfem::SparseMatrix & A,
                              mfem::Array2D<mfem::SparseMatrix *> & Comp);

std::unique_ptr<mfem::SparseMatrix> Distribute(
    const mfem::SparseMatrix & A,
    const mfem::SparseMatrix & AE_row,
    const mfem::SparseMatrix & AE_col);

/**
   Takes some rows of the matrix A, between columns colStart and colEnd, and
   puts them in the MultiVector V, where each *row* becomes a *vector*
   in the MultiVector.

   I (ATB) fell like a Tranpose is going on under the hood here: it is the rows
   of A that get put in (column?) vectors of V.
*/
void GetRows(const mfem::SparseMatrix & A,
             const mfem::Array<int> & rows,
             const int colStart,
             const int colEnd,
             MultiVector & V);

class DropEntryPredicate
{
public:
    virtual ~DropEntryPredicate(){ };
    virtual bool operator()(int row, int col, double val) const = 0;
};

class DropEntryAccordingToColumnMarker : public DropEntryPredicate
{
public:
    DropEntryAccordingToColumnMarker(const mfem::Array<int> & columnMarker,
                                     int labelToKeep)
        : ColumnMarker_(columnMarker),
          LabelToKeep_(labelToKeep)
    {
    };

    void SetLabel(int labelToKeep){LabelToKeep_ = labelToKeep;}

    virtual bool operator()(int, int col, double) const override
    {return ColumnMarker_[col] != LabelToKeep_;}

private:
    const mfem::Array<int> & ColumnMarker_;
    int LabelToKeep_;
};

class DropEntryAccordingToColumnMarkerAndId : public DropEntryPredicate
{
public:
    DropEntryAccordingToColumnMarkerAndId(const mfem::Array<int> & columnMarker,
                                          int labelToKeep,
                                          int smallestCol)
        : ColumnMarker_(columnMarker),LabelToKeep_(labelToKeep),
          SmallestColumn_(smallestCol)
    {
    };

    void SetLabel(int labelToKeep){LabelToKeep_ = labelToKeep;}

    virtual bool operator()(int, int col, double) const override
    {return ColumnMarker_[col] != LabelToKeep_ || col < SmallestColumn_;}

private:
    const mfem::Array<int> & ColumnMarker_;
    int LabelToKeep_;
    int SmallestColumn_;
};

class DropEntryAccordingToId : public DropEntryPredicate
{
public:
    DropEntryAccordingToId(int smallestCol)
        : SmallestColumn_(smallestCol)
    {
    };

    virtual bool operator()(int, int col, double) const override
    {return (col < SmallestColumn_);}

private:
    int SmallestColumn_;
};

class DropZerosFromSparseMatrix : public DropEntryPredicate
{
public:
    DropZerosFromSparseMatrix() {}
    bool operator()(int, int, double val) const override
    {
        if(val==0.0) return true;
        return false;
    }
};

std::unique_ptr<mfem::SparseMatrix> DropEntriesFromSparseMatrix(
    const mfem::SparseMatrix & A, const DropEntryPredicate & drop);

}//namespace parelag
#endif
