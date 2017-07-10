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

#ifndef SUBMATRIXEXTRACTION_HPP_
#define SUBMATRIXEXTRACTION_HPP_

SparseMatrix * ExtractRowAndColumns(const SparseMatrix & A, const Array<int> & rows, const Array<int> & cols, Array<int> & colMapper);
BlockMatrix * ExtractRowAndColumns(const BlockMatrix * A, const Array<int> & gRows, const Array<int> & gCols, Array<int> & colMapper);
SparseMatrix * ExtractColumns(const SparseMatrix & A, const Array<int> & cols, Array<int> & colMapper);
SparseMatrix * ExtractRowAndColumns(const SparseMatrix & A, const Array<int> & rows, const Array<int> & cols);
SparseMatrix * ExtractSubMatrix(const SparseMatrix & A, const int rowStart, const int rowEnd, const int colStart, const int colEnd);
void ExtractSubMatrices(const SparseMatrix & A, const Table & subMat_row, const Table & subMat_col, Array<SparseMatrix *> & subMats);

void ExtractComponents(const SparseMatrix & A, Array2D<SparseMatrix *> & Comp, int ordering);
void ExtractComponentsByVDIM(const SparseMatrix & A, Array2D<SparseMatrix *> & Comp);
void ExtractComponentsByNODES(const SparseMatrix & A, Array2D<SparseMatrix *> & Comp);

SparseMatrix * Distribute(const SparseMatrix & A, const SparseMatrix & AE_row, const SparseMatrix & AE_col);

//Each row is a Vector in the multivector V
void GetRows(const SparseMatrix & A, const Array<int> & rows, const int colStart, const int colEnd, MultiVector & V);
void GetRows(const SparseMatrix & A, const Array<int> & rows, const double * rowSign, const int colStart, const int colEnd, MultiVector & V);


class DropEntryPredicate
{
public:
	virtual ~DropEntryPredicate(){ };
	virtual bool operator()(int row, int col, double val) const = 0;

};

class DropEntryAccordingToColumnMarker : public DropEntryPredicate
{
public:
	DropEntryAccordingToColumnMarker(const Array<int> & columnMarker_, int labelToKeep_):columnMarker(columnMarker_), labelToKeep(labelToKeep_){};
	void SetLabel(int labelToKeep_){labelToKeep = labelToKeep_;}
	virtual bool operator()(int row, int col, double val) const {return columnMarker[col] != labelToKeep;}

private:
	const Array<int> & columnMarker;
	int labelToKeep;
};

class DropEntryAccordingToColumnMarkerAndId : public DropEntryPredicate
{
public:
	DropEntryAccordingToColumnMarkerAndId(const Array<int> & columnMarker_, int labelToKeep_, int smallestCol):
		columnMarker(columnMarker_),labelToKeep(labelToKeep_),
		smallestColumn(smallestCol)
		{};
	void SetLabel(int labelToKeep_){labelToKeep = labelToKeep_;}
	virtual bool operator()(int row, int col, double val) const {return columnMarker[col] != labelToKeep || col < smallestColumn;}

private:
	const Array<int> & columnMarker;
	int labelToKeep;
	int smallestColumn;
};

class DropEntryAccordingToId : public DropEntryPredicate
{
public:
	DropEntryAccordingToId(int smallestCol):
		smallestColumn(smallestCol)
		{};
	virtual bool operator()(int row, int col, double val) const {return (col < smallestColumn);}

private:
	int smallestColumn;
};


SparseMatrix * DropEntriesFromSparseMatrix(const SparseMatrix & A, const DropEntryPredicate & drop);

#endif
