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

#include "elag_amge.hpp"

#include "../linalg/MatrixUtils.hpp"

ElementalMatricesContainer::ElementalMatricesContainer(int nEntities):
emat(nEntities)
{
	emat = static_cast<DenseMatrix *>(NULL);
}

ElementalMatricesContainer::ElementalMatricesContainer(const ElementalMatricesContainer & orig):
emat(orig.emat.Size())
{
  for(int i(0); i < emat.Size(); ++i)
    emat[i] = new DenseMatrix(*(orig.emat[i]));
}

ElementalMatricesContainer::~ElementalMatricesContainer()
{
	for(int i(0); i < emat.Size(); ++i)
		delete emat[i];
}

int ElementalMatricesContainer::Finalized()
{
	for(int i(0); i < emat.Size(); ++i)
		if(emat[i] == NULL)
			return 0;
	return 1;
}

SparseMatrix * ElementalMatricesContainer::GetAsSparseMatrix()
{

//	double smallEntry(1e-16);
	double smallEntry(0.);

	if( !Finalized() )
		mfem_error(" ElementalMatricesContainer::GetAsSparseMatrix() #1");

	int localNRows(0), localNCols(0), nrows(0), ncols(0), maxLocalNRows(0), maxLocalNCols(0);

	for(int i(0); i < emat.Size(); ++i)
	{
		localNRows = emat[i]->Height();
		localNCols = emat[i]->Width();

		nrows += localNRows;
		ncols += localNCols;

		if(localNRows > maxLocalNRows)
			maxLocalNRows = localNRows;

		if(localNCols > maxLocalNCols)
			maxLocalNCols = localNCols;
	}

	SparseMatrix * out = new SparseMatrix(nrows, ncols);

	Array<int> row_ind(maxLocalNRows), col_ind(maxLocalNCols);

	int row_ind_start(0), col_ind_start(0);

	for(int i(0); i < emat.Size(); ++i)
	{
		localNRows = emat[i]->Height();
		localNCols = emat[i]->Width();

		row_ind.SetSize(localNRows);
		for(int irow(0); irow < localNRows; ++irow)
			row_ind[irow] = (row_ind_start++);

		col_ind.SetSize(localNCols);
		for(int icol(0); icol < localNCols; ++icol)
			col_ind[icol] = (col_ind_start++);

		double * data = emat[i]->Data();
		for(double * end = data+(localNRows*localNCols); data != end; ++data)
		{
			if(fabs(*data) < smallEntry)
				*data = 0.0;
		}

		out->SetSubMatrix(row_ind, col_ind, *(emat[i]) );

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

	for(int i(0); i <  emat.Size(); ++i)
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

	if(row_end != y.Size())
		mfem_error("ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const #1");

	if(col_end != x.Size())
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

	if( x.NumberOfVectors() != y.NumberOfVectors() )
		mfem_error("ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const #1");

	if(nrows != y.Size())
	{
		std::cout << "y.Size() " << y.Size() << " this->Height() " << nrows << "\n";
		mfem_error("ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const #2");
	}

	if(ncols != x.Size())
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

	for(int i(0); i <  emat.Size(); ++i)
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

	if( row_end != nrows)
	{
		std::cout << "row_end " << row_end << " nrows " << nrows << "\n";
		mfem_error("ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const #4");
	}

	if( col_end != ncols)
	{
		std::cout << "col_end " << col_end << " ncols " << ncols << "\n";
		mfem_error("ElementalMatricesContainer::Mult(const MultiVector & x, MultiVector & y) const #4");
	}
}
*/
void ElementalMatricesContainer::SetElementalMatrix(int i, DenseMatrix * smat)
{
	if( emat[i] != NULL)
		mfem_error("SetElementalMatrix(int i, DenseMatrix * smat) #1");

	emat[i] = smat;
}

void ElementalMatricesContainer::ResetElementalMatrix(int i, DenseMatrix * smat)
{
	elag_assert( emat[i] != NULL);
	delete emat[i];
	emat[i] = smat;
}

void ElementalMatricesContainer::SetElementalMatrix(int i, const double & val)
{
	if( emat[i] != NULL)
		mfem_error("SetElementalMatrix(int i, DenseMatrix * smat) #1");
        
	emat[i] = new DenseMatrix(1);
        *(emat[i]) = val;
}

DenseMatrix & ElementalMatricesContainer::GetElementalMatrix(int i)
{
	return *emat[i];
}
