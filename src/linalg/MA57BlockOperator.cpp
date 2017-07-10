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

#include "elag_linalg.hpp"

// prototypes of MA57 routines
extern "C"
{
  void ma57id_(double * cntl, int * icntl);
  void ma57ad_(int * n, int * ne, int * irn, int * jcn, int * lkeep, int * keep, int * iwork, int * icntl, int * info, double *rinfo);
  void ma57bd_(int * n, int * ne, double *a, double *fact, int *lfact, int *ifact, int *lifact, int *lkeep, int *keep,   int *iwork, int *icntl, double *cntl, int *info, double *rinfo);
  void ma57cd_( int *job, int *n, double *fact, int *lfact, int *ifact, int *lifact,int *nrhs, double *rhs, int *lrhs, double *work, int *lwork, int *iwork, int *icntl, int *info);
}

MA57BlockOperator::MA57BlockOperator(const int nBlocks_):
	Solver(),
	nBlocks(nBlocks_),
	isComputed(false),
	Aij(nBlocks, nBlocks),
	offsets(nBlocks+1),
	iRow(0),
	jCol(0),
	val(0),
	info(40),
	rinfo(20),
	cntl(5),
	icntl(20),
	keep(0),
	fact(0),
	ifact(0),
	iwork(0)
{
	info = 0;
	rinfo = 0.0;
	cntl = 0.0;
	icntl = 0;
	keep = 0;
	fact = 0.0;
	ifact = 0;
	iwork = 0;

	//  Set default values for control parameters.
	ma57id_ ( cntl.GetData(), icntl.GetData() );
	// set verbosity level;
	icntl[4] = 2;
	cntl[1] = 1.e-10;

	opt_ordering = AMD_ORDERING;

	std::fill(Aij[0], Aij[0]+(nBlocks*nBlocks), (SparseMatrix *)NULL);

}

void MA57BlockOperator::SetBlock(int i, int j, const SparseMatrix & Aij_)
{
	if(j > i) mfem_error("MA57BlockOperator::SetBlock: only lower triangular part of A should be given");
	Aij(i,j) = &Aij_;

	if( i == j && Aij_.Size() != Aij_.Width() )
		mfem_error("MA57BlockOperator::SetBlock: diagonal block is not square! \n");
}

int MA57BlockOperator::Compute()
{

	int numFactInfo(0);

	computeOffsets();
	matrix2lowerTriple();
	symbolicFactorization();
	numFactInfo = numericFactorization();

	isComputed = (0 <= numFactInfo);

	return numFactInfo;

}

void MA57BlockOperator::Mult (const Vector & x, Vector & y) const
{
    MFEM_VERIFY(isComputed, "");
    MFEM_VERIFY( x.Size() == y.Size(), "MA57BlockOperator::Mult x and y are of incompatible size");
    MFEM_VERIFY( x.Size() == height, "MA57BlockOperator::Mult x and this are of incompatible size");

	solve(&x[0], &y[0],1);
}

void MA57BlockOperator::Mult (const MultiVector & x, MultiVector & y) const
{
	MFEM_VERIFY(isComputed, "");
	MFEM_VERIFY( x.NumberOfVectors() == y.NumberOfVectors(), "MA57BlockOperator::Mult x and y are of incompatible number of vectors");
	MFEM_VERIFY( x.Size() == y.Size(), "MA57BlockOperator::Mult x and y are of incompatible size");
	MFEM_VERIFY( x.Size() == height, "MA57BlockOperator::Mult x and this are of incompatible size");
	MFEM_VERIFY(x.LeadingDimension() == x.Size() && y.LeadingDimension() == y.Size(), "MA57BlockOperator::Mult LDA not supported");

	solve(&x[0], &y[0], x.NumberOfVectors());
}

void MA57BlockOperator::Mult (const DenseMatrix & x, DenseMatrix & y) const
{
    MFEM_VERIFY(isComputed, "");
    MFEM_VERIFY( x.Height() == y.Height(), "MA57BlockOperator::Mult x and y are of incompatible height");
	MFEM_VERIFY( x.Width() == y.Width(), "MA57BlockOperator::Mult x and y are of incompatible width");
	MFEM_VERIFY( x.Height() == height, "MA57BlockOperator::Mult x and this are of incompatible size");

	assert(isComputed);
	solve( x.Data(), y.Data(), x.Width() );
}

//------------------------------------

void MA57BlockOperator::computeOffsets()
{
	offsets[0] = 0;
	for(int i(0); i<nBlocks; ++i)
	{
		for(int j(0); j<nBlocks; ++j)
		{
			if(Aij(i,j))
			{
				offsets[i+1] = offsets[i]+Aij(i,j)->Size();
				break;
			}
			if(j == nBlocks)
			{
				std::cout<<" All blocks in row " << i << " are pointers to NULL\n";
				mfem_error();
			}
		}
	}

	height = width = offsets[nBlocks];
}

int MA57BlockOperator::computeNnz()
{
	int nnz(0);
	int blockSize(0);
	int blockNnz(0);

	//I start with the diagonal blocks
	for(int i(0); i<nBlocks; ++i)
	{
		if(Aij(i,i))
		{
			blockNnz = (Aij(i,i)->NumNonZeroElems());
			blockSize = (Aij(i,i)->Size());
			nnz += (blockNnz-blockSize)/2 + blockSize;
		}
	}

	//Then I count the lower triangular part
	for(int i(1); i<nBlocks; ++i)
		for(int j(0); j<i; ++j)
		{
			if(Aij(i,j))
				nnz += Aij(i,j)->NumNonZeroElems();
		}

	return nnz;
}


void MA57BlockOperator::matrix2lowerTriple()
{
	int nnz( computeNnz() );

	iRow.reserve(nnz);
	jCol.reserve(nnz);
	val.reserve(nnz);

	// I do first the diagonal blocks
	for(int i(0); i<nBlocks; ++i)
	{
		if(Aij(i,i))
		{
			int * A_I = Aij(i,i)->GetI();
			int * A_J = Aij(i,i)->GetJ();
			double * A_V = Aij(i,i)->GetData();
			int localSize = Aij(i,i)->Size();

			for (int row(0); row<localSize; ++row)
				for (int colind(A_I[row]); colind<A_I[row+1]; ++colind)
				{
					int col=A_J[colind];
					if (col>row)
						continue;
					iRow.push_back(row+1+offsets[i]); //iRow index starts from 1, while row starts from 0
					jCol.push_back(col+1+offsets[i]);
					val.push_back(A_V[colind]);
				}
		}
	}

	// Then I do the off-diagonals ones
	for(int i(1); i<nBlocks; ++i)
		for(int j(0); j<i; ++j)
		{
			if(Aij(i,j))
			{
				int * A_I = Aij(i,j)->GetI();
				int * A_J = Aij(i,j)->GetJ();
				double * A_V = Aij(i,j)->GetData();
				int localSize = Aij(i,j)->Size();
				for (int row(0); row<localSize; ++row)
					for (int colind(A_I[row]); colind<A_I[row+1]; ++colind)
					{
						int col=A_J[colind];
						iRow.push_back(row+1+offsets[i]); //iRow index starts from 1, while row starts from 0
						jCol.push_back(col+1+offsets[j]);
						val.push_back(A_V[colind]);
					}

			}
		}

	if( static_cast<int>(val.size()) > nnz )
	{
		std::cout << "Actual number of nnz is "<< val.size() << " Estimate number of nnz is " << nnz << "\n";

		for(int i(0); i<nBlocks; ++i)
			if(Aij(i,i))
				Aij(i,i)->PrintMatlab( std::cout<<" A_ " << i << i << "\n" );

		mfem_error("matrix2lowerTriple()");
	}
}


int MA57BlockOperator::symbolicFactorization()
{
	// Analyse sparsity pattern
	int m_size(this->height);
	int nnz(val.size());

	int lkeep= 5*m_size+nnz+std::max(m_size,nnz)+42  + 2*m_size;
	keep.SetSize(lkeep);

	iwork.SetSize(5*m_size);

	switch(opt_ordering)
	{
	case METIS_ORDERING:
		icntl[5] = 4;
		break;
	case AMD_ORDERING:
		icntl[5] = 2;
		break;
	default:
		mfem_error("Unknown Ordering method");
	}

	ma57ad_ (&m_size,&nnz,&iRow[0],&jCol[0],&lkeep,keep.GetData(),iwork.GetData(),icntl.GetData(),info.GetData(),rinfo.GetData());

	return info[0];
}

int MA57BlockOperator::numericFactorization()
{
	//  Factorize matrix
	int nnz(val.size());
	int lkeep(keep.Size());

	int lfact=2*info[8];
	int lifact=int(info[9]*1.1);
	fact.SetSize(lfact);
	ifact.SetSize(lifact);

	ma57bd_ (&(this->height),&nnz,&val[0],
			 fact.GetData(),&lfact,ifact.GetData(),&lifact,&lkeep,keep.GetData(),
			 iwork.GetData(),icntl.GetData(),cntl.GetData(),info.GetData(),rinfo.GetData());
	assert(info[0]>=0);

	return info[0];
}

int MA57BlockOperator::solve(const double * x, double * y, int nrhs) const
{
	// Solve the equations
	if(y!=x)
		std::copy(x, x+nrhs*this->height, y);

	int m_size(this->height);
	int lfact(fact.Size());
	int lifact(ifact.Size());

	int job = 1;

	int lwork=this->height*nrhs;
	std::vector<double> work(lwork);

	ma57cd_(&job,&m_size,fact.GetData(),&lfact,ifact.GetData(),&lifact,
			&nrhs,y,&m_size,
			&work[0],&lwork,iwork.GetData(),icntl.GetData(),info.GetData());
	assert(info[0]>=0);

	return info[0];
}
