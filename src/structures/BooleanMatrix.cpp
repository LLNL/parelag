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

#include "elag_structures.hpp"
#include "../linalg/elag_linalg.hpp"

BooleanMatrix::BooleanMatrix(SerialCSRMatrix & A)
:
owner(0),
size(A.Size()),
width(A.Width()),
I(A.GetI()),
J(A.GetJ())
{

}

BooleanMatrix::BooleanMatrix(int * i, int * j, int size_, int width_):
		owner(1),
		size(size_),
		width(width_),
		I(i),
		J(j)
{

}

int BooleanMatrix::Size() const
{
	return size;
}

int BooleanMatrix::Width() const
{
	return width;
}

BooleanMatrix * BooleanMatrix::Transpose() const
{

   int i, j, end;
   int m, n, nnz, *At_i, *At_j;

   m      = size;   // number of rows of A
   n      = width;  // number of columns of A
   nnz    = I[size];

   At_i = new int[n+1];
   At_j = new int[nnz];

   for (i = 0; i <= n; i++)
      At_i[i] = 0;
   for (i = 0; i < nnz; i++)
      At_i[J[i]+1]++;
   for (i = 1; i < n; i++)
      At_i[i+1] += At_i[i];

   for (i = j = 0; i < m; i++)
   {
      end = I[i+1];
      for ( ; j < end; j++)
      {
         At_j[At_i[J[j]]] = i;
         At_i[J[j]]++;
      }
   }

   for (i = n; i > 0; i--)
      At_i[i] = At_i[i-1];
   At_i[0] = 0;

   return  new BooleanMatrix(At_i, At_j, n, m);

}

void BooleanMatrix::SetOwnerShip(int owns)
{
	owner = owns;
}

int BooleanMatrix::MaxRowSize() const
{
	int m = -1, row_size;
	int * it_next = I+1;
	int * end = I+size;
	for(int * it = I; it != end; ++it, ++it_next)
	{
		row_size = *it_next - *it;
		if(m < row_size)
			m = row_size;
	}

	return m;

}

void BooleanMatrix::GetRow(int i, Array<int> & cols)
{
	cols.MakeRef(J+I[i], I[i+1] - I[i] );
}

void BooleanMatrix::GetRow(int i, Array<int> & cols) const
{
	cols.MakeRef(J+I[i], I[i+1] - I[i] );
}

void BooleanMatrix::GetRowCopy(int i, Array<int> & cols) const
{
	cols.SetSize(I[i+1] - I[i]);
	cols.Assign(J+I[i]);
}

SerialCSRMatrix * BooleanMatrix::AsCSRMatrix() const
{
	int nnz = NumNonZeroElems();

	double * a = new double[nnz];

	std::fill(a,a+nnz, 1.);

	return new SerialCSRMatrix(I,J,a,size, width);

}

BooleanMatrix::~BooleanMatrix()
{
	if(owner)
	{
		delete[] I;
		delete[] J;
	}
}

void BooleanMatrix::Mult(const Vector & x, Vector & y) const
{
	elag_assert(x.Size() == width );
	elag_assert(y.Size() == size  );

	const double * xp = x.GetData();
	double * yp = y.GetData();

	int *Jp = J, *Ip = I;
	int i,j, end;
    for (i = j = 0; i < size; i++)
    {
       double d = 0.0;
       for (end = Ip[i+1]; j < end; j++)
          d += xp[Jp[j]];

       yp[i] = d;
    }
}
void BooleanMatrix::MultTraspose(const Vector & x, Vector & y) const
{
	elag_assert(x.Size() == size );
	elag_assert(y.Size() == width  );

	y = 0.;
	const double * xp = x.GetData();
	double * yp = y.GetData();
	int i,j, end;
	for (i = 0; i < size; i++)
	{
		double xi = x(i);
	    end = I[i+1];
	    for(j = I[i]; j < end; j++)
	         yp[J[j]] += xi;
	}
}

void BooleanMatrix::Mult(const MultiVector & x, MultiVector & y) const
{
	elag_assert(x.Size() == width );
	elag_assert(y.Size() == size  );

	int nv = x.NumberOfVectors();
	elag_assert(y.NumberOfVectors() == nv );

	const double * xp;
	double * yp;
	int i,j, end;

	for(int k(0); k < nv; ++k)
	{
		int *Jp = J, *Ip = I;

		xp = x.GetDataFromVector(k);
		yp = y.GetDataFromVector(k);

		for (i = j = 0; i < size; i++)
		{
			double d = 0.0;
			for (end = Ip[i+1]; j < end; j++)
				d += xp[Jp[j]];

			yp[i] = d;
		}
    }
}
void BooleanMatrix::MultTraspose(const MultiVector & x, MultiVector & y) const
{
	elag_assert(x.Size() == size );
	elag_assert(y.Size() == width  );

	int nv = x.NumberOfVectors();
	elag_assert(y.NumberOfVectors() == nv );

	const double * xp;
	double * yp;
	int i,j, end;
	y = 0.0;
	for(int k(0); k < nv; ++k)
	{
		int *Jp = J, *Ip = I;

		xp = x.GetDataFromVector(k);
		yp = y.GetDataFromVector(k);
		for (i = 0; i < size; i++)
		{
			double xi = xp[i];
		    end = I[i+1];
		    for(j = I[i]; j < end; j++)
		         yp[J[j]] += xi;
		}
	}
}
void BooleanMatrix::Mult(const DenseMatrix & x, DenseMatrix & y) const
{
	elag_assert(x.Height() == width );
	elag_assert(y.Height() == size  );

	int nv = x.Width();
	elag_assert(y.Width() == nv );

	const double * xp = x.Data();
	double * yp = y.Data();
	int i,j,end;

	for(int k(0); k < nv; ++k, xp += width, yp += size)
	{
		int *Jp = J, *Ip = I;

		for (i = j = 0; i < size; i++)
		{
			double d = 0.0;
			for (end = Ip[i+1]; j < end; j++)
				d += xp[Jp[j]];

			yp[i] = d;
		}
    }
}
void BooleanMatrix::MultTraspose(const DenseMatrix & x, DenseMatrix & y) const
{
	elag_assert(x.Height() == size );
	elag_assert(y.Height() == width  );

	int nv = x.Width();
	elag_assert(y.Width() == nv );

	const double * xp = x.Data();
	double * yp = y.Data();
	int i,j, end;
	y = 0.0;
	for(int k(0); k < nv; ++k, xp += size, yp += width)
	{
		int *Jp = J, *Ip = I;

		for (i = 0; i < size; i++)
		{
			double xi = xp[i];
		    end = I[i+1];
		    for(j = I[i]; j < end; j++)
		         yp[J[j]] += xi;
		}
	}
}

BooleanMatrix * BoolMult(const BooleanMatrix & A, const BooleanMatrix & B)
{
	   int nrowsA, ncolsA, nrowsB, ncolsB;
	   const int *A_i, *A_j, *B_i, *B_j;
	   int *C_i, *C_j, *B_marker;
	   int ia, ib, ic, ja, jb, num_nonzeros;
	   int row_start, counter;

	   nrowsA = A.Size();
	   ncolsA = A.Width();
	   nrowsB = B.Size();
	   ncolsB = B.Width();

	   if (ncolsA != nrowsB)
	      mfem_error("Sparse matrix multiplication, Mult (...) #1");

	   A_i    = A.GetI();
	   A_j    = A.GetJ();
	   B_i    = B.GetI();
	   B_j    = B.GetJ();

	   B_marker = new int[ncolsB];

	   for (ib = 0; ib < ncolsB; ib++)
	      B_marker[ib] = -1;


	   C_i = new int[nrowsA+1];
	   C_i[0] = num_nonzeros = 0;

	   for (ic = 0; ic < nrowsA; ic++)
	   {
		   for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
	       {
			   ja = A_j[ia];
	           for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
	           {
	              jb = B_j[ib];
	              if (B_marker[jb] != ic)
	              {
	                 B_marker[jb] = ic;
	                 num_nonzeros++;
	              }
	           }
	       }
	      C_i[ic+1] = num_nonzeros;
	    }

	   C_j    = new int[num_nonzeros];

	   for (ib = 0; ib < ncolsB; ib++)
	       B_marker[ib] = -1;


	   counter = 0;
	   for (ic = 0; ic < nrowsA; ic++)
	   {
	      // row_start = C_i[ic];
	      row_start = counter;
	      for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
	      {
	         ja = A_j[ia];
	         for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
	         {
	            jb = B_j[ib];
	            if (B_marker[jb] < row_start)
	            {
	               B_marker[jb] = counter;
	               C_j[counter] = jb;
	               counter++;
	            }
	         }
	      }
	   }

	   delete [] B_marker;

	   return new BooleanMatrix (C_i, C_j,  nrowsA, ncolsB);
}

BooleanMatrix * BoolMult(const SerialCSRMatrix & A, const SerialCSRMatrix & B)
{
	BooleanMatrix a( const_cast<SerialCSRMatrix &>(A) );
	BooleanMatrix b( const_cast<SerialCSRMatrix &>(B) );
	return BoolMult(a,b);
}

BooleanMatrix * BoolMult(const BooleanMatrix & A, const SerialCSRMatrix & B)
{
	BooleanMatrix b( const_cast<SerialCSRMatrix &>(B) );
	return BoolMult(A,b);
}

BooleanMatrix * BoolMult(const SerialCSRMatrix & A, const BooleanMatrix & B)
{
	BooleanMatrix a( const_cast<SerialCSRMatrix &>(A) );
	return BoolMult(a,B);
}

SerialCSRMatrix * Mult(const BooleanMatrix & A, const SerialCSRMatrix & B)
{
	SerialCSRMatrix * Adata = A.AsCSRMatrix();
	SerialCSRMatrix * out = Mult(*Adata, B);
	delete[] Adata->GetData();
	Adata->LoseData();
	delete Adata;
	return out;
}

SerialCSRMatrix * Mult(const SerialCSRMatrix & A, const BooleanMatrix & B)
{
	SerialCSRMatrix * Bdata = B.AsCSRMatrix();
	SerialCSRMatrix * out = Mult(A, *Bdata);
	delete[] Bdata->GetData();
	Bdata->LoseData();
	delete Bdata;
	return out;
}
