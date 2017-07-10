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

#include "elag_topology.hpp"
#include <numeric>

TopologyTable::TopologyTable(SerialCSRMatrix & A):
	SerialCSRMatrix(NULL,NULL,NULL,0,0)
{
	elag_assert(A.Finalized());
	Swap(*this, A);
}

TopologyTable::TopologyTable(SerialCSRMatrix * A):
	SerialCSRMatrix(NULL,NULL,NULL,0,0)
{
	elag_assert(A->Finalized());
	Swap(*this, *A);
	delete A;
}


TopologyTable::TopologyTable(int *i, int *j, double *data, int m, int n):
	SerialCSRMatrix(i,j,data,m,n)
{

}

int SafeWidth(const Table & T)
{
	int nnz = T.Size_of_connections();
	const int * j = T.GetJ();

	int w = 0;
	int jval;
	for(int i = 0; i < nnz; ++i)
	{
		jval = j[i];
		if( jval < 0 )
			jval = -jval-1;
		if(jval > w)
			w = jval;
	}
	return w;
}

TopologyTable::TopologyTable(const Table & T):
		SerialCSRMatrix(new int[T.Size()+1],
				new int[T.Size_of_connections()],
				new double[T.Size_of_connections()],
				T.Size(),
				SafeWidth(T) )
{
	std::copy(T.GetI(), T.GetI()+height+1, GetI() );

	int * my_j = GetJ();
	double * o = GetData();

	int nnz = T.Size_of_connections();
	const int * j_T = T.GetJ();

	int jval;
	for(int i = 0; i < nnz; ++i)
	{
		jval = j_T[i];
		if( jval < 0 )
		{
			my_j[i] = -jval-1;
			o[i] = -1.0;
		}
		else
		{
			my_j[i] = jval;
			o[i] = 1.0;
		}
	}
}

void TopologyTable::OrientationTransform()
{
	double tol = 1e-10;

	int nnz = NumNonZeroElems();
	double * it = GetData();
	for(double * end = it+nnz; it != end; ++it)
	{
		elag_assert( fabs(*it) > tol )
		if(fabs(*it) < tol)
			*it = 0.;
		else
			*it = (*it > 0) ? 1.:-1.;
	}
}

void TopologyTable::DropSmallEntries(double tol)
{
	dropSmallEntry(*this, tol);
}

void TopologyTable::WedgeMult(const Vector & x, Vector & y)
{
	elag_assert(Size() == y.Size() );
	elag_assert(Width() == x.Size() );
	wedgeMult(x.GetData(), y.GetData());
}

void TopologyTable::WedgeMult(const Array<int> & x, Array<int> & y)
{
	elag_assert(Size() == y.Size() );
	elag_assert(Width() == x.Size() );
	wedgeMult(x.GetData(), y.GetData());
}

void TopologyTable::WedgeMultTranspose(const Vector & x, Vector & y)
{
	elag_assert(Size() == x.Size() );
	elag_assert(Width() == y.Size() );
	wedgeMultTranspose(x.GetData(), y.GetData());
}

void TopologyTable::WedgeMultTranspose(const Array<int> & x, Array<int> & y)
{
	elag_assert(Size() == x.Size() );
	elag_assert(Width() == y.Size() );
	wedgeMultTranspose(x.GetData(), y.GetData());
}

TopologyTable * TopologyTable::GetSubTable(const Array<int> & rows, const Array<int> & cols, Array<int> & marker)
{
	return new TopologyTable(ExtractRowAndColumns(*this, rows,cols, marker));
}

TopologyTable * TopologyTable::Transpose()
{
	return new TopologyTable( ::Transpose(*this) );
}

TopologyTable::~TopologyTable()
{

}

TopologyTable * MultOrientation(const TopologyTable & A, const TopologyTable & B)
{
	TopologyTable * out = new TopologyTable( Mult(const_cast<TopologyTable &>(A) , const_cast<TopologyTable &>(B) ) );
	out->DropSmallEntries(1e-10);
	out->OrientationTransform();
	return out;
}

TopologyTable * MultBoolean(const TopologyTable & A, const TopologyTable & B)
{
	int nrowsA, ncolsA, nrowsB, ncolsB;
	int *A_i, *A_j, *B_i, *B_j, *C_i, *C_j, *B_marker;
	int ia, ib, ic, ja, jb, num_nonzeros;

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

   for (ib = 0; ib < ncolsB; ib++)
	   B_marker[ib] = -1;

   C_j    = new int[num_nonzeros];
   double * C_data = new double[num_nonzeros];
   std::fill(C_data, C_data+num_nonzeros, 1.);

   int counter = 0, row_start;
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

   delete[] B_marker;

   return new TopologyTable(C_i, C_j, C_data, nrowsA, ncolsB);
}

TopologyTable * TransposeOrientation(const Array<int> & j, int nrowsOut)
{
	elag_assert(j.Max() < nrowsOut );

	int * const i_out = new int[nrowsOut+2];
	std::fill(i_out, i_out + nrowsOut+2, 0);
	int * counts = i_out + 2;
	for(int i = 0; i < j.Size(); ++i)
	{
		elag_assert(j[i] >= -1)
		++(counts[j[i]]);
	}
	i_out[1] = 0;
	std::partial_sum(i_out, i_out + nrowsOut+2, i_out);

	int nnz = i_out[nrowsOut+1];
	int * j_out = new int[nnz];
	double * a_out = new double[nnz];
	std::fill(a_out, a_out+nnz, 1.);

	counts = i_out + 1;
	int irow;
	for(int i = 0; i < j.Size(); ++i)
	{
		if( (irow = j[i]) > -1 )
		{
			j_out[ counts[irow] ] = i;
			counts[irow]++;
		}
	}

	return new TopologyTable(i_out, j_out, a_out, nrowsOut, j.Size() );
}

