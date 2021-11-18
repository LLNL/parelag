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
#include <numeric>

#include "TopologyTable.hpp"

#include "elag_typedefs.hpp"
#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "linalg/utilities/ParELAG_SubMatrixExtraction.hpp"
#include "utilities/elagError.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

TopologyTable::TopologyTable(SerialCSRMatrix & A_in)
    : SerialCSRMatrix()
{
    elag_assert(A_in.Finalized());
    this->Swap(A_in);
}

TopologyTable::TopologyTable(std::unique_ptr<SerialCSRMatrix> A_in)
    : SerialCSRMatrix()
{
    elag_assert(A_in->Finalized());
    this->Swap(*A_in);
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
    constexpr double tol = 1e-10;

    const int nnz = NumNonZeroElems();
    double * it = GetData();
    for(double * end = it+nnz; it != end; ++it)
    {
        elag_assert( fabs(*it) > tol );
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

unique_ptr<TopologyTable> TopologyTable::GetSubTable(
    const Array<int> & rows,const Array<int> & cols,Array<int> & marker)
{
    return make_unique<TopologyTable>(
        ExtractRowAndColumns(*this,rows,cols,marker));
}

unique_ptr<TopologyTable> TopologyTable::Transpose() const
{
    return make_unique<TopologyTable>(ToUnique(mfem::Transpose(*this)));
}

unique_ptr<TopologyTable> MultOrientation(
    const TopologyTable & A,
    const TopologyTable & B )
{
    auto out = make_unique<TopologyTable>(ToUnique(Mult(A,B)));

    out->DropSmallEntries(1e-10);
    out->OrientationTransform();
    return out;
}

unique_ptr<TopologyTable> MultBoolean(
    const TopologyTable & A,
    const TopologyTable & B )
{
    const int *A_i, *A_j, *B_i, *B_j;
    int *C_i, *C_j;
    int ia, ib, ic, ja, jb, num_nonzeros;

    const int nrowsA = A.Size();
    const int ncolsA = A.Width();
    const int nrowsB = B.Size();
    const int ncolsB = B.Width();

    PARELAG_TEST_FOR_EXCEPTION(
        ncolsA != nrowsB,
        std::logic_error,
        "MultBoolean(): "
        "Column dimension of A does not match row dimension of B!");

    A_i = A.GetI();
    A_j = A.GetJ();
    B_i = B.GetI();
    B_j = B.GetJ();

    std::vector<int> B_marker(ncolsB,-1);

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

    C_j = new int[num_nonzeros];
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

    return make_unique<TopologyTable>(C_i, C_j, C_data, nrowsA, ncolsB);
}

unique_ptr<TopologyTable>
TransposeOrientation(const Array<int> & j,int nrowsOut)
{
    if (j.Size() > 0)
    {
        elag_assert(j.Max() < nrowsOut );
    }

    int * const i_out = new int[nrowsOut+2];
    std::fill(i_out, i_out + nrowsOut+2, 0);
    int * counts = i_out + 2;
    for(int i = 0; i < j.Size(); ++i)
    {
        elag_assert(j[i] >= -1);
        ++(counts[j[i]]);
    }
    i_out[1] = 0;
    std::partial_sum(i_out, i_out + nrowsOut+2, i_out);

    const int nnz = i_out[nrowsOut+1];
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

    return make_unique<TopologyTable>(i_out, j_out, a_out, nrowsOut, j.Size());
}
}//namespace parelag
