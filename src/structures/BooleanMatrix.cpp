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

#include "BooleanMatrix.hpp"

#include "utilities/elagError.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

BooleanMatrix::BooleanMatrix(SerialCSRMatrix & A)
    :
    OwnIJ_(false),
    Size_(A.Size()),
    Width_(A.Width()),
    I_(A.GetI()),
    J_(A.GetJ())
{
}

BooleanMatrix::BooleanMatrix(int * i, int * j, int size, int width):
    OwnIJ_(true),
    Size_(size),
    Width_(width),
    I_(i),
    J_(j)
{
}

int BooleanMatrix::Size() const noexcept
{
    return Size_;
}

int BooleanMatrix::Width() const noexcept
{
    return Width_;
}

unique_ptr<BooleanMatrix> BooleanMatrix::Transpose() const
{
    int i, j, end;
    int *At_i, *At_j;

    const int m = Size_;   // number of rows of A
    const int n = Width_;  // number of columns of A
    const int nnz = I_[Size_];

    At_i = new int[n+1];
    At_j = new int[nnz];

    for (i = 0; i <= n; i++)
        At_i[i] = 0;
    for (i = 0; i < nnz; i++)
        At_i[J_[i]+1]++;
    for (i = 1; i < n; i++)
        At_i[i+1] += At_i[i];

    for (i = j = 0; i < m; i++)
    {
        end = I_[i+1];
        for ( ; j < end; j++)
        {
            At_j[At_i[J_[j]]] = i;
            At_i[J_[j]]++;
        }
    }

    for (i = n; i > 0; i--)
        At_i[i] = At_i[i-1];
    At_i[0] = 0;

    return make_unique<BooleanMatrix>(At_i, At_j, n, m);
}

void BooleanMatrix::SetOwnerShip(bool owns) noexcept
{
    OwnIJ_ = owns;
}

int BooleanMatrix::MaxRowSize() const
{
    int m = -1, row_size;
    int * it_next = I_+1;
    int * end = I_+Size_;
    for(int * it = I_; it != end; ++it, ++it_next)
    {
        row_size = *it_next - *it;
        if(m < row_size)
            m = row_size;
    }

    return m;

}

void BooleanMatrix::GetRow(int i, Array<int> & cols)
{
    cols.MakeRef(J_+I_[i], I_[i+1] - I_[i] );
}

void BooleanMatrix::GetRow(int i, Array<int> & cols) const
{
    cols.MakeRef(J_+I_[i], I_[i+1] - I_[i] );
}

void BooleanMatrix::GetRowCopy(int i, Array<int> & cols) const
{
    cols.SetSize(I_[i+1] - I_[i]);
    cols.Assign(J_+I_[i]);
}

unique_ptr<SerialCSRMatrix> BooleanMatrix::AsCSRMatrix() const
{
    int nnz = NumNonZeroElems();

    double * a = new double[nnz];

    std::fill(a,a+nnz, 1.);

    return make_unique<SerialCSRMatrix>(I_,J_,a,Size_,Width_,false,true,false);
}

BooleanMatrix::~BooleanMatrix()
{
    if(OwnIJ_)
    {
        delete[] I_;
        delete[] J_;
    }
}

void BooleanMatrix::Mult(const Vector & x, Vector & y) const
{
    elag_assert(x.Size() == Width_);
    elag_assert(y.Size() == Size_);

    const double * xp = x.GetData();
    double * yp = y.GetData();

    int i,j, end;
    for (i = j = 0; i < Size_; i++)
    {
        double d = 0.0;
        for (end = I_[i+1]; j < end; j++)
            d += xp[J_[j]];

        yp[i] = d;
    }
}
void BooleanMatrix::MultTranspose(const Vector & x, Vector & y) const
{
    elag_assert(x.Size() == Size_ );
    elag_assert(y.Size() == Width_  );

    y = 0.;

    double * yp = y.GetData();
    for (int i = 0; i < Size_; i++)
    {
        const double xi = x(i);
        for(int j = I_[i]; j < I_[i+1]; j++)
            yp[J_[j]] += xi;
    }
}

void BooleanMatrix::Mult(const MultiVector & x, MultiVector & y) const
{
    elag_assert(x.Size() == Width_ );
    elag_assert(y.Size() == Size_  );

    const int nv = x.NumberOfVectors();
    elag_assert(y.NumberOfVectors() == nv );

    const double * xp;
    double * yp;
    int i,j, end;

    for(int k(0); k < nv; ++k)
    {
        xp = x.GetDataFromVector(k);
        yp = y.GetDataFromVector(k);

        for (i = j = 0; i < Size_; i++)
        {
            double d = 0.0;
            for (end = I_[i+1]; j < end; j++)
                d += xp[J_[j]];

            yp[i] = d;
        }
    }
}
void BooleanMatrix::MultTranspose(const MultiVector & x, MultiVector & y) const
{
    elag_assert(x.Size() == Size_);
    elag_assert(y.Size() == Width_);

    const int nv = x.NumberOfVectors();
    elag_assert(y.NumberOfVectors() == nv );

    const double * xp;
    double * yp;
    int i,j, end;
    y = 0.0;
    for(int k(0); k < nv; ++k)
    {
        xp = x.GetDataFromVector(k);
        yp = y.GetDataFromVector(k);
        for (i = 0; i < Size_; i++)
        {
            double xi = xp[i];
            end = I_[i+1];
            for(j = I_[i]; j < end; j++)
                yp[J_[j]] += xi;
        }
    }
}
void BooleanMatrix::Mult(const DenseMatrix & x, DenseMatrix & y) const
{
    elag_assert(x.Height() == Width_);
    elag_assert(y.Height() == Size_);

    const int nv = x.Width();
    elag_assert(y.Width() == nv );

    const double * xp = x.Data();
    double * yp = y.Data();
    int i,j,end;

    for(int k(0); k < nv; ++k, xp += Width_, yp += Size_)
    {
        for (i = j = 0; i < Size_; i++)
        {
            double d = 0.0;
            for (end = I_[i+1]; j < end; j++)
                d += xp[J_[j]];

            yp[i] = d;
        }
    }
}
void BooleanMatrix::MultTranspose(const DenseMatrix & x, DenseMatrix & y) const
{
    elag_assert(x.Height() == Size_);
    elag_assert(y.Height() == Width_);

    const int nv = x.Width();
    elag_assert(y.Width() == nv );

    const double * xp = x.Data();
    double * yp = y.Data();
    int i,j, end;
    y = 0.0;
    for(int k(0); k < nv; ++k, xp += Size_, yp += Width_)
    {
        for (i = 0; i < Size_; i++)
        {
            double xi = xp[i];
            end = I_[i+1];
            for(j = I_[i]; j < end; j++)
                yp[J_[j]] += xi;
        }
    }
}

unique_ptr<BooleanMatrix> BoolMult(
    const BooleanMatrix & A, const BooleanMatrix & B)
{
    const int *A_i, *A_j, *B_i, *B_j;
    int *C_i, *C_j, *B_marker;
    int ia, ib, ic, ja, jb, num_nonzeros;
    int row_start, counter;

    const int nrowsA = A.Size(),
        ncolsA = A.Width(),
        nrowsB = B.Size(),
        ncolsB = B.Width();

    if (ncolsA != nrowsB)
        mfem_error("Sparse matrix multiplication, Mult (...) #1");

    A_i = A.GetI();
    A_j = A.GetJ();
    B_i = B.GetI();
    B_j = B.GetJ();

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

    C_j = new int[num_nonzeros];

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

    return make_unique<BooleanMatrix>(C_i,C_j,nrowsA,ncolsB);
}

unique_ptr<BooleanMatrix> BoolMult(
    const SerialCSRMatrix & A, const SerialCSRMatrix & B)
{
    BooleanMatrix a( const_cast<SerialCSRMatrix &>(A) );
    BooleanMatrix b( const_cast<SerialCSRMatrix &>(B) );
    return BoolMult(a,b);
}

unique_ptr<BooleanMatrix> BoolMult(
    const BooleanMatrix & A, const SerialCSRMatrix & B)
{
    BooleanMatrix b( const_cast<SerialCSRMatrix &>(B) );
    return BoolMult(A,b);
}

unique_ptr<BooleanMatrix> BoolMult(
    const SerialCSRMatrix & A, const BooleanMatrix & B)
{
    BooleanMatrix a( const_cast<SerialCSRMatrix &>(A) );
    return BoolMult(a,B);
}

unique_ptr<SerialCSRMatrix> Mult(const BooleanMatrix & A,
                                 const SerialCSRMatrix & B)
{
    return unique_ptr<SerialCSRMatrix>{Mult(*(A.AsCSRMatrix()),B)};
}

unique_ptr<SerialCSRMatrix> Mult(const SerialCSRMatrix & A,
                                 const BooleanMatrix & B)
{
    return unique_ptr<SerialCSRMatrix>{Mult(A,*(B.AsCSRMatrix()))};
}
}//namespace parelag
