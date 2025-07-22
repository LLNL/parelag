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

#include "linalg/utilities/ParELAG_MatrixUtils.hpp"

#include "hypreExtension/hypreExtension.hpp"
#include "utilities/elagError.hpp"
#include "utilities/HypreTraits.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{
using mfem::Array;
using mfem::BlockMatrix;
using mfem::DenseMatrix;
using mfem::SparseMatrix;
using mfem::HypreParMatrix;
using mfem::Vector;
using std::unique_ptr;

bool AreAlmostEqual(const SparseMatrix & A,
                    const SparseMatrix & B,
                    const std::string & Aname,
                    const std::string & Bname,
                    double tol,
                    bool verbose,
                    std::ostream & os)
{
    const int nrows = A.Size();
    const int ncols = A.Width();
    const int brows = B.Size();
    const int bcols = B.Width();

    PARELAG_TEST_FOR_EXCEPTION(
        nrows != brows || ncols != bcols,
        std::logic_error,
        "AreAlmostEqual(): "
        "Size(" << Aname << ") = " << nrows << "x" << nrows << std::endl <<
        "Size(" << Bname << ") = " << brows << "x" << bcols << std::endl <<
        "Sizes don't match!" );

    const int * i_A = A.GetI();
    const int * j_A = A.GetJ();
    const double * v_A = A.GetData();

    const int * i_B = B.GetI();
    const int * j_B = B.GetJ();
    const double * v_B = B.GetData();

    Vector rowVals(ncols);
    rowVals = 0.0;

    int ndiff(0);
    double maxErr(0.0), mydiff(0);

    for(int irow(0); irow < nrows; ++irow)
    {
        for(int jpos = i_A[irow]; jpos < i_A[irow+1]; ++ jpos)
            rowVals[ j_A[jpos] ] = v_A[jpos];

        for(int jpos = i_B[irow]; jpos < i_B[irow+1]; ++ jpos)
            rowVals[ j_B[jpos] ] -= v_B[jpos];

        for(int jpos = i_A[irow]; jpos < i_A[irow+1]; ++ jpos)
        {
            mydiff = fabs( rowVals[ j_A[jpos] ] );

            if( mydiff > tol)
                ++ndiff;
            if( mydiff > maxErr)
                maxErr = mydiff;

            rowVals[ j_A[jpos] ] = 0.0;
        }

        for(int jpos = i_B[irow]; jpos < i_B[irow+1]; ++ jpos)
        {
            mydiff = fabs( rowVals[ j_B[jpos] ] );
            if(mydiff  > tol)
                ++ndiff;

            if( mydiff > maxErr)
                maxErr = mydiff;

            rowVals[ j_B[jpos] ] = 0.0;
        }
    }

    if(mydiff > tol || verbose)
    {
        os << "    nnz(" << Aname << " - " << Bname << ") = " << ndiff
           << std::endl
           << "normInf(" << Aname << " - " << Bname << ") = " << maxErr
           << std::endl;
    }

    return (mydiff <= tol);
}


bool AreAlmostEqual(const SparseMatrix & A,
                    const SparseMatrix & B,
                    const SparseMatrix & G,
                    const std::string & Aname,
                    const std::string & Bname,
                    const std::string & Gname,
                    double tol,
                    bool verbose,
                    std::ostream & os)
{
    const int nrows = A.Size();
    const int ncols = A.Width();
    const int brows = B.Size();
    const int bcols = B.Width();
    const int grows = G.Size();
    const int gcols = G.Width();

    PARELAG_TEST_FOR_EXCEPTION(
        nrows != brows || ncols != bcols || grows != nrows || gcols != ncols,
        std::logic_error,
        "AreAlmostEqual(): " <<
        "Size(" << Aname << ") = " << nrows << "x" << ncols << std::endl <<
        "Size(" << Bname << ") = " << brows << "x" << bcols << std::endl <<
        "Size(" << Gname << ") = " << grows << "x" << gcols << std::endl <<
        "Sizes don't match!" );

    const int * i_A = A.GetI();
    const int * j_A = A.GetJ();
    const double * v_A = A.GetData();

    const int * i_B = B.GetI();
    const int * j_B = B.GetJ();
    const double * v_B = B.GetData();

    const int * i_G = G.GetI();
    const int * j_G = G.GetJ();

    Vector rowVals(ncols);
    rowVals = 0.0;

    int ndiff(0);
    double maxErr(0.0), mydiff(0);

    for(int irow(0); irow < nrows; ++irow)
    {
        for(int jpos = i_A[irow]; jpos < i_A[irow+1]; ++ jpos)
            rowVals[ j_A[jpos] ] = v_A[jpos];

        for(int jpos = i_B[irow]; jpos < i_B[irow+1]; ++ jpos)
            rowVals[ j_B[jpos] ] -= v_B[jpos];

        for(int jpos = i_G[irow]; jpos < i_G[irow+1]; ++ jpos)
        {
            mydiff = fabs( rowVals[ j_G[jpos] ] );

            if( mydiff > tol)
                ++ndiff;
            if( mydiff > maxErr)
                maxErr = mydiff;
        }

        //Restore rowVals = 0
        for(int jpos = i_A[irow]; jpos < i_A[irow+1]; ++ jpos)
            rowVals[ j_A[jpos] ] = 0.0;


        for(int jpos = i_B[irow]; jpos < i_B[irow+1]; ++ jpos)
            rowVals[ j_B[jpos] ] = 0.0;
    }

    if(mydiff > tol || verbose)
    {
        os << "Restrict the comparison on the sparsisty pattern of " << Gname
           << std::endl
           << "    nnz(" << Aname << " - " << Bname << ") = " << ndiff
           << std::endl
           << "normInf(" << Aname << " - " << Bname << ") = " << maxErr
           << std::endl;
    }

    return (mydiff <= tol);
}

bool IsAlmostIdentity(const SparseMatrix & A, double tol, bool verbose )
{
    const int nrows = A.Size();
    const int ncols = A.Width();

    const int * i_A = A.GetI();
    const int * j_A = A.GetJ();
    const double * v_A = A.GetData();

    PARELAG_TEST_FOR_EXCEPTION(
        nrows != ncols,
        std::logic_error,
        "IsAlmostIdentity(): A should be a square matrix");

    int ndiff(0);
    double maxErr(0.0), mydiff(0);

    for(int irow(0); irow < nrows; ++irow)
        for(int jpos = i_A[irow]; jpos < i_A[irow+1]; ++ jpos)
        {
            if(j_A[jpos] == irow)
                mydiff = fabs(v_A[jpos] - 1.);
            else
                mydiff = fabs(v_A[jpos]);

            if( mydiff > tol)
                ++ndiff;
            if( mydiff > maxErr)
                maxErr = mydiff;
        }

    if(mydiff > tol || verbose)
    {
        std::cout << "    nnz(A - I) = " << ndiff << std::endl;
        std::cout << "normInf(A - I) = " << maxErr << std::endl;
    }

    return (mydiff <= tol);
}

bool IsDiagonal(const SparseMatrix & A)
{
    if( A.Size() != A.Width() || A.NumNonZeroElems() != A.Size() )
        return false;

    const int size = A.Size();

    const int * it = A.GetJ();
    for(int i = 0; i < size; ++i, ++it)
        if(*it != i)
            return false;

    return true;
}

void fillSparseIdentity(int * I, int * J, double * A, int size)
{
    for (int ii = 0; ii < size; ++ii)
    {
        I[ii] = ii;
        J[ii] = ii;
        A[ii] = 1.0;
    }
    I[size] = size;
}

unique_ptr<SparseMatrix> createSparseIdentityMatrix(int size)
{
    std::unique_ptr<int[]> I(new int[size+1]);
    std::unique_ptr<int[]> J(new int[size]);
    std::unique_ptr<double[]> A(new double[size]);

    fillSparseIdentity(I.get(), J.get(), A.get(), size);

    return make_unique<SparseMatrix>(
        I.release(),J.release(),A.release(),size,size);
}


// Returns a matrix 1 by width, with entries A(0,i) = data[i] for i
// \in 0 ... width-1
unique_ptr<SparseMatrix>
createSparseMatrixRepresentationOfScalarProduct(double * data, int width)
{
    std::unique_ptr<int[]> I(new int[2]);
    std::unique_ptr<int[]> J(new int[width]);

    I[0] = 0; I[1] = width;
    std::iota(J.get(),J.get()+width,0);

    return make_unique<SparseMatrix>(I.release(),J.release(),data,1,width);
}

void destroySparseMatrixRepresentationOfScalarProduct(SparseMatrix *& A)
{
    delete[] A->GetI();
    delete[] A->GetJ();
    A->LoseData();
    delete A;
    A = nullptr;
}

unique_ptr<SparseMatrix> diagonalMatrix(double * data, int size)
{
    std::unique_ptr<int[]> i_A(new int[size+1]);
    std::unique_ptr<int[]> j_A(new int[size]);

    for(int i(0); i < size; ++i)
    {
        i_A[i] = i;
        j_A[i] = i;
    }
    i_A[size] = size;

    return make_unique<SparseMatrix>(
        i_A.release(), j_A.release(), data, size, size);
}

unique_ptr<SparseMatrix> diagonalMatrix(int size)
{
    std::unique_ptr<double[]> data(new double[size]);
    std::fill_n(data.get(), size, 0.);
    return diagonalMatrix(data.release(), size);
}

unique_ptr<SparseMatrix> spzeros(int nrows, int ncols)
{
    std::unique_ptr<int[]> i(new int[nrows+1]);
    std::fill_n(i.get(), nrows+1, 0);
    return make_unique<SparseMatrix>(i.release(),nullptr,nullptr, nrows, ncols);
}

unique_ptr<SparseMatrix> DeepCopy(SparseMatrix & A)
{
    PARELAG_TEST_FOR_EXCEPTION(
        !A.Finalized(),
        std::logic_error,
        "DeepCopy(): Matrix A should be finalized.");

    const int nrows = A.Size();
    const int ncols = A.Width();
    const int nnz   = A.NumNonZeroElems();

    std::unique_ptr<int[]> i_new(new int[nrows+1]);
    std::unique_ptr<int[]> j_new(new int[nnz]);
    std::unique_ptr<double[]> a_new(new double[nnz]);

    const int * a_I = A.GetI();
    const int * a_J = A.GetJ();
    const double * a_Data = A.GetData();
    std::copy_n(a_I,nrows+1,i_new.get());
    std::copy_n(a_J,nnz,j_new.get());
    std::copy_n(a_Data,nnz,a_new.get());

    return make_unique<SparseMatrix>(
        i_new.release(), j_new.release(), a_new.release(), nrows, ncols);
}

void dropSmallEntry(SparseMatrix & A, double tol)
{
    int * i_it = A.GetI();
    int * j_it = A.GetJ();
    double * a_val = A.GetData();

    int * new_j = j_it;
    double * new_a = a_val;

    const int nrows = A.Size();
    int currentCounter = 0;
    int end;
    int nnz = 0;

    for(int irow = 0; irow < nrows; ++irow)
    {
        for(end = i_it[irow+1]; currentCounter < end; ++currentCounter, ++j_it, ++a_val)
        {
            if( fabs(*a_val) > tol )
            {
                nnz++;
                *(new_a++) = *a_val;
                *(new_j++) = *j_it;
            }
        }
        i_it[irow+1] = nnz;
    }
}

void signumTransformation(SparseMatrix & A)
{
    constexpr double tol = 1e-10;
    const int nnz = A.NumNonZeroElems();
    double * a = A.GetData();

    for(double * it = a; it != a+nnz; ++it)
        if(fabs(*it) < tol)
            *it = 0.;
        else
            *it = (*it > 0) ? 1.:-1.;
}

void CheckMatrix(const SparseMatrix & A)
{
    const int nCols = A.Width();

    const int nnz = A.NumNonZeroElems();
    const int * j = A.GetJ();
    const int * end = j+nnz;

    for( ;j != end;++j)
        PARELAG_TEST_FOR_EXCEPTION(
            *j < 0 || *j >= nCols,
            std::range_error,
            "CheckMatrix(): Invalid column index detected!");

    const Vector v(const_cast<double *>(A.GetData()), nnz);
    PARELAG_TEST_FOR_EXCEPTION(
        v.CheckFinite(),
        std::range_error,
        "CheckMatrix(): A contains non-finite data!");

}

unique_ptr<SparseMatrix> Add(double a,
                             const SparseMatrix & A,
                             double b,
                             const SparseMatrix & B,
                             double c,
                             const SparseMatrix & C)
{
    PARELAG_TEST_FOR_EXCEPTION(
        A.Size() != B.Size() || B.Size() != C.Size(),
        std::logic_error,
        "Add(): A,B,C have different number of rows.");

    PARELAG_TEST_FOR_EXCEPTION(
        A.Width() != B.Width() || B.Width() != C.Width(),
        std::logic_error,
        "Add(): A,B,C have different number of cols.");

    const int nrows = A.Size();
    const int ncols = A.Width();

    std::unique_ptr<int[]> R_i(new int[nrows+1]);

    const int * A_i = A.GetI();
    const int * A_j = A.GetJ();
    const double * A_data = A.GetData();

    const int * B_i = B.GetI();
    const int * B_j = B.GetJ();
    const double * B_data = B.GetData();

    const int * C_i = C.GetI();
    const int * C_j = C.GetJ();
    const double * C_data = C.GetData();

    std::vector<int> marker(ncols,-1);

    int num_nonzeros = 0, jcol;
    R_i[0] = 0;
    for (int ir = 0; ir < nrows; ir++)
    {
        for (int ia = A_i[ir]; ia < A_i[ir+1]; ia++)
        {
            jcol = A_j[ia];
            marker[jcol] = ir;
            num_nonzeros++;
        }
        for (int ib = B_i[ir]; ib < B_i[ir+1]; ib++)
        {
            jcol = B_j[ib];
            if (marker[jcol] != ir)
            {
                marker[jcol] = ir;
                num_nonzeros++;
            }
        }

        for (int ic = C_i[ir]; ic < C_i[ir+1]; ic++)
        {
            jcol = C_j[ic];
            if (marker[jcol] != ir)
            {
                marker[jcol] = ir;
                num_nonzeros++;
            }
        }

        R_i[ir+1] = num_nonzeros;
    }

    std::unique_ptr<int[]> R_j(new int[num_nonzeros]);
    std::unique_ptr<double[]> R_data(new double[num_nonzeros]);

    std::fill(marker.begin(), marker.end(), -1);

    int pos = 0;
    for (int ir = 0; ir < nrows; ir++)
    {
        for (int ia = A_i[ir]; ia < A_i[ir+1]; ia++)
        {
            jcol = A_j[ia];
            R_j[pos] = jcol;
            R_data[pos] = a*A_data[ia];
            marker[jcol] = pos;
            pos++;
        }

        for (int ib = B_i[ir]; ib < B_i[ir+1]; ib++)
        {
            jcol = B_j[ib];
            if (marker[jcol] < R_i[ir])
            {
                R_j[pos] = jcol;
                R_data[pos] = b*B_data[ib];
                marker[jcol] = pos;
                pos++;
            }
            else
            {
                R_data[marker[jcol]] += b*B_data[ib];
            }
        }

        for (int ic = C_i[ir]; ic < C_i[ir+1]; ic++)
        {
            jcol = C_j[ic];
            if (marker[jcol] < R_i[ir])
            {
                R_j[pos] = jcol;
                R_data[pos] = c*C_data[ic];
                marker[jcol] = pos;
                pos++;
            }
            else
            {
                R_data[marker[jcol]] += c*C_data[ic];
            }
        }
    }

    return make_unique<SparseMatrix>(
        R_i.release(), R_j.release(), R_data.release(), nrows, ncols);
}

void Full(const SparseMatrix & Asparse, DenseMatrix & Adense)
{
    const int nrow = Asparse.Size();
    const int ncol = Asparse.Width();

    Adense.SetSize(nrow, ncol);
    Adense = 0.;

    const int * i_A = Asparse.GetI();
    const int * j_A = Asparse.GetJ();
    const double * a_A = Asparse.GetData();

    int jcol = 0;
    int end;

    for(int irow(0); irow < nrow; ++irow)
        for(end = i_A[irow+1]; jcol != end; ++jcol)
            Adense(irow, j_A[jcol]) = a_A[jcol];
}

void Full(const SparseMatrix & Asparse, DenseMatrix & Adense, int i_offset,
          int j_offset)
{
    const int nrow = Asparse.Size();

    const int * i_A = Asparse.GetI();
    const int * j_A = Asparse.GetJ();
    const double * a_A = Asparse.GetData();

    int jcol = 0;
    int end;

    for (int irow(0); irow < nrow; ++irow)
        for (end = i_A[irow+1]; jcol != end; ++jcol)
            Adense(irow + i_offset, j_A[jcol] + j_offset) = a_A[jcol];
}

void AddMatrix(const DenseMatrix &A, const SparseMatrix & B, DenseMatrix & C)
{
    const int nrows = B.Size();

    PARELAG_TEST_FOR_EXCEPTION(
        A.Height() != nrows || nrows != C.Height(),
        std::logic_error,
        "AddMatrix(): Rows of A,B,C don't match!");

    PARELAG_TEST_FOR_EXCEPTION(
        A.Width() != B.Width() || B.Width() != C.Width(),
        std::logic_error,
        "AddMatrix(): Columns of A,B,C don't match!");

    const int * i_B = B.GetI();
    const int * j_B = B.GetJ();
    const double * v_B = B.GetData();

    if( C.Data() != A.Data())
        C = A;

    for(int irow(0); irow < nrows; ++irow)
    {
        for(int jpos = i_B[irow]; jpos < i_B[irow+1]; ++ jpos)
            C( irow, j_B[jpos] ) += v_B[jpos];
    }

}

unique_ptr<SparseMatrix> Kron(SparseMatrix & A, SparseMatrix & B)
{
    const int nrowsA = A.Size();
    const int ncolsA = A.Width();
    const int nnzA   = A.NumNonZeroElems();

    const int nrowsB = B.Size();
    const int ncolsB = B.Width();
    const int nnzB   = B.NumNonZeroElems();

    const int nrowsR = nrowsA * nrowsB;
    const int ncolsR = ncolsA * ncolsB;
    const int nnzR   = nnzA   * nnzB;

    const int * i_A = A.GetI();
    const int * j_A = A.GetJ();
    const double * v_A = A.GetData();

    const int * i_B = B.GetI();
    const int * j_B = B.GetJ();
    const double * v_B = B.GetData();

    std::unique_ptr<int[]> ii_R(new int[nrowsR+2]);
    ii_R[0] = 0;
    int * i_R  = ii_R.get()+1;
    std::unique_ptr<int[]> j_R(new int[nnzR]);
    std::unique_ptr<double[]> v_R(new double[nnzR]);

    int irowA, jposA, jcolA, irownnzA;
    int irowB, jposB, jcolB, irownnzB;
    int irowR, jposR, jcolR;

    double vA, vB;

    //DO NOT ACCELERATE THIS LOOP
    i_R[0] = 0;
    for(irowA = 0; irowA < nrowsA; ++irowA)
    {
        irownnzA = i_A[irowA+1] - i_A[irowA];
        for(irowB = 0; irowB < nrowsB; ++irowB)
        {
            irownnzB = i_B[irowB+1] - i_B[irowB];
            irowR = irowA * nrowsB + irowB;
            i_R[irowR+1] = i_R[irowR]+(irownnzA*irownnzB);
        }
    }

#ifdef ELAG_DEBUG
    PARELAG_TEST_FOR_EXCEPTION(
        i_R[nrowsR] != nnzR,
        std::runtime_error,
        "Kron(): This routine is bugged!! ... ok then ...");
#endif


    for(irowA = 0; irowA < nrowsA; ++irowA)
    {
        for(jposA = i_A[irowA]; jposA < i_A[irowA+1]; ++jposA)
        {
            jcolA = j_A[jposA];
            vA    = v_A[jposA];
            for(irowB = 0; irowB < nrowsB; ++irowB)
            {
                irowR = irowA * nrowsB + irowB;
                for(jposB = i_B[irowB]; jposB < i_B[irowB+1]; ++jposB)
                {
                    jcolB = j_B[jposB];
                    vB    = v_B[jposB];
                    jcolR = jcolA * ncolsB + jcolB;
                    jposR = i_R[irowR]++;
                    j_R[jposR] = jcolR;
                    v_R[jposR] = vA*vB;
                }
            }
        }
    }

    return make_unique<SparseMatrix>(
        ii_R.release(), j_R.release(), v_R.release(), nrowsR, ncolsR);
}

void AddMatrix(double sa,
               const DenseMatrix &A,
               double sb,
               const SparseMatrix & B,
               DenseMatrix & C)
{
    const int nrows = B.Size();

    PARELAG_TEST_FOR_EXCEPTION(
        A.Height() != nrows || nrows != C.Height(),
        std::logic_error,
        "AddMatrix(): Rows of A,B,C don't match!");

    PARELAG_TEST_FOR_EXCEPTION(
        A.Width() != B.Width() || B.Width() != C.Width(),
        std::logic_error,
        "AddMatrix(): Columns of A,B,C don't match!");

    const int * i_B = B.GetI();
    const int * j_B = B.GetJ();
    const double * v_B = B.GetData();

    C = A;
    C *= sa;

    for(int irow(0); irow < nrows; ++irow)
    {
        for(int jpos = i_B[irow]; jpos < i_B[irow+1]; ++ jpos)
            C( irow, j_B[jpos] ) += sb * v_B[jpos];
    }
}

void AddOpenFormat(SparseMatrix & A, SparseMatrix & B)
{
    const int nrows = A.Size();

    elag_assert( nrows == B.Size() );
    elag_assert( A.Width() >= B.Width() );

    Array<int> rows(1);
    Array<int> cols;
    DenseMatrix loc;

    const int * i_B = B.GetI();
    int * j_B = B.GetJ();
    double * val = B.GetData();

    for(int i(0); i < nrows; ++i)
    {
        const int& start = i_B[i];
        const int& end = i_B[i+1];
        const int len = end - start;

        rows[0] = i;
        cols.MakeRef(j_B+start, len);
        loc.UseExternalData(val+start, 1, len);

        A.AddSubMatrix(rows, cols, loc);

        loc.ClearExternalData();
    }
}

void Mult(const SparseMatrix & A, const DenseMatrix & B, DenseMatrix & out)
{
    const int size = A.Size();
    PARELAG_TEST_FOR_EXCEPTION(
        A.Width() != B.Height() || size != out.Height() || B.Width() != out.Width(),
        std::logic_error,
        "Mult(): Dimensions of A,B,out don't match.");

    const int * I = A.GetI();
    const int * J = A.GetJ();
    const double * val = A.GetData();

    int i,j,end;
    double * bi_data(B.Data() ), * outi_data(out.Data() );

    for( int icol(0); icol < B.Width(); ++icol)
    {
        for (i = j = 0; i < size; i++)
        {
            double d = 0.0;
            for (end = I[i+1]; j < end; j++)
            {
                d += val[j] * bi_data[J[j]];
            }
            outi_data[i] = d;
        }
        bi_data += B.Height();
        outi_data += out.Height();
    }
}

std::unique_ptr<SparseMatrix> MultAbs (const SparseMatrix &A, const SparseMatrix &B)
{
   int nrowsA, ncolsA, nrowsB, ncolsB;
   const int *A_i, *A_j, *B_i, *B_j;
   const double *A_data, *B_data;
   int ia, ib, ic, ja, jb, num_nonzeros;
   int row_start, counter;
   double a_entry, b_entry;

   nrowsA = A.Height();
   ncolsA = A.Width();
   nrowsB = B.Height();
   ncolsB = B.Width();

   PARELAG_TEST_FOR_EXCEPTION(
       ncolsA != nrowsB,
       std::logic_error,
       "MultAbs(): "
       "ncolsA = " << ncolsA << " and nrowsB = " << nrowsB <<
       ": sizes don't match!" );

   A_i    = A.GetI();
   A_j    = A.GetJ();
   A_data = A.GetData();
   B_i    = B.GetI();
   B_j    = B.GetJ();
   B_data = B.GetData();

   std::unique_ptr<int[]> B_marker(new int[ncolsB]);

   for (ib = 0; ib < ncolsB; ib++)
   {
      B_marker[ib] = -1;
   }

   std::unique_ptr<int[]> C_i(new int[nrowsA+1]);

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

   std::unique_ptr<int[]> C_j(new int[num_nonzeros]);
   std::unique_ptr<double[]> C_data(new double[num_nonzeros]);

   for (ib = 0; ib < ncolsB; ib++)
   {
      B_marker[ib] = -1;
   }

   counter = 0;
   for (ic = 0; ic < nrowsA; ic++)
   {
      // row_start = C_i[ic];
      row_start = counter;
      for (ia = A_i[ic]; ia < A_i[ic+1]; ia++)
      {
         ja = A_j[ia];
         a_entry = fabs(A_data[ia]);
         for (ib = B_i[ja]; ib < B_i[ja+1]; ib++)
         {
            jb = B_j[ib];
            b_entry = fabs(B_data[ib]);
            if (B_marker[jb] < row_start)
            {
               B_marker[jb] = counter;
               C_j[counter] = jb;
               C_data[counter] = a_entry*b_entry;
               counter++;
            }
            else
            {
               C_data[B_marker[jb]] += a_entry*b_entry;
            }
         }
      }
   }

   return make_unique<SparseMatrix>(
       C_i.release(), C_j.release(), C_data.release(), nrowsA, ncolsB);
}

unique_ptr<mfem::HypreParMatrix> IgnoreNonLocalRange(
    hypre_ParCSRMatrix* RT, hypre_ParCSRMatrix* A, hypre_ParCSRMatrix* P)
{
    using sercsr_ptr_t = HypreTraits<hypre_CSRMatrix>::unique_ptr_t;

    hypre_ParCSRMatrix * out;

    // Create a new, temporary offd block
    sercsr_ptr_t tmp_offd{
        hypre_ZerosCSRMatrix(
            hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(RT)), 0),
            hypre_CSRMatrixDestroy};

    // Swap out the offd piece of "RT"
    hypre_CSRMatrix* hold_offd = hypre_ParCSRMatrixOffd(RT);
    hypre_ParCSRMatrixOffd(RT) = tmp_offd.get();

    // Swap out the commpkg
    hypre_ParCSRCommPkg* hold_commpkg = hypre_ParCSRMatrixCommPkg(RT);
    hypre_ParCSRMatrixCommPkg(RT) = nullptr;

    hypre_BoomerAMGBuildCoarseOperator(RT,A,P,&out);

    // Cleanup the newly created objects
    if (hypre_ParCSRMatrixCommPkg(RT))
        hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkg(RT));

    // Restore the old objects
    hypre_ParCSRMatrixOffd(RT) = hold_offd;
    hypre_ParCSRMatrixCommPkg(RT) = hold_commpkg;

#if MFEM_HYPRE_VERSION <= 22200
    // These are owned elsewhere
    hypre_ParCSRMatrixOwnsRowStarts(out) = 0;
    hypre_ParCSRMatrixOwnsColStarts(out) = 0;
#endif

    return make_unique<mfem::HypreParMatrix>(out);
}

// Definition here in the CPP to prevent extraneous compilations
template <class MatrixType>
unique_ptr<MatrixType> RAP(const MatrixType & R,
                           const MatrixType & A,
                           const MatrixType & P)
{
    unique_ptr<MatrixType> AP{Mult(A,P)};
    return unique_ptr<MatrixType>{Mult(R,*AP)};
}

// Instantiate for SparseMatrix
template unique_ptr<SparseMatrix> RAP(const SparseMatrix&,
                                      const SparseMatrix&,
                                      const SparseMatrix&);
// Instantiate for BlockMatrix
template unique_ptr<BlockMatrix> RAP(const BlockMatrix&,
                                     const BlockMatrix&,
                                     const BlockMatrix&);
// Instantiate for HypreParMatrix
template unique_ptr<HypreParMatrix> RAP(const HypreParMatrix&,
                                        const HypreParMatrix&,
                                        const HypreParMatrix&);

// Definition here in the CPP to prevent extraneous compilations
template <class MatrixType>
unique_ptr<MatrixType> PtAP(const MatrixType & A,
                            const MatrixType & P)
{
    unique_ptr<MatrixType> Pt{Transpose(P)};
    return RAP<MatrixType>(*Pt,A,P);
}

// Instantiate for SparseMatrix
template unique_ptr<SparseMatrix> PtAP(const SparseMatrix&,
                                       const SparseMatrix&);
// Instantiate for BlockMatrix
template unique_ptr<BlockMatrix> PtAP(const BlockMatrix&,
                                      const BlockMatrix&);

double RowNormL1(const SparseMatrix & A, int irow)
{
    Vector vals;
    Array<int> cols;

    A.GetRow(irow, cols, vals);

    return vals.Norml1();
}

void Weightedl1Smoother(const SparseMatrix& A, Vector& diagonal_matrix)
{
    const int n = A.Size();
    elag_assert(A.Width() == n);
    diagonal_matrix.SetSize(n);
    double *Data = diagonal_matrix.GetData();
    double sum;

    for (int i=0; i < n; ++i)
    {
        const int *row;
        int beg;
        const double *a;
        double diag;

        sum = 0.;
        diag = A(i, i);
        beg = A.GetI()[i];
        row = A.GetJ() + beg;
        a = A.GetData() + beg;
        elag_assert(diag > 0.);
        const int a_rsz = const_cast<SparseMatrix&>(A).RowSize(i);
        for (int j=0; j < a_rsz; ++j)
        {
            elag_assert(A(row[j], row[j]) > 0.);
            sum += fabs(a[j]) * sqrt(diag / A(row[j], row[j]));
        }
        elag_assert(sum > 0.);
        Data[i] = sum;
    }
}

void Block2by2(DenseMatrix &A00, DenseMatrix &A01, DenseMatrix &A10,
               DenseMatrix &A11, DenseMatrix &A)
{
    PARELAG_ASSERT(A00.Height() == A01.Height());
    PARELAG_ASSERT(A10.Height() == A11.Height());
    PARELAG_ASSERT(A00.Width() == A10.Width());
    PARELAG_ASSERT(A01.Width() == A11.Width());

    const int middle_row = A00.Height();
    const int middle_col = A00.Width();
    const int nrows = middle_row + A10.Height();
    const int ncols = middle_col + A01.Width();

    A.SetSize(nrows, ncols);
    A = 0.;
    A.AddMatrix(A00, 0, 0);
    A.AddMatrix(A01, 0, middle_col);
    A.AddMatrix(A10, middle_row, 0);
    A.AddMatrix(A11, middle_row, middle_col);
}

void BlockDiag2by2(DenseMatrix &A00, DenseMatrix &A11, DenseMatrix &A)
{
    const int middle_row = A00.Height();
    const int middle_col = A00.Width();
    const int nrows = middle_row + A11.Height();
    const int ncols = middle_col + A11.Width();

    A.SetSize(nrows, ncols);
    A = 0.;
    A.AddMatrix(A00, 0, 0);
    A.AddMatrix(A11, middle_row, middle_col);
}

void SplitMatrixHorizontally(const DenseMatrix &A, int middle_row,
                             DenseMatrix &top, DenseMatrix &bottom)
{
    PARELAG_ASSERT(A.Height() > middle_row);

    const int nrows = A.Height();
    const int ncols = A.Width();

    DenseMatrix At(A, 't');
    double *data = At.Data();
    At.UseExternalData(data, ncols, middle_row);
    top = At;
    top.Transpose();
    At.UseExternalData(data + ncols*middle_row, ncols, nrows - middle_row);
    bottom = At;
    bottom.Transpose();
    At.UseExternalData(data, ncols, nrows);
    delete [] data;
}

std::unique_ptr<mfem::HypreParMatrix>
Mult(const mfem::HypreParMatrix& A, const mfem::HypreParMatrix& B, bool own_starts)
{
    auto out = std::unique_ptr<mfem::HypreParMatrix>(mfem::ParMult(&A, &B));
    if (own_starts)
    {
       out->CopyRowStarts();
       out->CopyColStarts();
    }
    return out;
}
}//namespace parelag
