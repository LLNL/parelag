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

#include "linalg/utilities/ParELAG_SubMatrixExtraction.hpp"

#include "utilities/elagError.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

unique_ptr<SparseMatrix> ExtractRowAndColumns(const SparseMatrix & A,
                                              const Array<int> & rows,
                                              const Array<int> & cols,
                                              Array<int> & colMapper)
{
    const int * i_A = A.GetI();
    const int * j_A = A.GetJ();
    const double * a_A = A.GetData();

#ifdef ELAG_DEBUG
    const int nrow_A = A.Size();
    const int ncol_A = A.Width();

    PARELAG_TEST_FOR_EXCEPTION(
        rows.Size() && rows.Max() >= nrow_A,
        std::logic_error,
        "ExtractRowsAndColumns(): Row index exceeds matrix size.");

    PARELAG_TEST_FOR_EXCEPTION(
        cols.Size() && cols.Max() >= ncol_A,
        std::logic_error,
        "ExtractRowsAndColumns(): Column index exceeds matrix width.");

    PARELAG_TEST_FOR_EXCEPTION(
        colMapper.Size() != ncol_A,
        std::logic_error,
        "ExtractRowsAndColumns(): tmpColMarker and A are incompatible sizes.");
#endif


    for(int jcol(0); jcol < cols.Size(); ++jcol)
        colMapper[cols[jcol]] = jcol;

    const int nrow_sub = rows.Size();
    const int ncol_sub = cols.Size();

    int * i_sub = new int[nrow_sub+1];
    i_sub[0] = 0;

    // Find the number of nnz.
    int currentRow(-1);
    int nnz(0);
    for(int i(0); i<nrow_sub; ++i)
    {
        currentRow = rows[i];
        for(const int * it = j_A+i_A[currentRow]; it != j_A+i_A[currentRow+1]; ++it)
            if( colMapper[*it] >= 0)
                ++nnz;

        i_sub[i+1] = nnz;
    }

    //Allocate memory
    int * j_sub = new int[nnz];
    double * a_sub = new double[nnz];

    //Fill in the matrix
    const double * it_a;
    int * it_j_sub = j_sub;
    double * it_a_sub = a_sub;
    for(int i(0); i<nrow_sub; ++i)
    {
        currentRow = rows[i];
        it_a = a_A+i_A[currentRow];
        for(const int * it = j_A+i_A[currentRow]; it != j_A+i_A[currentRow+1]; ++it, ++it_a)
            if( colMapper[*it] >= 0)
            {
                *(it_j_sub++) = colMapper[*it];
                *(it_a_sub++) = *it_a;
            }
    }

    // Restore colMapper so it can be reused other times!
    for(int jcol(0); jcol < cols.Size(); ++jcol)
        colMapper[cols[jcol]] = -1;

    return make_unique<SparseMatrix>(i_sub, j_sub, a_sub, nrow_sub, ncol_sub);
}

void findGlobalIndex(int iglobal,const Array<int> & offsets, int & iblock, int & iloc)
{
    int nblocks = offsets.Size()-1;
    if(iglobal > offsets[nblocks])
        mfem_error("findGlobalIndex");

    for(iblock = 0; iblock < nblocks; ++iblock)
        if(offsets[iblock+1] > iglobal)
            break;

    iloc = iglobal - offsets[iblock];
}

unique_ptr<BlockMatrix> ExtractRowAndColumns(const BlockMatrix * A,
                                             const Array<int> & gRows,
                                             const Array<int> & gCols,
                                             Array<int> & colMapper)
{
    //(1) Count how many gRows, gCols belongs to each block
    const int nRowBlocks = A->NumRowBlocks();
    const int nColBlocks = A->NumColBlocks();
    const Array<int> & row_offsets(A->RowOffsets());
    const Array<int> & col_offsets(A->ColOffsets());
    Array<int> offset_lrows(nRowBlocks+1), offset_lcols(nColBlocks+1);
    offset_lrows = 0;
    offset_lcols = 0;

    int block, lrow, lcol;
    for(const int * it = gRows.GetData(), * end = gRows.GetData()+gRows.Size(); it != end; ++it)
    {
        findGlobalIndex(*it, row_offsets, block, lrow);
        ++offset_lrows[block+1];
    }
    offset_lrows.PartialSum();

    for(const int * it = gCols.GetData(), * end = gCols.GetData()+gCols.Size(); it != end; ++it)
    {
        findGlobalIndex(*it, col_offsets, block, lcol);
        ++offset_lcols[block+1];
    }
    offset_lcols.PartialSum();

    //(2) Go from global to block indexing
    int * lrows_data = new int[offset_lrows.Last()];
    int * lcols_data = new int[offset_lcols.Last()];

    Array<int *> lrows_it(nRowBlocks), lcols_it(nColBlocks);

    for(int iblock(0); iblock < nRowBlocks; ++iblock)
        lrows_it[iblock] = lrows_data + offset_lrows[iblock];
    for(int jblock(0); jblock < nRowBlocks; ++jblock)
        lcols_it[jblock] = lcols_data + offset_lcols[jblock];

    for(const int * it = gRows.GetData(), * end = gRows.GetData()+gRows.Size(); it != end; ++it)
    {
        findGlobalIndex(*it, row_offsets, block, lrow);
        *(lrows_it[block]) = lrow;
        ++(lrows_it[block]);
    }

    for(const int * it = gCols.GetData(), * end = gCols.GetData()+gCols.Size(); it != end; ++it)
    {
        findGlobalIndex(*it, col_offsets, block, lcol);
        *(lcols_it[block]) = lcol;
        ++(lcols_it[block]);
    }

    auto out = make_unique<BlockMatrix>(offset_lrows, offset_lcols);
    out->owns_blocks = 1;

    Array<int> bRows, bCols, bColMapper;
    for(int iblock(0); iblock < nRowBlocks; ++iblock)
        for(int jblock(0); jblock < nColBlocks; ++jblock)
        {
            if( !A->IsZeroBlock(iblock,jblock) )
            {
                bRows.MakeRef(lrows_data+offset_lrows[iblock],
                              offset_lrows[iblock+1] - offset_lrows[iblock]);
                bCols.MakeRef(lcols_data+offset_lcols[jblock],
                              offset_lcols[jblock+1] - offset_lcols[jblock]);
                bColMapper.MakeRef(colMapper.GetData()+col_offsets[jblock],
                                   col_offsets[jblock+1] - col_offsets[jblock]);
                auto tmp = ExtractRowAndColumns(A->GetBlock(iblock,jblock),
                                                bRows,
                                                bCols,
                                                bColMapper);
                out->SetBlock(iblock, jblock, tmp.release());
            }
        }

    out->RowOffsets().MakeDataOwner();
    out->ColOffsets().MakeDataOwner();

    offset_lrows.LoseData();
    offset_lcols.LoseData();

    delete[] lrows_data;
    delete[] lcols_data;

    return out;
}


unique_ptr<SparseMatrix> ExtractColumns(const SparseMatrix & A,
                                        const Array<int> & cols,
                                        Array<int> & colMapper)
{
    const int * i_A = A.GetI();
    const int * j_A = A.GetJ();
    const double * a_A = A.GetData();
    const int nrow_A = A.Size();

#ifdef ELAG_DEBUG
    const int ncol_A = A.Width();

    PARELAG_TEST_FOR_EXCEPTION(
        cols.Max() >= ncol_A,
        std::logic_error,
        "ExtractColumns(): Column index exceeds matrix width.");

    PARELAG_TEST_FOR_EXCEPTION(
        colMapper.Size() != ncol_A,
        std::logic_error,
        "ExtractColumns(): tmpColMarker and A are of incompatible size.");
#endif

    for(int jcol(0); jcol < cols.Size(); ++jcol)
        colMapper[cols[jcol]] = jcol;

    const int ncol_sub = cols.Size();

    int * i_sub = new int[nrow_A+1];
    i_sub[0] = 0;

    // Find the number of nnz.
    int nnz(0);
    for(int currentRow(0); currentRow<nrow_A; ++currentRow)
    {
        for(const int * it = j_A+i_A[currentRow]; it != j_A+i_A[currentRow+1]; ++it)
            if( colMapper[*it] >= 0)
                ++nnz;

        i_sub[currentRow+1] = nnz;
    }

    //Allocate memory
    int * j_sub = new int[nnz];
    double * a_sub = new double[nnz];

    //Fill in the matrix
    const double * it_a;
    int * it_j_sub = j_sub;
    double * it_a_sub = a_sub;
    for(int currentRow(0); currentRow<nrow_A; ++currentRow)
    {
        it_a = a_A+i_A[currentRow];
        for(const int * it = j_A+i_A[currentRow]; it != j_A+i_A[currentRow+1]; ++it, ++it_a)
            if( colMapper[*it] >= 0)
            {
                *(it_j_sub++) = colMapper[*it];
                *(it_a_sub++) = *it_a;
            }
    }

    // Restore colMapper so it can be reused other times!
    for(int jcol(0); jcol < cols.Size(); ++jcol)
        colMapper[cols[jcol]] = -1;

    return make_unique<SparseMatrix>(i_sub, j_sub, a_sub, nrow_A, ncol_sub);
}

unique_ptr<SparseMatrix> ExtractRowAndColumns(const SparseMatrix & A,
                                              const Array<int> & rows,
                                              const Array<int> & cols)
{
    const int nCols = A.Width();
    Array<int> tmpColMarker(nCols);
    std::fill(tmpColMarker.GetData(), tmpColMarker.GetData()+nCols, -1);

    return ExtractRowAndColumns(A, rows, cols, tmpColMarker);
}


void ExtractSubMatrices(const SparseMatrix & A,
                        const Table & subMat_row,
                        const Table & subMat_col,
                        std::vector<unique_ptr<SparseMatrix>> & subMats)
{
#ifdef ELAG_DEBUG
    const auto subsize = subMats.size();
    PARELAG_TEST_FOR_EXCEPTION(
        subMat_row.Size() != subMat_col.Size(),
        std::logic_error,
        "ExtractSubMatrices(): "
        "subMat_row and subMat_col have incompatible sizes.");
    PARELAG_TEST_FOR_EXCEPTION(
        subMats.size() != decltype(subsize)(subMat_row.Size()),
        std::logic_error,
        "ExtractSubMatrices(): "
        "subMats and subMat_row have incompatible sizes.");
#endif

    const int nSubMatrices = subMat_row.Size();

    const int nCols = A.Width();

    Array<int> tmpColMarker(nCols);
    std::fill(tmpColMarker.GetData(), tmpColMarker.GetData()+nCols, -1);

    const int * i_subMat_row = subMat_row.GetI();
    const int * j_subMat_row = subMat_row.GetJ();

    const int * i_subMat_col = subMat_col.GetI();
    const int * j_subMat_col = subMat_col.GetJ();


    for(int i(0); i < nSubMatrices; ++i)
    {
        int n_loc_row = i_subMat_row[i+1] - i_subMat_row[i];
        int n_loc_col = i_subMat_col[i+1] - i_subMat_col[i];

        Array<int> loc_row(const_cast<int*>(j_subMat_row+i_subMat_row[i]), n_loc_row);
        Array<int> loc_col(const_cast<int*>(j_subMat_col+i_subMat_col[i]), n_loc_col);

        subMats[i] = ExtractRowAndColumns(A, loc_row, loc_col, tmpColMarker);
    }
}

unique_ptr<SparseMatrix> ExtractSubMatrix(const SparseMatrix & A,
                                          const int rowStart,
                                          const int rowEnd,
                                          const int colStart,
                                          const int colEnd)
{
    const int * i_A = A.GetI();
    const int * j_A = A.GetJ();
    const double * a_A = A.GetData();

#ifdef ELAG_DEBUG
    const int nrow_A = A.Size();
    const int ncol_A = A.Width();

    PARELAG_TEST_FOR_EXCEPTION(
        rowEnd > nrow_A,
        std::logic_error,
        "ExtractSubMatrix(): Rows index exceeds matrix size." );

    PARELAG_TEST_FOR_EXCEPTION(
        colEnd > ncol_A,
        std::logic_error,
        "ExtractSubMatrix(): Columns index exceeds matrix width.");
#endif

    const int nrow_sub = rowEnd - rowStart;
    const int ncol_sub = colEnd - colStart;

    int * i_sub = new int[nrow_sub+1];
    i_sub[0] = 0;

    // Find the number of nnz.
    int currentRow(-1);
    int nnz(0);
    for(int i(0); i<nrow_sub; ++i)
    {
        currentRow = rowStart+i;
        for(const int * it = j_A+i_A[currentRow]; it != j_A+i_A[currentRow+1]; ++it)
            if( (*it >= colStart) && (*it < colEnd) )
                ++nnz;
        i_sub[i+1] = nnz;
    }

    //Allocate memory
    int * j_sub = new int[nnz];
    double * a_sub = new double[nnz];

    //Fill in the matrix
    const double * it_a;
    int * it_j_sub = j_sub;
    double * it_a_sub = a_sub;
    for(int i(0); i<nrow_sub; ++i)
    {
        currentRow = rowStart+i;
        it_a = a_A+i_A[currentRow];
        for(const int * it = j_A+i_A[currentRow]; it != j_A+i_A[currentRow+1]; ++it, ++it_a)
            if( (*it >= colStart) && (*it < colEnd) )
            {
                *(it_j_sub++) = *it-colStart;
                *(it_a_sub++) = *it_a;
            }
    }

    return make_unique<SparseMatrix>(i_sub, j_sub, a_sub, nrow_sub, ncol_sub);
}

void ExtractComponents(const SparseMatrix & A,
                       Array2D<SparseMatrix *> & Comp,
                       int ordering)
{
    switch(ordering)
    {
    case Ordering::byNODES:
        ExtractComponentsByNODES(A, Comp);
        break;
    case Ordering::byVDIM:
        ExtractComponentsByVDIM( A,  Comp);
        break;
    default:
        mfem_error("ExtractComponents: Unknown Ordering \n");
    }
}

void ExtractComponentsByVDIM(const SparseMatrix & A,
                             Array2D<SparseMatrix *> & Comp)
{
    const int nRowComp = Comp.NumRows();
    const int nColComp = Comp.NumCols();

    if(A.Size() % nRowComp || A.Width() % nColComp)
        mfem_error("A does not have the structure required \n");

    Array2D<int *> i_C(nRowComp, nColComp);
    Array2D<int *> j_C(nRowComp, nColComp);
    Array2D<double *> v_C(nRowComp, nColComp);

    const int nrowsA = A.Size();
    const int ncolsA = A.Width();

    const int nrowsC = nrowsA / nRowComp;
    const int ncolsC = ncolsA / nColComp;
    Array2D<int> nnzC(nRowComp, nColComp);
    nnzC = 0;

    int icomp, jcomp;
    for(icomp = 0; icomp < nRowComp; ++icomp)
        for(jcomp = 0; jcomp < nColComp; ++jcomp)
        {
            i_C(icomp, jcomp) = new int[nrowsC+1];
            i_C(icomp, jcomp)[0] = 0;
        }

    // Loop on the entries of A to build i_C.
    const int * i_A = A.GetI();
    const int * j_A = A.GetJ();
    const double * v_A = A.GetData();
    int irowC, jcolC;
    for(int irowA(0); irowA < nrowsA; ++irowA)
    {
        icomp = irowA % nRowComp;
        irowC = irowA / nRowComp;
        for(int jposA(i_A[irowA]); jposA < i_A[irowA+1]; ++jposA)
        {
            int jcolA = j_A[jposA];
            jcomp = jcolA % nColComp;
            nnzC(icomp, jcomp)++;
        }
        for(jcomp = 0; jcomp < nColComp; ++jcomp)
            i_C(icomp, jcomp)[irowC+1] = nnzC(icomp, jcomp);
    }

    for(icomp = 0; icomp < nRowComp; ++icomp)
        for(jcomp = 0; jcomp < nColComp; ++jcomp)
        {
            j_C(icomp, jcomp) = new int[nnzC(icomp, jcomp)];
            v_C(icomp, jcomp) = new double[nnzC(icomp, jcomp)];
        }

    nnzC = 0;
    for(int irowA(0); irowA < nrowsA; ++irowA)
    {
        icomp = irowA % nRowComp;
        irowC = irowA / nRowComp;
        for(int jposA(i_A[irowA]); jposA < i_A[irowA+1]; ++jposA)
        {
            int jcolA = j_A[jposA];
            jcomp = jcolA % nColComp;
            jcolC = jcolA / nColComp;

            j_C(icomp, jcomp)[ nnzC(icomp, jcomp) ] = jcolC;
            v_C(icomp, jcomp)[ nnzC(icomp, jcomp) ] = v_A[jposA];

            nnzC(icomp, jcomp)++;
        }
    }

    for(icomp = 0; icomp < nRowComp; ++icomp)
        for(jcomp = 0; jcomp < nColComp; ++jcomp)
            Comp(icomp, jcomp) = new SparseMatrix(i_C(icomp, jcomp), j_C(icomp, jcomp), v_C(icomp, jcomp), nrowsC, ncolsC);

}
void ExtractComponentsByNODES(const SparseMatrix & A,
                              Array2D<SparseMatrix *> & Comp)
{
    const int size = A.Size();
    const int width = A.Width();

    const int * i_A = A.GetI();
    const int * j_A = A.GetJ();
    const double * v_A = A.GetData();

    const int br = Comp.NumRows(), bc = Comp.NumCols();
    const int nr = (size + br - 1)/br, nc = (width + bc - 1)/bc;

    Array2D<int *> i_Comp(br,bc);

    int *bI = nullptr;
    for (int j = 0; j < bc; j++)
        for (int i = 0; i < br; i++)
        {
            bI = new int[nr + 1];
            for (int k = 0; k <= nr; k++)
                bI[k] = 0;
            i_Comp(i,j) = bI;
        }

    for (int gr = 0; gr < size; gr++)
    {
        int bi = gr/nr, i = gr%nr + 1;
        for (int jj = i_A[gr]; jj < i_A[gr+1]; ++jj)
            if (v_A[jj] != 0.0)
                (i_Comp(bi,j_A[jj]/nc)[i])++;
    }

    int *bJ;
    double * bA;
    for (int j = 0; j < bc; j++)
        for (int i = 0; i < br; i++)
        {
            bI = i_Comp(i,j);
            int nnz = 0, rs;
            for (int k = 1; k <= nr; k++)
                rs = bI[k], bI[k] = nnz, nnz += rs;
            bJ = new int[nnz];
            bA = new double[nnz];
            Comp(i,j) = new SparseMatrix(bI, bJ, bA, nr, nc);
        }

    for (int gr = 0; gr < size; gr++)
    {
        int bi = gr/nr, i = gr%nr + 1;
        for (int jj = i_A[gr]; jj < i_A[gr+1]; ++jj)
            if (v_A[jj] != 0.0)
            {
                SparseMatrix &b = *Comp(bi,j_A[jj]/nc);
                bI = b.GetI();
                bJ = b.GetJ();
                bA = b.GetData();
                bJ[bI[i]] = j_A[jj] % nc;
                bA[bI[i]] = v_A[jj];
                bI[i]++;
            }
    }
}

void GetRows(const SparseMatrix & A,
             const Array<int> & rows,
             const int colStart,
             const int colEnd,
             MultiVector & V)
{
    PARELAG_TEST_FOR_EXCEPTION(
        V.Size() != colEnd - colStart,
        std::logic_error,
        "GetRows(): MultiVector V has the wrong length!");

    PARELAG_TEST_FOR_EXCEPTION(
        V.NumberOfVectors() != rows.Size(),
        std::logic_error,
        "GetRows(): MultiVector V has the wrong number of vectors!");

    const int * i_A = A.GetI();
    const int * j_A = A.GetJ();
    const double * a_A = A.GetData();

#ifdef ELAG_DEBUG
    const int nrow_A = A.Size();
    const int ncol_A = A.Width();

    if (rows.Size() > 0)
    {
        PARELAG_TEST_FOR_EXCEPTION(
            rows.Max() >= nrow_A,
            std::logic_error,
            "GetRows(): Row index exceeds matrix size.");
    }

    PARELAG_TEST_FOR_EXCEPTION(
        colEnd > ncol_A,
        std::logic_error,
        "GetRows(): Column index exceeds matrix width.");
#endif

    const int nrow_sub = rows.Size();

    // Fill in the MultiVectorV
    const double * it_a;
    double * Vval;
    int currentRow;
    for (int i(0); i<nrow_sub; ++i)
    {
        currentRow = rows[i];
        it_a = a_A+i_A[currentRow];
        Vval = V.GetDataFromVector(i);
        for (const int * it = j_A+i_A[currentRow];
             it != j_A+i_A[currentRow+1];
             ++it, ++it_a)
        {
            if ((*it >= colStart) && (*it < colEnd))
                Vval[*it-colStart] = *it_a;
        }
    }
}

unique_ptr<SparseMatrix> Distribute(const SparseMatrix & A,
                                    const SparseMatrix & AE_row,
                                    const SparseMatrix & AE_col)
{
    PARELAG_TEST_FOR_EXCEPTION(
        AE_row.Size() != AE_col.Size(),
        std::logic_error,
        "Distribute(): AE_row and AE_col must have the same number of rows.");

    PARELAG_TEST_FOR_EXCEPTION(
        AE_row.Width() > A.Size(),
        std::logic_error,
        "Distribute(): "
        "AE_row.Width() = " << AE_row.Width() << std::endl <<
        "A.Size() = " << A.Size() << std::endl <<
        "The number of columns in AE_row must be smaller "
        "or equal to the number of rows in A");

    PARELAG_TEST_FOR_EXCEPTION(
        AE_col.Width() > A.Width(),
        std::logic_error,
        "Distribute(): "
        "AE_col.Width() = " << AE_col.Width() << std::endl <<
        "A.Width() = " << A.Width() << std::endl <<
        "The number of columns in AE_col must be smaller "
        "or equal to the number of cols in A");

    const int * i_A = A.GetI();
    const int * j_A = A.GetJ();
    const double * a_A = A.GetData();
    const int ncol_A = A.Width();

    const int nAE = AE_row.Size();
    const int nrow = AE_row.NumNonZeroElems();
    const int ncol = AE_col.NumNonZeroElems();

    const int * i_AE_row = AE_row.GetI();
    const int * j_AE_row = AE_row.GetJ();
    const double * a_AE_row = AE_row.GetData(); //Should be all ones

    const int * i_AE_col = AE_col.GetI();
    const int * j_AE_col = AE_col.GetJ();
    const double * a_AE_col = AE_col.GetData(); //Should be all ones

    for(const double *a = a_AE_row; a != a_AE_row+nrow; ++a)
        PARELAG_TEST_FOR_EXCEPTION(
            *a != 1.0,
            std::logic_error,
            "Distribute(): AR_row(i,j) != 1.0.");

    for(const double *a = a_AE_col; a != a_AE_col+ncol; ++a)
        PARELAG_TEST_FOR_EXCEPTION(
            *a != 1.0,
            std::logic_error,
            "Distribute(): AR_col(i,j) != 1.0.");

    int * I = new int[nrow+1];
    I[0] = 0;

    int nnz(0);

    int currentARow(-1);

    Array<int> colMapper(ncol_A);
    colMapper = -1;
    //Compute the number of nnz of the matrix
    for(int iAE(0); iAE < nAE; ++iAE)
    {
        for(int jcol(i_AE_col[iAE]); jcol < i_AE_col[iAE+1]; ++jcol)
            colMapper[j_AE_col[jcol]] = iAE;

        // Find the number of nnz.
        for(int i(i_AE_row[iAE]); i<i_AE_row[iAE+1]; ++i)
        {
            currentARow = j_AE_row[i];
            for(const int * it = j_A+i_A[currentARow]; it != j_A+i_A[currentARow+1]; ++it)
                if( colMapper[*it] == iAE)
                    ++nnz;
            I[i+1] = nnz;
        }
    }

    // Reset colMapper to -1.
    colMapper = -1;

    //Allocate memory
    int * J = new int[nnz];
    double * Data = new double[nnz];

    //Fill in the matrix
    const double * it_a;
    int * it_J = J;
    double * it_Data = Data;

    for(int iAE(0); iAE < nAE; ++iAE)
    {
        for(int jcol(i_AE_col[iAE]); jcol < i_AE_col[iAE+1]; ++jcol)
            colMapper[j_AE_col[jcol]] = jcol;

        for(int i(i_AE_row[iAE]); i<i_AE_row[iAE+1]; ++i)
        {
            currentARow = j_AE_row[i];
            it_a = a_A+i_A[currentARow];
            for(const int * it = j_A+i_A[currentARow]; it != j_A+i_A[currentARow+1]; ++it, ++it_a)
                if( colMapper[*it] >= 0)
                {
                    *(it_J++) = colMapper[*it];
                    *(it_Data++) = *it_a;
                }
        }

        // Restore colMapper
        for(int jcol(i_AE_col[iAE]); jcol < i_AE_col[iAE+1]; ++jcol)
            colMapper[j_AE_col[jcol]] = -1;
    }

    return make_unique<SparseMatrix>(I, J, Data, nrow, ncol);
}

unique_ptr<SparseMatrix> DropEntriesFromSparseMatrix(
    const SparseMatrix & A, const DropEntryPredicate & drop )
{
    const int * i_A = A.GetI();
    const int * j_A = A.GetJ();
    const double * a_A = A.GetData();
    const int nrow = A.Size();
    const int ncol = A.Width();

    int * I = new int[nrow+1];

    const int * j_A_it = j_A;
    const int * end;
    const double * a_A_it = a_A;

    int nnz(0);
    for(int irow(0); irow < nrow; ++irow)
    {
        I[irow] = nnz;
        end = j_A + i_A[irow+1];
        for( ; j_A_it != end; ++j_A_it, ++a_A_it)
            if( !drop(irow, *j_A_it, *a_A_it) )
                ++nnz;
    }
    I[nrow] = nnz;

    if(nnz == 0)
        return make_unique<SparseMatrix>(I,new int[1],new double[1],nrow,ncol);

    int * J = new int[nnz];
    double * val = new double[nnz];

    int * j_it = J;
    double * val_it = val;

    j_A_it = j_A;
    a_A_it = a_A;

    for(int irow(0); irow < nrow; ++irow)
    {
        end = j_A + i_A[irow+1];
        for( ; j_A_it != end; ++j_A_it, ++a_A_it)
            if( !drop(irow, *j_A_it, *a_A_it) )
            {
                *(j_it++) = *j_A_it;
                *(val_it++) = *a_A_it;
            }
    }

    return make_unique<SparseMatrix>(I,J,val, nrow, ncol);
}
}//namespace parelag
