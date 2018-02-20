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

#include "minimalIntersectionSet.hpp"

#include <numeric>

#include "utilities/elagError.hpp"
#include "utilities/MemoryUtils.hpp"
#include "utilities/mpiUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

/**
 * findMinimalIntersectionSets
 * Inputs:
 *
 *   \param Z the intersection matrix (should be square) [was bedeutet
 *   "intersection matrix"? something like face_AE_face].
 *
 *   \param skipDiagEntryLessThan if the diagonal entry is less that
 *   this quantity, the entity will not belong to any MIS.
 *
 * Ouput: the entity_MIS relationship
 *
 *   \param entity_MIS: SparseMatrix where entry i,j is non-zero if
 *   entity i belongs to MIS j.  The numerical value of the matrix is
 *   +/-1, according to the relative orientation of MIS and entity.
 */
unique_ptr<SerialCSRMatrix> findMinimalIntersectionSets(
    SerialCSRMatrix & Z, double skipDiagEntryLessThan)
{
    using size_type = int;
    using ordinal_type = int;
    using value_type = double;
    using magnitude_type = double;

    elag_assert(Z.Size() == Z.Width());

    constexpr magnitude_type tol = 1e-10;
    const size_type nEntities = Z.Size();

    // Grab the diagonal entries
    Vector diagZ(nEntities);
    Z.GetDiag(diagZ);

    // Create the "row_ptr" of the CSR matrix
    size_type * i_entity_MIS = new size_type[nEntities+1];

    size_type nnz = 0;
    for (size_type i = 0; i < nEntities; ++i)
    {
        i_entity_MIS[i] = nnz;
        // An entity belongs to a MIS iff corresponding diagonal is
        // big enough.
        if (diagZ(i) - skipDiagEntryLessThan > -tol)
            ++nnz;
    }
    i_entity_MIS[nEntities] = nnz;

    // Create the "col_ind" and "value" array for CSR
    ordinal_type* j_entity_MIS = new ordinal_type[nnz];
    value_type * a_entity_MIS = new value_type[nnz];

    std::fill_n(j_entity_MIS, nnz, -1);

    const size_type * Z_I = Z.GetI();
    const ordinal_type * Z_J = Z.GetJ();
    const value_type * Z_D = Z.GetData();

    ordinal_type current_MIS = 0;
    for (size_type iEntity = 0; iEntity < nEntities; ++iEntity)
    {
        const value_type& Z_ii = diagZ(iEntity);
        // FIXME (trb 12/09/15): See NOTE below.
        /* NOTE: in the if statement first we check whenever iEntity
         * belongs to a minimal intersection set (Z_ii >=
         * skipDiagEntryLessThan) then we check if we have already
         * found the MIS which contains iEntity (j_entity_MIS[
         * i_entity_MIS[iEntity] ] == -1).  If by mistake one invert
         * the ordering of the two conditions valgrind may complain:
         * in fact if the last entity (iEntity = nEntities-1) does not
         * belong to any minimal intersection set (i.e. the first
         * condition is false), then i_entity_MIS[iEntity] will be
         * equal to nnz and therefore outside the area of memory
         * pointed by j_entity_MIS.
         */

        // If this is a valid entry and we have NOT found its MIS yet...
        if (Z_ii - skipDiagEntryLessThan > -tol &&
            j_entity_MIS[i_entity_MIS[iEntity]] == -1)
        {
            for (size_type ind = Z_I[iEntity]; ind < Z_I[iEntity+1]; ++ind)
            {
                // note that we do not skip self-interaction; that is,
                // at some point jEntity will be iEntity
                const size_type& jEntity = Z_J[ind];
                const value_type& Z_ij = Z_D[ind];
                const value_type& Z_jj = diagZ(jEntity);
                if (fabs(Z_jj - Z_ii) < tol)     //  Z_ij (=Z_ji)
                {
                    // if (Z_ij == +/- Z_ii)
                    if (fabs(Z_ij - Z_ii) < tol || fabs(Z_ij + Z_ii) < tol)
                    {
                        // the following are going into *rows* of the
                        // output matrix corresponding to the current
                        // *column* of the input matrix Z
                        j_entity_MIS[i_entity_MIS[jEntity]] = current_MIS;
                        a_entity_MIS[i_entity_MIS[jEntity]] = Z_ij / Z_ii;
                    }
                }
            }
            ++current_MIS;
        }
    }

    return make_unique<SerialCSRMatrix>(
        i_entity_MIS, j_entity_MIS, a_entity_MIS, nEntities, current_MIS);
}


inline int getMin(HYPRE_Int * it,
                  HYPRE_Int * it_end,
                  HYPRE_Int * map,
                  HYPRE_Int defaultValue)
{
    int min = defaultValue;

    for (; it != it_end; ++it)
        if (map[*it] < min)
            min = map[*it];

    return min;
}

/*
  ZZ is a symmetric matrix, such that ZZ(i,i) != 0, and if ZZ(i,j) =
  +- ZZ(i,i) if i and j correspond to the same trueZ, 0
  otherwise. NOTE hypre_diag(ZZ) should be a diagonal matrix.

  At the output we construct Z_trueZ, where Z_true(i, any_fix_k)
  and Z_true(j, any_fix_k) are both non zeros if i and j correspond
  to the same trueZ k. The orientation in Z_trueZ depends on the
  reciprocal orientation of ZZ(i,j).

  We assign the trueZ to the processor of lower Pid. The orientation
  of trueZ is the same Z for the master processor.
*/
unique_ptr<ParallelCSRMatrix> ParUnique(
    hypre_ParCSRMatrix * ZZ, Array<int> & trueZ_start)
{
    hypre_CSRMatrix * diag_ZZ = hypre_ParCSRMatrixDiag(ZZ);
    hypre_CSRMatrix * offd_ZZ = hypre_ParCSRMatrixOffd(ZZ);

    // Check that diag_ZZ is actually diagonal
    elag_assert(hypre_CSRMatrixNumRows(diag_ZZ) ==
                hypre_CSRMatrixNumCols(diag_ZZ));
    elag_assert(hypre_CSRMatrixNumRows(diag_ZZ) ==
                hypre_CSRMatrixNumNonzeros(diag_ZZ));
#ifdef ELAG_DEBUG
    {
        HYPRE_Int * i_diag_ZZ = hypre_CSRMatrixI(diag_ZZ);
        HYPRE_Int * j_diag_ZZ = hypre_CSRMatrixJ(diag_ZZ);
        double    * a_diag_ZZ = hypre_CSRMatrixData(diag_ZZ);
        int localSize        = hypre_CSRMatrixNumRows(diag_ZZ);
        int nnz = hypre_CSRMatrixNumNonzeros(diag_ZZ);

        for (int irow = 0; irow < localSize+1; ++irow)
            elag_assert(i_diag_ZZ[irow] == irow);

        for (int innz = 0; innz < nnz; ++innz)
            elag_assert(j_diag_ZZ[innz] == innz);

        for (int innz = 0; innz < nnz; ++innz)
        {
            try
            {
                elag_assert(fabs(a_diag_ZZ[innz] - 1.) < 1e-10);
            } catch (int)
            {
                std::cout << "fabs(a_diag_ZZ[" << innz << "] - 1.) = "
                          << fabs(a_diag_ZZ[innz] - 1.) << "\n";
            }
        }
    }
#endif

    int localSize        = hypre_CSRMatrixNumRows(diag_ZZ);
    // int myFirstGlobalZ   = hypre_ParCSRMatrixFirstRowIndex(ZZ);
    int myLastGlobalZ    = hypre_ParCSRMatrixLastRowIndex(ZZ);
    HYPRE_Int * offdGlobalZmap  = hypre_ParCSRMatrixColMapOffd(ZZ);

    HYPRE_Int * i_offd_ZZ = hypre_CSRMatrixI(offd_ZZ);
    HYPRE_Int * j_offd_ZZ = hypre_CSRMatrixJ(offd_ZZ);

    // Create a selector matrix S of dimension (localSize, localTrueSize).
    // nnz(S) = localTrueSize;
    // S(i,j) != 0 if the local Z i is matched with the local TrueZ j.
    // TrueZ belongs to this processor if this processor owns the Z
    // with lowest i.d. that is matched to TrueZ.

    int localTrueSize = 0;
    int * i_S = new int[localSize+1];

    for (int i = 0; i < localSize; ++i)
    {
        i_S[i] = localTrueSize;
        if (myLastGlobalZ < getMin(j_offd_ZZ + i_offd_ZZ[i],
                                   j_offd_ZZ + i_offd_ZZ[i+1],
                                   offdGlobalZmap, myLastGlobalZ+1))
            localTrueSize++;
    }
    i_S[localSize] = localTrueSize;

    int * j_S = new int[localTrueSize];
    double * a_S = new double[localTrueSize];

    std::iota(j_S, j_S+localTrueSize, 0);
    std::fill_n(a_S, localTrueSize, 1.);

    SerialCSRMatrix S(i_S, j_S, a_S, localSize, localTrueSize);

    MPI_Comm comm = hypre_ParCSRMatrixComm(ZZ);
    ParPartialSums_AssumedPartitionCheck(comm , localTrueSize, trueZ_start);

    ParallelCSRMatrix Sd(
        comm, hypre_ParCSRMatrixGlobalNumRows(ZZ), trueZ_start.Last(),
        hypre_ParCSRMatrixRowStarts(ZZ), trueZ_start.GetData(), &S);

    return make_unique<ParallelCSRMatrix>(hypre_ParMatmul(ZZ,Sd));
}
}//namespace parelag
