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

#ifndef LOGICALPARTITIONER_HPP_
#define LOGICALPARTITIONER_HPP_

#include <mfem.hpp>

#include "elag_typedefs.hpp"
#include "topology/TopologyTable.hpp"
#include "utilities/elagError.hpp"

namespace parelag
{
//! @class
/*!
 * @brief this class allows for easy implementation of various agglomeration strategies.
 *        For any given coarsening operator, 2 methods need to be implemented to
 *        use LogicalPartitioner, namely operator== [for T]
 *        and operator() [for CoarsenLogicalOp].
 *        (TODO---don't think I like the semantics here, operator() is not very
 *         instructive---how about InheritPartitioning() or something?)
 *
 *        CoarsenLogicalOp is either CoarsenMetisMaterialID or CoarsenLogicalCartesianOperatorMaterialId
 *        T is either MetisMaterialID or LogicalCartesianMaterialId
 *
 *        See CartesianPartitioner and MetisGraphPartitioner for examples of
 *        how to use LogicalPartitioner
 *
 */
class LogicalPartitioner
{
public:
    LogicalPartitioner() { }

    template<class T, class CoarsenLogicalOp>
    void Partition(const SerialCSRMatrix & elem_elem,
                   const mfem::Array<T> & fine_logical,
                   const CoarsenLogicalOp & coarsening_logical,
                   mfem::Array<int> & partitioning)
    {
        int i, j, k;
        int num_elem;

        num_elem = elem_elem.Height();
        const int * i_elem_elem = const_cast<int *>(elem_elem.GetI());
        const int * j_elem_elem = const_cast<int *>(elem_elem.GetJ());

        partitioning.SetSize(num_elem);
        int num_parts = 0;
        for (i = 0; i < num_elem; i++)
            partitioning[i] = -1;

        mfem::Array<int> elem_stack(num_elem);
        int stack_p, stack_top_p, elem;

        stack_p = 0;
        stack_top_p = 0;  // points to the first unused element in the stack

        T coarse_logical_i, coarse_logical_k;
        for (elem = 0; elem < num_elem; elem++)
        {
            // The element was already considered, go to the next one
            if (partitioning[elem] >= 0)
                continue;

            // This element will start a new partition
            partitioning[elem] = num_parts;
            ++num_parts;

            elem_stack[stack_top_p++] = elem;

            for ( ; stack_p < stack_top_p; stack_p++)
            {
                i = elem_stack[stack_p];
                coarsening_logical(fine_logical[i], coarse_logical_i);

                for (j = i_elem_elem[i]; j < i_elem_elem[i+1]; j++)
                {
                    k = j_elem_elem[j];
                    coarsening_logical(fine_logical[k], coarse_logical_k);
                    if (coarse_logical_i == coarse_logical_k)
                    {
                        if (partitioning[k] < 0)
                        {
                            PARELAG_ASSERT(partitioning[i] == num_parts-1);
                            partitioning[k] = partitioning[i];
                            elem_stack[stack_top_p++] = k;
                        }
                    }
                }
            }
        }
    }

    /// ComputeCoarseLogical prepares the logical information needed to
    /// agglomerate from a coarse level to an even coarser level
    template<class T, class CoarsenLogicalOp>
    void ComputeCoarseLogical(
        const CoarsenLogicalOp & coarsen_map,
        const TopologyTable & AE_element,
        const mfem::Array<T> & fine_logical,
        mfem::Array<T> & coarse_logical)
    {
        const int * i_AE_element = const_cast<int *>(AE_element.GetI());
        const int * j_AE_element = const_cast<int *>(AE_element.GetJ());

        coarse_logical.SetSize(AE_element.Height());

        for(int iAE(0); iAE < AE_element.Height(); ++iAE)
        {
            coarsen_map( fine_logical[ j_AE_element[ i_AE_element[iAE] ]], coarse_logical[iAE] );
#ifdef ELAG_DEBUG
            for(int k(i_AE_element[iAE]); k < i_AE_element[iAE+1]; ++k)
            {
                T tmp;
                coarsen_map( fine_logical[j_AE_element[k]], tmp );
                elag_assert(tmp==coarse_logical[iAE]);
            }
#endif
        }
    }

    ~LogicalPartitioner() { }
private:
};

/// Distribute() is used copy logical information such as ijk indices from serial to parallel
template<class T>
void Distribute(MPI_Comm comm, const mfem::Array<T> & globalLogical, const mfem::Array<int> & partitioning, mfem::Array<T> & localLogical)
{
    int myrank;
    MPI_Comm_rank(comm, &myrank);
    int myelem = 0;
    for(int i = 0; i < partitioning.Size(); ++i)
    {
        if(partitioning[i] == myrank)
        {
            localLogical[myelem] = globalLogical[i];
            ++myelem;
        }
    }
}
}//namespace parelag
#endif /* LOGICALPARTITIONER_HPP_ */
