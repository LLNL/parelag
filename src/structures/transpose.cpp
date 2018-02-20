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

#include "transpose.hpp"

#include "utilities/elagError.hpp"
#include "utilities/MemoryUtils.hpp"

using namespace mfem;
using std::unique_ptr;

namespace parelag
{
void transposePartitioning(const Array<int> & partitioning,
                           Array<int> & i_MIS_entities,
                           Array<int> & j_MIS_entities)
{
    elag_assert(partitioning.Max() == i_MIS_entities.Size() - 2);

    i_MIS_entities = 0;
    int * counts = i_MIS_entities.GetData() + 2;
    for(int i = 0; i < partitioning.Size(); ++i)
    {
        elag_assert(partitioning[i] >= -1);
        ++(counts[partitioning[i]]);
    }
    i_MIS_entities[1] = 0;
    i_MIS_entities.PartialSum();

    elag_assert( j_MIS_entities.Size() == i_MIS_entities.Last() );

    counts = i_MIS_entities.GetData() + 1;
    int irow;
    for(int i = 0; i < partitioning.Size(); ++i)
    {
        if( (irow = partitioning[i]) > -1 )
            j_MIS_entities[ counts[irow]++ ] = i;
    }

    elag_assert(i_MIS_entities.Last() == i_MIS_entities[i_MIS_entities.Size()-2]);

}

unique_ptr<SerialCSRMatrix> transpose(const Array<int> & j,
                                      Vector & a,
                                      int nrowsOut)
{
    elag_assert(j.Max() <= nrowsOut );
    elag_assert(j.Size() == a.Size() );

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

    counts = i_out + 1;
    int irow;
    for(int i = 0; i < j.Size(); ++i)
    {
        if( (irow = j[i]) > -1 )
        {
            j_out[ counts[irow] ] = i;
            a_out[ counts[irow] ] = a(i);
            counts[irow]++;
        }
    }

    return make_unique<SerialCSRMatrix>(i_out,j_out,a_out,nrowsOut,j.Size());

}

unique_ptr<SerialCSRMatrix> transpose(const Array<int> & j, int nrowsOut)
{
    elag_assert(j.Max() < nrowsOut );

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

    return make_unique<SerialCSRMatrix>(i_out,j_out,a_out,nrowsOut,j.Size());
}
}//namespace parelag
