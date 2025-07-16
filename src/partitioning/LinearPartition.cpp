/*
  Copyright (c) 2024, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-745557. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#include "LinearPartition.hpp"
#include "utilities/elagError.hpp"


namespace parelag
{

void DoLinearPartition(int nElements, int nParts, mfem::Array<int> & partitioning)
{
    if (nElements == 0 && nParts == 0)
        return;

    elag_assert(nElements > nParts);
    elag_assert(nElements % nParts == 0);
    partitioning.SetSize(nElements);

    const int coarseningFactor = nElements / nParts;
    int * p = partitioning.GetData();

    int j = 0;
    for(int i = 0; i < nParts; ++i, j+=coarseningFactor)
        std::fill_n(&p[j], coarseningFactor, i);
    elag_assert(j == nElements);
}

}
