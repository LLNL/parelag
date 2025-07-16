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

#ifndef LINEARPARTITION_HPP_
#define LINEARPARTITION_HPP_

#include <mfem.hpp>

namespace parelag
{


void DoLinearPartition(int nElements, int nParts, mfem::Array<int> & partitioning);

}



#endif // LINEARPARTITION_HPP_