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

#include "CartesianPartitioner.hpp"

namespace parelag
{
using namespace mfem;

void CartesianIJK::SetupCartesianIJK(const Mesh & mesh, const Array<int> & N, const Ordering & ordering, Array<LogicalCartesian> & ijk)
{
    setupCartesianIJK<LogicalCartesian>(mesh,N,ordering,ijk);
}

void CartesianIJK::SetupCartesianIJKMaterialId(const Mesh & mesh, const Array<int> & N, const Ordering & ordering, Array<LogicalCartesianMaterialId> & ijk)
{
    setupCartesianIJK<LogicalCartesianMaterialId>(mesh,N,ordering,ijk);
    for(int i(0); i < mesh.GetNE(); ++i)
        ijk[i].materialId = mesh.GetElement(i)->GetAttribute();
}
}//namespace parelag
