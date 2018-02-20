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

#ifndef BUILD3DHEXMESH_HPP__
#define BUILD3DHEXMESH_HPP__

#include <mfem.hpp>
#include "utilities/MemoryUtils.hpp"

namespace parelag
{
namespace testhelpers
{

std::unique_ptr<mfem::Mesh> Build3DHexMesh()
{
    return make_unique<mfem::Mesh>(2,2,2,mfem::Element::HEXAHEDRON,1);
}

}// namespace testhelpers
}// namespace parelag
#endif /* BUILD3DHEXMESH_HPP__ */
