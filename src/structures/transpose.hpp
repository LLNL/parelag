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

#ifndef STRUCTURES_TRANSPOSE_HPP_
#define STRUCTURES_TRANSPOSE_HPP_

#include <memory>

#include <mfem.hpp>

#include "elag_typedefs.hpp"

namespace parelag
{
// FIXME (trb 12/29/15): Do these functions actually get called?
// 'grep' can't find them outside of this hpp and the
// corresponding cpp...

void transposePartitioning(const mfem::Array<int> & partitioning,
                           mfem::Array<int> & i_MIS_entities,
                           mfem::Array<int> & j_MIS_entities);

std::unique_ptr<SerialCSRMatrix> transpose(const mfem::Array<int> & j,
                                           mfem::Vector & a,
                                           int nrowsOut);

std::unique_ptr<SerialCSRMatrix> transpose(const mfem::Array<int> & j,
                                           int nrowsOut);
}//namespace parelag
#endif
