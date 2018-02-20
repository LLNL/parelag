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


#include "linalg/sparse_direct/ParELAG_SuperLUSolver.hpp"
#include "linalg/sparse_direct/ParELAG_SuperLUSolver_impl.hpp"

namespace parelag
{

// This is really the only meaningful instantiation for now...
template class SuperLUSolver<double>;

#ifdef PARELAG_ENABLE_SINGLE_PRECISION_DIRECT_SOLVES
template class SuperLUSolver<float>;
#endif /* PARELAG_ENABLE_SINGLE_PRECISION_DIRECT_SOLVES */

#ifdef PARELAG_ENABLE_COMPLEX
template class SuperLUSolver<std::complex<float>>;
template class SuperLUSolver<std::complex<double>>;

// If users *happen* to magically already be using SLU types
template class SuperLUSolver<SLU::SCX::complex>>;
template class SuperLUSolver<SLU::DCX::doublecomplex>>;
#endif /* PARELAG_ENABLE_COMPLEX */

}// namespace parelag
