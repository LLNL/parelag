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


#include "linalg/sparse_direct/ParELAG_StrumpackSolver.hpp"
#include "linalg/sparse_direct/ParELAG_StrumpackSolver_impl.hpp"

namespace parelag
{

template class StrumpackSolver<double,double,int>;

#ifdef PARELAG_ENABLE_COMPLEX
template class StrumpackSolver<std::complex<double>,double,int>;
#endif /* PARELAG_ENABLE_COMPLEX */

}// namespace parelag
