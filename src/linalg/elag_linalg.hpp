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

#ifndef ELAG_LINALG_HPP_
#define ELAG_LINALG_HPP_

#include "dense/ParELAG_Eigensolver.hpp"
#include "dense/ParELAG_InnerProduct.hpp"
#include "dense/ParELAG_LAPACK.hpp"
#include "dense/ParELAG_LDLCalculator.hpp"
#include "dense/ParELAG_QDQCalculator.hpp"
#include "dense/ParELAG_MultiVector.hpp"
#include "dense/ParELAG_SVDCalculator.hpp"

#include "legacy/ParELAG_AuxHypreSmoother.hpp"
#include "legacy/ParELAG_HypreExtension.hpp"
#include "legacy/ParELAG_MLDivFree.hpp"
#include "legacy/ParELAG_MLHiptmairSolver.hpp"

#include "solver_core/ParELAG_SymmetrizedUmfpack.hpp"
#include "solver_core/ParELAG_SolverLibrary.hpp"

#include "utilities/ParELAG_MatrixUtils.hpp"
#include "utilities/ParELAG_SubMatrixExtraction.hpp"
#include "utilities/ParELAG_MfemBlockOperator.hpp"

#endif // ELAG_LINALG_HPP_
