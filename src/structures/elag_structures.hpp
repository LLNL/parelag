/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#ifndef ELAG_STRUCTURES_HPP_
#define ELAG_STRUCTURES_HPP_

#include <mfem.hpp>
using namespace mfem;

#include "../hypreExtension/hypreExtension.hpp"
#include "../utilities/elag_utilities.hpp"

typedef mfem::SparseMatrix SerialCSRMatrix;
typedef mfem::HypreParMatrix ParallelCSRMatrix;

#include "BooleanMatrix.hpp"

int connectedComponents(Array<int> & partitioning, SerialCSRMatrix & conn );
int connectedComponents(Array<int> & partitioning, SerialCSRMatrix & conn, Array<int> & materialSubdomains );
SerialCSRMatrix * findMinimalIntersectionSets(SerialCSRMatrix & Z, double skipDiagEntryLessThan);

hypre_ParCSRMatrix * ParUnique(hypre_ParCSRMatrix * ZZ, Array<int> & trueZ_start);

void transposePartitioning(const Array<int> & partitioning, Array<int> & i_MIS_entities, Array<int> & j_MIS_entities);
SerialCSRMatrix * transpose(const Array<int> & j, Vector & a, int nrowsOut);
SerialCSRMatrix * transpose(const Array<int> & j, int nrowsOut);

#include "SharingMap.hpp"
#include "Coloring.hpp"
#include "ProcessorCoarsening.hpp"

#endif /* ELAG_STRUCTURES_HPP_ */
