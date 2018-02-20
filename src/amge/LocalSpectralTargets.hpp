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

#ifndef LOCALSPECTRALTARGETS_HPP_
#define LOCALSPECTRALTARGETS_HPP_

#include <memory>
#include <vector>

#include "amge/DOFAgglomeration.hpp"
#include "topology/Topology.hpp"
#include "topology/TopologyTable.hpp"

namespace parelag
{
/// Computes spectral targets on all agglomerated entities of a
/// certain type (usually agglomerated elements). Entity/element
/// matrices (in "DG-like" form) are needed and it assembles the
/// agglomerated entity/element matrices itself.
std::vector<std::unique_ptr<MultiVector>>
ComputeLocalSpectralTargetsFromEntity(
    AgglomeratedTopology::Entity entity_type,
    mfem::SparseMatrix& M_e,
    DofAgglomeration *dofAgg,
    double rel_tol,
    int max_evects );

/// Computes spectral targets on all agglomerated entities of a
/// certain type (usually agglomerated elements). Assembled
/// agglomerated entity/element matrices (in "DG-like" form) are
/// needed.
std::vector<std::unique_ptr<MultiVector>>
ComputeLocalSpectralTargetsFromAEntity(
    AgglomeratedTopology::Entity entity_type,
    mfem::SparseMatrix& M_AEntity,
    DofAgglomeration *dofAgg,
    double rel_tol,
    int max_evects );

/// Compute spectral targets for the Hdiv-L2 pair on all agglomerated
/// elements. Assembled agglomerated entity/element matrices (in
/// "DG-like" form) are needed.
void ComputeLocalHdivL2SpectralTargetsFromAEntity(
    mfem::SparseMatrix& M_AEntity,
    mfem::SparseMatrix& D_AEntity,
    mfem::SparseMatrix& W_AEntity,
    mfem::SparseMatrix& Q_AEntity,
    DofAgglomeration *HdivdofAgg,
    DofAgglomeration *L2dofAgg,
    TopologyTable & AF_AE,
    double rel_tol,
    int max_evects,
    std::vector<std::unique_ptr<MultiVector>>& localHdivTracetargets,
    std::vector<std::unique_ptr<MultiVector>>& localL2targets);
//mfem::Array<MultiVector *>* localHdivTracetargets,
//mfem::Array<MultiVector *>* localL2targets );
}//namespace parelag
#endif /* LOCALSPECTRALTARGETS_HPP_ */
