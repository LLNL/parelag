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

#ifndef HYBRIDHDIVL2_HPP_
#define HYBRIDHDIVL2_HPP_

#include <memory>
#include <vector>

#include <mfem.hpp>

#include "amge/DeRhamSequence.hpp"
#include "amge/DofHandler.hpp"
#include "topology/Topology.hpp"
#include "linalg/dense/ParELAG_DenseInverseCalculator.hpp"

namespace parelag
{
// This class assumes no flow boundary condition on the boundary where
// essential boundary condition is imposed
class HybridHdivL2
{
public:
    HybridHdivL2(const std::shared_ptr<DeRhamSequence>& sequence,
                 bool IsSameOrient_,
                 double W_weight_,
                 mfem::Array<int>& ess_HdivDofs_,
                 std::shared_ptr<mfem::Vector> elemMatrixScaling_=nullptr);

    HybridHdivL2(HybridHdivL2 const&) = delete;
    HybridHdivL2(HybridHdivL2&&) = delete;

    HybridHdivL2& operator=(HybridHdivL2 const&) = delete;
    HybridHdivL2& operator=(HybridHdivL2&&) = delete;

    virtual ~HybridHdivL2();

    // Transform original RHS to the RHS of the hybridized system and
    // obtain essential data for the hybridized system from original RHS
    void RHSTransform(const mfem::BlockVector& OriginalRHS,
    		mfem::Vector& HybridRHS, mfem::Vector& essentialData);
    // This function assumes the offsets of RecoveredSol have been defined
    void RecoverOriginalSolution(const mfem::Vector & HybridSol,
                                 mfem::BlockVector & RecoveredSol);
    mfem::SparseMatrix * GetHybridSystem() { return HybridSystem.get(); }
    DofHandler * GetDofMultiplier() { return dofMultiplier.get(); }

    // So returning the address to a private data member is a good
    // idea? Ok.
    const std::vector<std::unique_ptr<mfem::DenseMatrix>>& GetHybrid_el() const noexcept
    {
        return Hybrid_el;
    }

    const mfem::Array<int>& GetEssentialMultiplierDofs() const noexcept
	{
        return ess_MultiplierDofs;
	}

    const mfem::Vector& GetRescaling() const noexcept
    {
        return CCT_inv_CBT1;
    }

private:

    bool IsSameOrient;

    // The system being solved is [ M B^T; B -W_weight * W ]
    double W_weight;

    void AssembleHybridSystem();

    std::unique_ptr<mfem::SparseMatrix> HybridSystem;

    AgglomeratedTopology * topo;

    int nDimension;

    std::unique_ptr<mfem::SparseMatrix> facet_elem;

    DofHandler * dofHdiv;
    DofHandler * dofL2;
    std::unique_ptr<DofHandler> dofMultiplier;

    // This guy is just a view of another matrix
    mfem::SparseMatrix * M_el;

    std::unique_ptr<mfem::SparseMatrix> W;
    std::unique_ptr<mfem::SparseMatrix> B;

    std::shared_ptr<mfem::Vector> elemMatrixScaling;

    std::unique_ptr<mfem::SparseMatrix> Multiplier_Hdiv_;

    std::vector<std::unique_ptr<mfem::DenseMatrix>> Hybrid_el;
    std::vector<std::unique_ptr<mfem::DenseMatrix>> AinvCT;
    std::vector<std::unique_ptr<mfem::Vector>> Ainv_f;
    std::vector<std::unique_ptr<DenseInverseCalculator>> Ainv;
    std::vector<std::unique_ptr<mfem::DenseMatrix>> A_el;

    const mfem::Vector& L2_const_rep_;
    mfem::Vector CCT_inv_CBT1;

    mfem::Array<int> ess_HdivDofs;
    mfem::Array<int> ess_MultiplierDofs;
};
}//namespace parelag
#endif /* HYBRIDHDIVL2_HPP_ */
