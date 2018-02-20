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

#ifdef ParELAG_ENABLE_OPENMP
#include <omp.h>
#endif

#include "ParELAG_MLDivFree.hpp"

#include "ParELAG_AuxHypreSmoother.hpp"

#include "linalg/legacy/ParELAG_HypreExtension.hpp"
#include "linalg/solver_core/ParELAG_SymmetrizedUmfpack.hpp"
#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "linalg/utilities/ParELAG_SubMatrixExtraction.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;

MLDivFree::MLDivFree(
    const std::shared_ptr<BlockMatrix>& A_,
    std::vector<shared_ptr<DeRhamSequence> >& seqs,
    Array<int> & label_ess)
    : sequence_(seqs),
      l2form_(seqs[0]->GetNumberOfForms()-1),
      hdivform_(seqs[0]->GetNumberOfForms()-2),
      hcurlform_(seqs[0]->GetNumberOfForms()-3),
      nLevels_(seqs.size()),
      arithmeticTrueComplexity_(0),
      arithmeticComplexity_(0),
      Al_(nLevels_),
      trueAl_(0),
      Cl_(0),
      P_(nLevels_),
      trueP_(nLevels_),
      AE_dof(0),
      levelTrueStart(0),
      levelTrueStartMultiplier(0),
      levelStart(0),
      levelStartMultiplier(0),
      dof_is_shared_among_AE_data(0),
      trueRhs_data(0),
      trueSol_data(0),
      rhs_data(0),
      sol_data(0),
      essnullspace_data(0),
      t_data(0),
      numerical_zero(1e-6)
{
    SetBlockMatrix(A_);

    Al_[0] = A_;

    AE_dof.resize(nLevels_-1);
    P_.resize(nLevels_-1);
    trueP_.resize(nLevels_-1);
    Pi_.resize(nLevels_-1);
    if (label_ess.Max())
        P00_to_be_deleted_.resize(nLevels_-1);

    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    for (int i(0); i < nLevels_-1; ++i)
    {
        Array<int> dof_offsets(3), elem_offsets(2);

        dof_offsets[0] = 0;
        dof_offsets[1] = seqs[i]->GetAEntityDof(at_elem, hdivform_).Width();
        dof_offsets[2] =
            seqs[i]->GetAEntityDof(at_elem, hdivform_).Width() +
            seqs[i]->GetAEntityDof(at_elem, l2form_).Width();

        elem_offsets[0] = 0;
        elem_offsets[1] = seqs[i]->GetAEntityDof(at_elem, hdivform_).Size();

        // FIXME (trb 06/28/16): I had to hack this to make the change
        // I wanted to make to DofAgglomeration, but it's terrible and
        // should be more cleanly resolved in the (very near) future.
        SparseMatrix& ae_dof00 =
            const_cast<SparseMatrix&>(seqs[i]->GetAEntityDof(at_elem, hdivform_));
        SparseMatrix& ae_dof01 =
            const_cast<SparseMatrix&>(seqs[i]->GetAEntityDof(at_elem, l2form_));
        AE_dof[i] = make_unique<BlockMatrix>(elem_offsets,dof_offsets);
        AE_dof[i]->SetBlock(0,0, &ae_dof00);
        AE_dof[i]->SetBlock(0,1, &ae_dof01);

        AE_dof[i]->RowOffsets().MakeDataOwner();
        AE_dof[i]->ColOffsets().MakeDataOwner();

        dof_offsets.LoseData();
        elem_offsets.LoseData();

        Array<int> fine_offsets(3), coarse_offsets(3);

        fine_offsets[0] = 0;
        fine_offsets[1] = seqs[i]->GetP(hdivform_)->Size();
        fine_offsets[2] = seqs[i]->GetP(hdivform_)->Size() +
            seqs[i]->GetP(l2form_)->Size();

        coarse_offsets[0] = 0;
        coarse_offsets[1] = seqs[i]->GetP(hdivform_)->Width();
        coarse_offsets[2] = seqs[i]->GetP(hdivform_)->Width() +
            seqs[i]->GetP(l2form_)->Width();

        P_[i] = make_unique<BlockMatrix>(fine_offsets,coarse_offsets);
        if (label_ess.Max())
        {
            P00_to_be_deleted_[i] = seqs[i]->GetP(hdivform_, label_ess);
            P_[i]->SetBlock(0,0, P00_to_be_deleted_[i].get());
        }
        else
            P_[i]->SetBlock(0,0, seqs[i]->GetP(hdivform_));
        P_[i]->SetBlock(1,1, seqs[i]->GetP(l2form_));

        Pi_[i] = make_unique<BlockMatrix>(coarse_offsets,fine_offsets);
        seqs[i]->GetPi(hdivform_)->ComputeProjector();
        seqs[i]->GetPi(l2form_)->ComputeProjector();
        Pi_[i]->SetBlock(0,0, const_cast<SparseMatrix*>(&(seqs[i]->GetPi(hdivform_)->GetProjectorMatrix())));
        Pi_[i]->SetBlock(1,1, const_cast<SparseMatrix*>(&(seqs[i]->GetPi(l2form_)->GetProjectorMatrix())));

        P_[i]->RowOffsets().MakeDataOwner();
        P_[i]->ColOffsets().MakeDataOwner();
        fine_offsets.LoseData();
        coarse_offsets.LoseData();

        Array<int> true_fine_offsets(3), true_coarse_offsets(3);

        true_fine_offsets[0] = 0;
        true_fine_offsets[1] = seqs[i]->GetNumberOfTrueDofs(hdivform_);
        true_fine_offsets[2] = seqs[i]->GetNumberOfTrueDofs(hdivform_) + seqs[i]->GetNumberOfTrueDofs(l2form_);

        true_coarse_offsets[0] = 0;
        true_coarse_offsets[1] = seqs[i+1]->GetNumberOfTrueDofs(hdivform_);
        true_coarse_offsets[2] = seqs[i+1]->GetNumberOfTrueDofs(hdivform_) + seqs[i+1]->GetNumberOfTrueDofs(l2form_);

        trueP_[i] = make_unique<BlockOperator>(
            true_fine_offsets, true_coarse_offsets);
        trueP_[i]->owns_blocks = 1;
        // NOTE: .release() because owns_blocks = 1
        trueP_[i]->SetBlock(
            0, 0, seqs[i]->ComputeTrueP(hdivform_, label_ess).release());
        trueP_[i]->SetBlock(1,1, seqs[i]->ComputeTrueP(l2form_).release());
        Swap(trueP_[i]->RowOffsets(), true_fine_offsets);
        Swap(trueP_[i]->ColOffsets(), true_coarse_offsets);
    }

    std::unique_ptr<SparseMatrix> W(seqs[0]->ComputeMassOperator(l2form_));
    ConstantCoefficient one(1.);
    Vector ones(W->Size());
    seqs[0]->ProjectCoefficient(l2form_, one, ones);

    Cl_.resize(nLevels_);
    for (int i(0); i < nLevels_; ++i)
        Cl_[i] = seqs[i]->ComputeTrueDerivativeOperator(hcurlform_, label_ess);

    Build(ones, *W);

    SetCycle(FULL_SPACE);
}

void MLDivFree::SetOperator(const Operator &)
{
    std::cout << "MLDivFree::SetOperator is ignored \n";
}

void MLDivFree::SetBlockMatrix(const std::shared_ptr<BlockMatrix> A_)
{
    Al_[0] = A_;

    elag_assert(
        Al_[0]->Height() == sequence_[0]->GetNumberOfDofs(hdivform_) + sequence_[0]->GetNumberOfDofs(l2form_));
    elag_assert(Al_[0]->GetBlock(0,0).Width() == Al_[0]->GetBlock(0,0).Height());
    elag_assert(Al_[0]->Height() == Al_[0]->Width());
}

void MLDivFree::Build(const Vector & ess_nullspace_p, const SparseMatrix & mass_p)
{
    levelTrueStart.SetSize(nLevels_+1);
    levelTrueStart[0] = arithmeticTrueComplexity_ = 0;
    for (int i(0); i < nLevels_; ++i)
    {
        levelTrueStart[i+1] =
            (arithmeticTrueComplexity_ += sequence_[i]->GetNumberOfTrueDofs(hdivform_) +
             sequence_[i]->GetNumberOfTrueDofs(l2form_));
    }

    trueRhs_data.SetSize(arithmeticTrueComplexity_);
    trueSol_data.SetSize(arithmeticTrueComplexity_);

    levelStart.SetSize(nLevels_+1);
    levelStart[0] = arithmeticComplexity_ = 0;
    for (int i(0); i < nLevels_; ++i)
    {
        levelStart[i+1] =
            (arithmeticComplexity_ += sequence_[i]->GetNumberOfDofs(hdivform_) +
             sequence_[i]->GetNumberOfDofs(l2form_));
    }

    dof_is_shared_among_AE_data.SetSize(arithmeticComplexity_);
    essnullspace_data.SetSize(arithmeticComplexity_);
    t_data.SetSize(arithmeticComplexity_);
    rhs_data.SetSize(levelStart[1]);
    sol_data.SetSize(levelStart[1]);

    levelTrueStartMultiplier.SetSize(nLevels_);
    for (int i(0); i < nLevels_; ++i)
    {
        levelTrueStartMultiplier[i] = levelTrueStart[i] +
            sequence_[i]->GetNumberOfTrueDofs(hdivform_);
    }

    levelStartMultiplier.SetSize(nLevels_);
    for (int i(0); i < nLevels_; ++i)
    {
        levelStartMultiplier[i] = levelStart[i] +
            sequence_[i]->GetNumberOfDofs(hdivform_);
    }

    Array<int> is_shared;
    for (int i(0); i < nLevels_-1; ++i)
    {
        is_shared.MakeRef(dof_is_shared_among_AE_data+levelStart[i],
                          levelStart[i+1] - levelStart[i]);
        computeSharedDof(i, is_shared);
    }

    for (int i(0); i < nLevels_-1; ++i)
        Al_[i+1] = this->PtAP(*Al_[i],*P_[i]);

    trueAl_.resize(nLevels_);
    for (int i(0); i < nLevels_; ++i)
    {
        SharingMap & maphdiv = sequence_[i]->GetDofHandler(hdivform_)->GetDofTrueDof();
        SharingMap & mapl2 = sequence_[i]->GetDofHandler(l2form_)->GetDofTrueDof();
        Array<int> trueblockoffsets(3);
        trueblockoffsets[0] = 0;
        trueblockoffsets[1] = maphdiv.GetTrueLocalSize();
        trueblockoffsets[2] = maphdiv.GetTrueLocalSize() + mapl2.GetTrueLocalSize();

        trueAl_[i] = make_unique<BlockOperator>(trueblockoffsets);
        trueAl_[i]->owns_blocks = 1;
        Swap(trueblockoffsets, trueAl_[i]->RowOffsets());
        trueAl_[i]->SetBlock(0,0, Assemble(maphdiv, Al_[i]->GetBlock(0,0), maphdiv).release());
        trueAl_[i]->SetBlock(0,1, Assemble(maphdiv, Al_[i]->GetBlock(0,1), mapl2).release());
        trueAl_[i]->SetBlock(1,0, Assemble(mapl2, Al_[i]->GetBlock(1,0), maphdiv).release());
    }

    Maux_.resize(nLevels_);
    for (int i(0); i < nLevels_; ++i)
    {
        Maux_[i] = make_unique<AuxHypreSmoother>(
            dynamic_cast<HypreParMatrix&>(trueAl_[i]->GetBlock(0,0)),
            *(Cl_[i]), HypreSmoother::l1GS, 3);
        Maux_[i]->iterative_mode = true;
    }

    Vector x_f, y_f, x_c, y_c;

    x_f.SetDataAndSize(essnullspace_data+levelStartMultiplier[0],
                       levelStart[1] - levelStartMultiplier[0]);
    y_f.SetDataAndSize(t_data+levelStartMultiplier[0],
                       levelStart[1] - levelStartMultiplier[0]);
    x_f= ess_nullspace_p;
    mass_p.Mult(x_f,y_f);

    for (int ilevel(1); ilevel < nLevels_; ++ilevel)
    {
        x_f.SetDataAndSize(essnullspace_data+levelStartMultiplier[ilevel-1],
                           levelStart[ilevel] - levelStartMultiplier[ilevel-1]);
        y_f.SetDataAndSize(t_data+levelStartMultiplier[ilevel-1],
                           levelStart[ilevel] - levelStartMultiplier[ilevel-1]);
        x_c.SetDataAndSize(essnullspace_data+levelStartMultiplier[ilevel],
                           levelStart[ilevel+1] - levelStartMultiplier[ilevel]);
        y_c.SetDataAndSize(t_data+levelStartMultiplier[ilevel],
                           levelStart[ilevel+1] - levelStartMultiplier[ilevel]);

        Pi_[ilevel-1]->GetBlock(PRESSURE,PRESSURE).Mult(x_f, x_c);
        P_[ilevel-1]->GetBlock(PRESSURE,PRESSURE).MultTranspose(y_f, y_c);
    }
}

void MLDivFree::SetCycle(MLDivFree::CycleType c)
{
    my_cycle = c;

    switch (c)
    {
    case FULL_SPACE:
        height = width = sequence_[0]->GetNumberOfTrueDofs(hdivform_) +
            sequence_[0]->GetNumberOfTrueDofs(l2form_);
        break;
    case DIVFREE_SPACE:
        height = width = sequence_[0]->GetNumberOfTrueDofs(hdivform_);
        break;
    }
}

void MLDivFree::Mult(const Vector & xconst, Vector & y) const
{
    elag_assert(xconst.Size() == height);
    elag_assert(y.Size() == width);
    Vector x(xconst.GetData(), xconst.Size());

    if (!iterative_mode)
    {
        if (xconst.GetData() == y.GetData())
        {
            x.SetSize(xconst.Size());
            x = xconst;
        }
        y = 0.0;
    }

    elag_assert(y.CheckFinite() == 0);
    switch (my_cycle)
    {
    case FULL_SPACE:
        MGVcycle(x, y);
        break;
    case DIVFREE_SPACE:
        MGVcycleOnlyU(x, y);
        break;
    }
}

void MLDivFree::Mult(const MultiVector & xconst, MultiVector & y) const
{
    MultiVector x(xconst.GetData(), xconst.NumberOfVectors(), xconst.Size());

    if (!iterative_mode)
    {
        if (xconst.GetData() == y.GetData())
        {
            x.SetSizeAndNumberOfVectors(xconst.Size(), x.NumberOfVectors());
            x = xconst;
        }
        y = 0.0;
    }

    Vector xi, yi;

    for (int i(0); i < x.NumberOfVectors(); ++i)
    {
        x.GetVectorView(i, xi);
        y.GetVectorView(i, yi);
        switch (my_cycle)
        {
        case FULL_SPACE:
            MGVcycle(xi, yi);
            break;
        case DIVFREE_SPACE:
            MGVcycleOnlyU(xi, yi);
            break;
        }

    }

}


void MLDivFree::MGVcycle(const Vector & x, Vector & y) const
{
//      std::fill(trueSol_data, trueSol_data+aritmeticComplexity, 0.);
    elag_assert(x.CheckFinite() == 0);
    int i(0);
    Vector rhs, rhs_coarse, sol, sol_coarse;
    Vector correction(trueAl_[0]->Height());

    //Step 1, go fine to coarse
    {
        rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i],
                           levelTrueStart[i+1]-levelTrueStart[i]);
        trueAl_[0]->Mult(y, rhs);
        add(1., x, -1., rhs,rhs);
        for (; i < nLevels_-1; ++i)
        {
            sol.SetDataAndSize(trueSol_data+levelTrueStart[i],
                               levelTrueStart[i+1]-levelTrueStart[i]);
            sol = 0.0;
            rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i],
                               levelTrueStart[i+1]-levelTrueStart[i]);
            //subdomainSmoother will always use the value of sol
            subdomainSmoother(i, rhs, sol);
            //nullSpaceSmoother will always use the value of sol
            nullSpaceSmoother(i, rhs, sol);
            correction.SetSize(levelTrueStart[i+1]-levelTrueStart[i]);
            trueAl_[i]->Mult(sol, correction);
            add(1., rhs, -1., correction, rhs);
            rhs_coarse.SetDataAndSize(trueRhs_data+levelTrueStart[i+1],
                                      levelTrueStart[i+2]-levelTrueStart[i+1]);
            trueP_[i]->MultTranspose(rhs, rhs_coarse);
        }
    }

    //Step 2, solve coarse grid problem
    {
        sol.SetDataAndSize(trueSol_data+levelTrueStart[i],
                           levelTrueStart[i+1]-levelTrueStart[i]);
        sol = 0.0;
        rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i],
                           levelTrueStart[i+1]-levelTrueStart[i]);
        coarseSolver(rhs, sol);
    }

    //Step 3, go coarse to fine
    {
        for (i=nLevels_-2; i >= 0; --i)
        {
            correction.SetSize(levelTrueStart[i+1]-levelTrueStart[i]);
            correction = 0.0;
            sol.SetDataAndSize(trueSol_data+levelTrueStart[i],
                               levelTrueStart[i+1]-levelTrueStart[i]);
            sol_coarse.SetDataAndSize(trueSol_data+levelTrueStart[i+1],
                                      levelTrueStart[i+2]-levelTrueStart[i+1]);
            rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i],
                               levelTrueStart[i+1]-levelTrueStart[i]);
            trueP_[i]->Mult(sol_coarse, correction);
            //nullSpaceSmoother will always use the value of corr
            nullSpaceSmoother(i, rhs, correction);
            //subdomainSmoother will always use the value of corr
            subdomainSmoother(i, rhs, correction);
            sol.Add(1., correction);
        }
    }

    y.Add(1., sol);
}

void MLDivFree::MGVcycleOnlyU(const Vector & x, Vector & y) const
{
    int i(0);
    Vector rhs, rhs_coarse, sol, sol_coarse;
    Vector correction(trueAl_[0]->GetBlock(0,0).Height());

    //Step 1, go fine to coarse
    {
        rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i],
                           levelTrueStartMultiplier[i]-levelTrueStart[i]);
        trueAl_[0]->GetBlock(0,0).Mult(y, rhs);
        add(1., x, -1., rhs, rhs);
        for (; i < nLevels_-1; ++i)
        {
            sol.SetDataAndSize(trueSol_data+levelTrueStart[i],
                               levelTrueStartMultiplier[i]-levelTrueStart[i]);
            sol = 0.0;
            rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i],
                               levelTrueStartMultiplier[i]-levelTrueStart[i]);
            //nullSpaceSmoother will always use the value of sol
            nullSpaceSmoother(i, rhs, sol);
            correction.SetSize(levelTrueStartMultiplier[i]-levelTrueStart[i]);
            trueAl_[i]->GetBlock(0,0).Mult(sol, correction);
            add(1., rhs, -1., correction, rhs);
            rhs_coarse.SetDataAndSize(
                trueRhs_data+levelTrueStart[i+1],
                levelTrueStartMultiplier[i+1]-levelTrueStart[i+1]);
            trueP_[i]->GetBlock(0,0).MultTranspose(rhs, rhs_coarse);
        }
    }

    //Step 2, solve fine grid problem
    {
        sol.SetDataAndSize(trueSol_data+levelTrueStart[i],
                           levelTrueStart[i+1]-levelTrueStart[i]);
        sol = 0.0;
        rhs.SetDataAndSize(trueRhs_data+levelTrueStartMultiplier[i],
                           levelTrueStart[i+1]-levelTrueStartMultiplier[i]);
        rhs = 0.0;
        rhs.SetDataAndSize(
            trueRhs_data+levelTrueStart[i],
            levelTrueStart[i+1]-levelTrueStart[i]);
        coarseSolver(rhs, sol);
    }

    //Step 3, go coarse to fine
    {
        for (i=nLevels_-2; i >= 0; --i)
        {
            correction.SetSize(levelTrueStartMultiplier[i]-levelTrueStart[i]);
            correction = 0.0;
            sol.SetDataAndSize(
                trueSol_data+levelTrueStart[i],
                levelTrueStartMultiplier[i]-levelTrueStart[i]);
            sol_coarse.SetDataAndSize(
                trueSol_data+levelTrueStart[i+1],
                levelTrueStartMultiplier[i+1]-levelTrueStart[i+1]);
            rhs.SetDataAndSize(
                trueRhs_data+levelTrueStart[i],
                levelTrueStartMultiplier[i]-levelTrueStart[i]);
            trueP_[i]->GetBlock(0,0).Mult(sol_coarse, correction);
            //nullSpaceSmoother will always use the value of corr
            nullSpaceSmoother(i, rhs, correction);
            sol.Add(1., correction);
        }
    }

    y.Add(1., sol);

}

// subdomainSmoother will always use the value of sol
void MLDivFree::subdomainSmoother(
    int i, const Vector & trueRhs, Vector & trueSol) const
{
    elag_assert(trueRhs.CheckFinite() == 0);
    elag_assert(trueSol.CheckFinite() == 0);

    Array<int> true_offsets(3);
    true_offsets[0] = 0;
    true_offsets[1] = levelTrueStartMultiplier[i] - levelTrueStart[i];
    true_offsets[2] = levelTrueStart[i+1] - levelTrueStart[i];

    Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = levelStartMultiplier[i] - levelStart[i];
    offsets[2] = levelStart[i+1] - levelStart[i];

    BlockVector res(rhs_data, offsets), sol(sol_data, offsets);
    BlockVector tRes(trueRhs.GetData(), true_offsets);
    BlockVector tSol(trueSol.GetData(), true_offsets);

    SharingMap & mapl2 = sequence_[i]->GetDofHandler(l2form_)->GetDofTrueDof();
    SharingMap & maphdiv =
        sequence_[i]->GetDofHandler(hdivform_)->GetDofTrueDof();

    maphdiv.Distribute(tRes.GetBlock(0),res.GetBlock(0));
    mapl2.Distribute(tRes.GetBlock(1),res.GetBlock(1));
    maphdiv.Distribute(tSol.GetBlock(0),sol.GetBlock(0));
    mapl2.Distribute(tSol.GetBlock(1),sol.GetBlock(1));

    elag_assert(res.CheckFinite() == 0);
    elag_assert(sol.CheckFinite() == 0);

    Al_[i]->AddMult(sol, res, -1.);
    Vector essnullspace, t;

    essnullspace.SetDataAndSize(
        const_cast<double*>(essnullspace_data+levelStartMultiplier[i]),
        levelStart[i+1]-levelStartMultiplier[i]);
    t.SetDataAndSize(
        const_cast<double*>(t_data+levelStartMultiplier[i]),
        levelStart[i+1]-levelStartMultiplier[i]);

    elag_assert(essnullspace.CheckFinite() == 0);
    elag_assert(t.CheckFinite() == 0);

    int nAE = AE_dof[i]->NumRows();

#ifdef ParELAG_ENABLE_OPENMP
#pragma omp parallel default(shared)
#endif
    {
        Vector sol_loc, rhs_loc, essnullspace_loc, t_loc;
        Array<int> colMapper, loc_dof, loc_dof_p;

        colMapper.SetSize(Al_[i]->NumCols());
        colMapper = -1;
#ifdef ParELAG_ENABLE_OPENMP
#pragma omp for schedule(guided) nowait
#endif
        for (int iAE = 0; iAE < nAE; ++iAE)
        {
            int nlocdof   = getLocalInternalDofs(i, iAE, loc_dof);
            int nlocdof_p = getLocalDofs(PRESSURE, i, iAE, loc_dof_p);

            if (nlocdof == 0)
            {
                std::cout << "AE " << iAE << " has 0 internal dofs! \n";
                mfem_error("");
            }

            auto Aloc = ExtractRowAndColumns(
                Al_[i].get(),loc_dof,loc_dof,colMapper);
            essnullspace.GetSubVector(loc_dof_p, essnullspace_loc);

            if (!isRankDeficient(Aloc->GetBlock(0,1), essnullspace_loc))
            {
                rhs_loc.SetSize(nlocdof);
                sol_loc.SetSize(nlocdof);
                res.GetSubVector(loc_dof, rhs_loc.GetData());
                elag_assert(rhs_loc.CheckFinite() == 0);
                SymmetrizedUmfpack solver(*Aloc);
                solver.Mult(rhs_loc, sol_loc);

                if (sol_loc.CheckFinite() != 0)
                {
                    essnullspace_loc.Print(std::cout, essnullspace_loc.Size());
                    Aloc->PrintMatlab(std::cout);
                    elag_error(1);
                }
                sol.AddElementVector(loc_dof, sol_loc.GetData());
            }
            else
            {
                rhs_loc.SetSize(nlocdof+1);
                sol_loc.SetSize(nlocdof+1);
                res.GetSubVector(loc_dof, rhs_loc.GetData());
                rhs_loc(nlocdof) = 0.;

                t_loc.SetSize(nlocdof_p);
                t.GetSubVector(loc_dof_p, t_loc.GetData());

                auto T = createSparseMatrixRepresentationOfScalarProduct(
                    t_loc.GetData(), t_loc.Size());
                std::unique_ptr<SparseMatrix> Tt{Transpose(*T)};

                Array<int> a_offset(4);
                a_offset[0] = Aloc->RowOffsets()[0];
                a_offset[1] = Aloc->RowOffsets()[1];
                a_offset[2] = Aloc->RowOffsets()[2];
                a_offset[3] = Aloc->RowOffsets()[2]+1;
                BlockMatrix a_Aloc(a_offset);
                a_Aloc.SetBlock(0,0, const_cast<SparseMatrix *>(&(Aloc->GetBlock(0,0))));
                a_Aloc.SetBlock(1,0, const_cast<SparseMatrix *>(&(Aloc->GetBlock(1,0))));
                a_Aloc.SetBlock(0,1, const_cast<SparseMatrix *>(&(Aloc->GetBlock(0,1))));
                a_Aloc.SetBlock(2,1, T.get());
                a_Aloc.SetBlock(1,2,Tt.get());
                SymmetrizedUmfpack solver(a_Aloc);
                solver.Mult(rhs_loc, sol_loc);
                elag_assert(sol_loc.CheckFinite() == 0);
                sol.AddElementVector(loc_dof, sol_loc.GetData());

                SparseMatrix * T_rel = T.release();
                destroySparseMatrixRepresentationOfScalarProduct(T_rel);
            }
        }
    }

    maphdiv.IgnoreNonLocal(sol.GetBlock(0),tSol.GetBlock(0));
    mapl2.IgnoreNonLocal(sol.GetBlock(1),tSol.GetBlock(1));

    elag_assert(tSol.CheckFinite() == 0);
}


//nullSpaceSmoother will always use the value of sol
void MLDivFree::nullSpaceSmoother(int i, const Vector & rhs, Vector & sol) const
{
    elag_assert(rhs.CheckFinite() == 0);
    Vector rrhs, rsol;
    rrhs.SetDataAndSize(
        rhs.GetData(), levelTrueStartMultiplier[i]-levelTrueStart[i]);
    rsol.SetDataAndSize(
        sol.GetData(), levelTrueStartMultiplier[i]-levelTrueStart[i]);
    Maux_[i]->Mult(rrhs, rsol);

    elag_assert(sol.CheckFinite() == 0);
}

//coarse solver
void MLDivFree::coarseSolver(const Vector & rhs, Vector & sol) const
{
    int i = nLevels_-1;
    BlockMatrix * Ac = Al_[i].get();

    SharingMap & mapl2 = sequence_[i]->GetDofHandler(l2form_)->GetDofTrueDof();
    SharingMap & maphdiv = sequence_[i]->GetDofHandler(hdivform_)->GetDofTrueDof();

    Array<int> true_offsets(3);
    true_offsets[0] = 0;
    true_offsets[1] = levelTrueStartMultiplier[i] - levelTrueStart[i];
    true_offsets[2] = levelTrueStart[i+1] - levelTrueStart[i];

    std::unique_ptr<HypreParMatrix>
        PAc00 = Assemble(maphdiv, Ac->GetBlock(0,0), maphdiv),
        PAc10 = Assemble(mapl2  , Ac->GetBlock(1,0), maphdiv),
        PAc01 = Assemble(maphdiv, Ac->GetBlock(0,1), mapl2);

    BlockOperator cOp(true_offsets);
    cOp.owns_blocks = 0;
    cOp.SetBlock(0,0, PAc00.get());
    cOp.SetBlock(1,0, PAc10.get());
    cOp.SetBlock(0,1, PAc01.get());

    Vector d(PAc00->Height());
    PAc00->GetDiag(d);

    auto invDBt = Assemble(maphdiv, Ac->GetBlock(0,1), mapl2);
    invDBt->InvScaleRows(d);
    auto S = ToUnique(ParMult(PAc10.get(),invDBt.get()));

    BlockDiagonalPreconditioner cPr(true_offsets);
    cPr.owns_blocks = 1;
    cPr.SetDiagonalBlock(0, new HypreSmoother(*PAc00));
    cPr.SetDiagonalBlock(1, new HypreExtension::HypreBoomerAMG(*S));

    MINRESSolver sc(maphdiv.GetComm());
    sc.SetMaxIter(1000);
    sc.SetPrintLevel(0);
    sc.SetAbsTol(std::numeric_limits<double>::epsilon());
    sc.SetRelTol(1e-8);
    sc.SetOperator(cOp);
    sc.SetPreconditioner(cPr);
    sol = 0.0;
    elag_assert(rhs.CheckFinite() == 0);
    sc.Mult(rhs,sol);
/*
  Vector essnullspace, t;
  essnullspace.SetDataAndSize(essnullspace_data+levelTrueStartMultiplier[i], levelTrueStart[i+1]-levelTrueStartMultiplier[i]);
  t.SetDataAndSize(t_data+levelTrueStartMultiplier[i], levelTrueStart[i+1]-levelTrueStartMultiplier[i]);
*/

}

void MLDivFree::computeSharedDof(int ilevel, Array<int> & is_shared)
{
    elag_assert(AE_dof[ilevel]->NumCols() == is_shared.Size());

    is_shared = 0;

    int nAE = AE_dof[ilevel]->NumRows();

    Array<int> cols;
    Vector svec;

    const int hdivform = sequence_[ilevel]->GetNumberOfForms() - 2;
    SharingMap &map = sequence_[ilevel]->GetDofHandler(hdivform)->GetDofTrueDof();

    const Array<int> & proc_shared_dof = map.SharedEntitiesId();

    for (const int * it = proc_shared_dof.GetData();
         it != proc_shared_dof.GetData()+proc_shared_dof.Size();
         ++it)
    {
        ++is_shared[*it];
    }

    for (int iAE(0); iAE < nAE; ++iAE)
    {
        AE_dof[ilevel]->GetRow(iAE, cols, svec);
        for (int * it = cols.GetData(),
                 * end = cols.GetData()+cols.Size();
             it != end; ++it)
        {
            ++is_shared[*it];
        }
    }

    for (int * it = is_shared.GetData(),
             * end = is_shared.GetData()+is_shared.Size();
         it != end; ++it)
    {
        (*it) = (*it > 1);
    }
}

int MLDivFree::getLocalInternalDofs(int ilevel, int iAE, Array<int> & loc_dof) const
{
    Vector val;
    Array<int> dofs;
    AE_dof[ilevel]->GetRow(iAE,dofs,val);
    loc_dof.SetSize(dofs.Size());
    int * idof = loc_dof.GetData();
    int nlocdof = 0;

    const Array<int> is_shared(
        const_cast<int *>(dof_is_shared_among_AE_data+levelStart[ilevel]),
        levelStart[ilevel+1] - levelStart[ilevel]);

    for (int *it = dofs.GetData(), * end = dofs.GetData()+dofs.Size(); it != end; ++it)
    {
        if (!is_shared[*it])
        {
            *(idof++) = *it;
            ++nlocdof;
        }
    }
    loc_dof.SetSize(nlocdof);
    return nlocdof;
}

int MLDivFree::getLocalInternalDofs(
    int comp, int ilevel, int iAE, Array<int> & loc_dof) const
{
    int * i_AE_dof = AE_dof[ilevel]->GetBlock(0, comp).GetI();
    int * j_AE_dof = AE_dof[ilevel]->GetBlock(0, comp).GetJ();
    int * comp_offset = AE_dof[ilevel]->ColOffsets().GetData();

    int * start = j_AE_dof + i_AE_dof[iAE];
    int * end   = j_AE_dof + i_AE_dof[iAE+1];

    loc_dof.SetSize(i_AE_dof[iAE+1] - i_AE_dof[iAE]);
    int * idof = loc_dof.GetData();
    int nlocdof = 0;

    const Array<int> is_shared(
        const_cast<int *>(dof_is_shared_among_AE_data+levelStart[ilevel]+comp_offset[comp]),
        comp_offset[comp+1]-comp_offset[comp]);

    for (int * it = start; it != end; ++it)
    {
        if (!is_shared[*it])
        {
            *(idof++) = *it;
            ++nlocdof;
        }
    }
    loc_dof.SetSize(nlocdof);
    return nlocdof;
}

int MLDivFree::getLocalDofs(int comp, int ilevel, int iAE, Array<int> & loc_dof) const
{
    int * i_AE_dof = AE_dof[ilevel]->GetBlock(0, comp).GetI();
    int * j_AE_dof = AE_dof[ilevel]->GetBlock(0, comp).GetJ();

    int * start = j_AE_dof + i_AE_dof[iAE];
    int len     = i_AE_dof[iAE+1] - i_AE_dof[iAE];

    loc_dof.MakeRef(start, len);

    return len;
}

bool MLDivFree::isRankDeficient(const SparseMatrix & Bt, const Vector & x) const
{
    // Here we check the l_inf norm of y = Bt*x.
    // In practice we compute the mat-vec row by row and we stop if
    // abs(y(i)) > numerical_zero
    const int * i_Bt = Bt.GetI();
    const int * j_Bt = Bt.GetJ();
    const double * a_Bt = Bt.GetData();
    int nrows = Bt.Size();
    int ncols = Bt.Width();

    PARELAG_TEST_FOR_EXCEPTION(
        ncols != x.Size(),
        std::runtime_error,
        "MLDivFree::isRankDeficient: ncols != x.Size()");

    double val(0);

    for (int irow = 0; irow < nrows; ++irow)
    {
        val = 0;
        for (int j = i_Bt[irow]; j < i_Bt[irow+1]; ++j)
            val += a_Bt[j] * x(j_Bt[j]);
        if (fabs(val) > numerical_zero)
            return false;
    }

    return true;
}

unique_ptr<BlockMatrix> MLDivFree::PtAP(const BlockMatrix & A, const BlockMatrix & P) const
{
    const int numBlocks = A.NumRowBlocks();

    unique_ptr<BlockMatrix> Pt{Transpose(P)};

    auto out = make_unique<BlockMatrix>(P.ColOffsets());
    out->owns_blocks = true;

    for (int bl_row = 0; bl_row < numBlocks; ++bl_row)
        for (int bl_col = 0; bl_col < numBlocks; ++bl_col)
            if (!A.IsZeroBlock(bl_row,bl_col))
            {
                const SparseMatrix & Pt_ii = Pt->GetBlock(bl_row,bl_row);
                const SparseMatrix & P_jj = P.GetBlock(bl_col,bl_col);

                unique_ptr<SparseMatrix> PtA{
                    mfem::Mult(Pt_ii,A.GetBlock(bl_row,bl_col))};
                // Ownership of the final product is assumed by
                // the output BlockMatrix
                out->SetBlock(bl_row, bl_col, mfem::Mult(*PtA,P_jj));
            }
    out->EliminateZeroRows();
    return out;
}
}//namespace parelag
