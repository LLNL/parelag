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

#include <memory>
#include <numeric>
#include <vector>

#include "LocalSpectralTargets.hpp"

#include "linalg/dense/ParELAG_Eigensolver.hpp"
#include "linalg/dense/ParELAG_MultiVector.hpp"
#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "linalg/utilities/ParELAG_SubMatrixExtraction.hpp"
#include "linalg/solver_core/ParELAG_SymmetrizedUmfpack.hpp"
#include "utilities/elagError.hpp"
#include "utilities/MemoryUtils.hpp"

using namespace mfem;
using std::unique_ptr;

namespace parelag
{

std::vector<unique_ptr<MultiVector>> ComputeLocalSpectralTargetsFromEntity(
    AgglomeratedTopology::Entity entity_type,
    SparseMatrix& M_e,DofAgglomeration *dofAgg,
    double rel_tol,int max_evects )
{
    auto M_AEntity =
        AssembleAgglomerateMatrix(entity_type, M_e, dofAgg, dofAgg);
    return ComputeLocalSpectralTargetsFromAEntity(
        entity_type,*M_AEntity,dofAgg,rel_tol,max_evects);
}

std::vector<unique_ptr<MultiVector>>
ComputeLocalSpectralTargetsFromAEntity(
    AgglomeratedTopology::Entity entity_type,
    SparseMatrix& M_AEntity,DofAgglomeration *dofAgg,
    double rel_tol,int max_evects)
{
    const SparseMatrix& AEntity_dof = dofAgg->GetAEntityDof(entity_type);
    // NOTE: We count on the numberings in \b M_AEntity to correspond
    //       to what you would get from \b AEntity_dof. That is,
    //       "repeated dofs" in terms of agglomerated entities and
    //       local dofs in these entities are numbered according to
    //       the indices of the J array of the CSR representation of
    //       \b AEntity_dof.
    elag_assert(M_AEntity.Height() == M_AEntity.Width());
    elag_assert(M_AEntity.Height() == AEntity_dof.NumNonZeroElems());
    const int nAEs = AEntity_dof.Height();
    const int * const i_AEntity_dof = AEntity_dof.GetI();

    std::vector<double> evals;
    Vector D_vec;
    DenseMatrix evects;
    const double max_eval = 1.;
    std::vector<unique_ptr<MultiVector>> localtargets(nAEs);
    SymEigensolver eigs;
    for (int i=0; i < nAEs; ++i)
    {
        const int start    = i_AEntity_dof[i];
        const int end      = i_AEntity_dof[i+1];
        const int loc_size = end-start;
        auto Mloc = ExtractSubMatrix(M_AEntity, start, end, start, end);
        elag_assert(Mloc->Height() == loc_size);
        elag_assert(Mloc->Width() == loc_size);
        Weightedl1Smoother(*Mloc, D_vec);
        const int ret = eigs.ComputeDiagonalSmallerMagnitude(
            *Mloc, D_vec, evals, evects, rel_tol, max_eval, max_evects);
        PARELAG_ASSERT(!ret);
        elag_assert(evects.Height() == loc_size);
        const int nevects = evects.Width();
        elag_assert(max_evects < 1 || nevects <= max_evects);
        localtargets[i] = make_unique<MultiVector>(
            evects.Data(),nevects,loc_size);
        localtargets[i]->MakeDataOwner();
        evects.ClearExternalData();
    }
    return localtargets;
}

void ComputeLocalHdivL2SpectralTargetsFromAEntity(
    SparseMatrix& M_AEntity,SparseMatrix& D_AEntity,
    SparseMatrix& W_AEntity,SparseMatrix& Q_AEntity,
    DofAgglomeration *HdivdofAgg,DofAgglomeration *L2dofAgg,
    TopologyTable & AE_AF, double rel_tol, int max_evects,
    std::vector<unique_ptr<MultiVector>>& localHdivTracetargets,
    std::vector<unique_ptr<MultiVector>>& localL2targets)
{
    // NOTE: We count on the numberings in \b M_AEntity to correspond
    //       to what you would get from \b AEntity_dof. That is,
    //       "repeated dofs" in terms of agglomerated entities and
    //       local dofs in these entities are numbered according to
    //       the indices of the J array of the CSR representation of
    //       \b AEntity_dof.
    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    constexpr auto at_facet = AgglomeratedTopology::FACET;

    auto AF_AE = AE_AF.Transpose();

    const int nAEs = AF_AE->Width(); // Number of coarse elements
    const int nAFs = AF_AE->Size(); // Number of coarse facets

    int u_start, u_end, ub_start, ub_end, p_start, p_end, uAF_start, uAF_end;
    std::vector<double> evals;
    DenseMatrix evects, MinvBt, MinvCt, BMinvBt, CMinvBt,
        BMinvCt, CMinvCt, BlocT, ClocT, Wloc, Qloc, S_mat, RHS, u_mat;
    std::vector<unique_ptr<DenseMatrix>> AE_u(nAEs);

    std::vector<int> hdivDofMarker(
        HdivdofAgg->GetAEntityDof(at_elem).Width(), -1);

    SymEigensolver eigs;
    for (int iAE=0; iAE < nAEs; ++iAE)
    {
        HdivdofAgg->GetAgglomerateDofRange(
            at_elem,iAE,u_start,u_end );
        HdivdofAgg->GetAgglomerateBdrDofRange(
            at_elem,iAE,ub_start,ub_end );
        L2dofAgg->GetAgglomerateInternalDofRange(
            at_elem,iAE,p_start,p_end );

        const int nL2LocalDofs = p_end - p_start;
        const int nBdrHdivLocalDofs = ub_end - ub_start;
        const int loc_size = nL2LocalDofs + nBdrHdivLocalDofs;

        auto Mloc = ExtractSubMatrix(M_AEntity,u_start,u_end,u_start,u_end);
        auto Bloc = ExtractSubMatrix(D_AEntity,p_start,p_end,u_start,u_end);
        auto Cloc = ExtractSubMatrix(Q_AEntity,ub_start,ub_end,u_start,u_end);
        auto Wloc_sp = ExtractSubMatrix(W_AEntity,p_start,p_end,p_start,p_end);
        auto Qloc_sp = ExtractSubMatrix(
            Q_AEntity,ub_start,ub_end,ub_start,ub_end);

        Full(*Bloc, BlocT);
        Full(*Cloc, ClocT);

        BlocT.Transpose();
        ClocT.Transpose();

        SymmetrizedUmfpack AEmass_solver(*Mloc);
        MinvBt.SetSize(Mloc->Size(), BlocT.Width());
        AEmass_solver.Mult(BlocT, MinvBt);
        MinvCt.SetSize(Mloc->Size(), ClocT.Width());
        AEmass_solver.Mult(ClocT, MinvCt);

        BMinvBt.SetSize(Bloc->Size(), MinvBt.Width());
        Mult(*Bloc, MinvBt, BMinvBt);
        CMinvBt.SetSize(Cloc->Size(), MinvBt.Width());
        Mult(*Cloc, MinvBt, CMinvBt);
        BMinvCt.SetSize(Bloc->Size(), MinvCt.Width());
        Mult(*Bloc, MinvCt, BMinvCt);
        CMinvCt.SetSize(Cloc->Size(), MinvCt.Width());
        Mult(*Cloc, MinvCt, CMinvCt);
        Block2by2(BMinvBt, BMinvCt, CMinvBt, CMinvCt, S_mat);
        PARELAG_ASSERT(S_mat.Height() == loc_size);
        PARELAG_ASSERT(S_mat.Width() == loc_size);

        Full(*Wloc_sp, Wloc);
        Full(*Qloc_sp, Qloc);

        BlockDiag2by2(Wloc, Qloc, RHS);

        const int ret = eigs.ComputeGeneralizedSmallerMagnitude(
            S_mat, RHS, evals, evects, rel_tol, max_evects );

        PARELAG_ASSERT(!ret);
        elag_assert(evects.Height() == loc_size);
        const int nevects = evects.Width();
        PARELAG_ASSERT(max_evects < 1 || nevects <= max_evects);

        PARELAG_TEST_FOR_EXCEPTION(
            fabs(evals[0]) > 1e-8,
            std::runtime_error,
            "ComputeLocalHdivL2SpectralTargetsFromAEntity(): "
            "AE " << iAE << " has nonzero smallest eigenvalue"
            " = " << evals[0] << "!");

        if (evects(0,0) < 0)
        {
            Vector ones;
            evects.GetColumnReference(0,ones);
            ones *= -1.;
        }

        auto p_mat = make_unique<DenseMatrix>();
        auto mu = make_unique<DenseMatrix>();

        SplitMatrixHorizontally(evects, nL2LocalDofs, *p_mat, *mu);
        PARELAG_ASSERT(p_mat->Height() == nL2LocalDofs);
        PARELAG_ASSERT(mu->Height() == nBdrHdivLocalDofs);
        PARELAG_ASSERT(mu->Width() == evects.Width());
        PARELAG_ASSERT(p_mat->Width() == evects.Width());

        AE_u[iAE] = std::move(mu);

        localL2targets[iAE] = make_unique<MultiVector>(
            p_mat->Data(), p_mat->Width(), nL2LocalDofs );
        // evects.ClearExternalData();
        p_mat->ClearExternalData();
        localL2targets[iAE]->MakeDataOwner();
    }

    for (int iAF = 0; iAF < nAFs; ++iAF)
    {
        // Extract the dofs i.d. for the face
        HdivdofAgg->GetAgglomerateDofRange(at_facet, iAF, uAF_start, uAF_end);
        const int nHdivAFDofs = uAF_end - uAF_start;
        const int * const j_AE_Hdivdof =
            HdivdofAgg->GetAEntityDof(at_elem).GetJ();
        const int * const j_AF_Hdivdof =
            HdivdofAgg->GetAEntityDof(at_facet).GetJ();
        PARELAG_ASSERT(1 <= nHdivAFDofs);

        unique_ptr<DenseMatrix> uAF;

        if (nHdivAFDofs > 1)
        {
            // Restrict local velocities in AE_u to the coarse face.
            const int nAE = AF_AE->RowSize(iAF);
            PARELAG_ASSERT(1 <= nAE && nAE <= 2);
            const int* AEs = AF_AE->GetRowColumns(iAF);

            const int total_vects =
                std::accumulate(
                    AEs,AEs+nAE,0,
                    [&AE_u](const int& a, const int& b)
                    {return a + AE_u[b]->Width();});

            uAF = make_unique<DenseMatrix>(total_vects, nHdivAFDofs);

            int start = 0;
            for (int i=0; i < nAE; ++i)
            {
                const int& AE = AEs[i];
                HdivdofAgg->GetAgglomerateBdrDofRange(
                    at_elem, AE, ub_start, ub_end);
                const int nBdrHdivLocalDofs = ub_end - ub_start;

                for (int j=0; j < nBdrHdivLocalDofs; ++j)
                    hdivDofMarker[*(j_AE_Hdivdof+ub_start+j)] = j;

                u_mat.Transpose(*(AE_u[AE]));
                PARELAG_ASSERT(u_mat.Width() == nBdrHdivLocalDofs);
                PARELAG_TEST_FOR_EXCEPTION(
                    u_mat.CheckFinite(),
                    std::runtime_error,
                    "ComputeLocalHdivL2SpectralTargetsFromAEntity(): "
                    "coarsenHdivTraces: u_mat.CheckFinite()");

                const int vects = u_mat.Height();
                for (int j=0; j < nHdivAFDofs; ++j)
                {
                    const int& id_in_AE =
                        hdivDofMarker[*(j_AF_Hdivdof+uAF_start+j)];
                    PARELAG_ASSERT(0 <= id_in_AE);
                    PARELAG_ASSERT(u_mat.Width() > id_in_AE);
                    PARELAG_ASSERT(start + vects <= total_vects);
                    std::copy(u_mat.Data() + id_in_AE*vects,
                              u_mat.Data() + (id_in_AE + 1)*vects,
                              uAF->Data() + j*total_vects + start);
                }
                start += vects;

                for (int j=0; j < nBdrHdivLocalDofs; ++j)
                    hdivDofMarker[*(j_AE_Hdivdof+ub_start+j)] = -1;
            }
            PARELAG_ASSERT(start == total_vects);

            uAF->Transpose();//uAF->Print(std::cout,6);

            PARELAG_TEST_FOR_EXCEPTION(
                uAF->CheckFinite(),
                std::runtime_error,
                "ComputeLocalHdivL2SpectralTargetsFromAEntity(): "
                "coarsenHdivTraces: uAF->CheckFinite().");
        }
        else
        {
            uAF = make_unique<DenseMatrix>(1,1);
            *uAF = 1.;
        }

        localHdivTracetargets[iAF] = make_unique<MultiVector>(
            uAF->Data(), uAF->Width(), uAF->Height() );
        uAF->ClearExternalData();
        localHdivTracetargets[iAF]->MakeDataOwner();
    }
}
}//namespace parelag
