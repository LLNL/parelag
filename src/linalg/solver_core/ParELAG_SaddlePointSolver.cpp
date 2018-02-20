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


#include "linalg/solver_core/ParELAG_SaddlePointSolver.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

void RidgePeakSaddlePoint::DumpDetails(std::string tag)
{
    std::string infofilename = tag + ".info";
    std::ofstream matinfo(infofilename.c_str());
    for (int i=0; i<4; ++i)
        matinfo << "local_offsets[" << i << "] = " << local_offsets_[i]
                << std::endl;
    matinfo << "number of vectors in multivec: " << rhs_.NumberOfVectors()
            << std::endl;

    std::string mfilename = tag + "M.mat";
    std::ofstream mfd(mfilename.c_str());
    Mloc_->Print(mfd);
    std::string bfilename = tag + "B.mat";
    std::ofstream bfd(bfilename.c_str());
    Bloc_->Print(bfd);
    std::string cfilename = tag + "C.mat";
    std::ofstream cfd(cfilename.c_str());
    mCloc_->Print(cfd);
    std::string rhsfilename = tag + "rhs.vec";
    std::ofstream rhsfd(rhsfilename.c_str());
    rhs_.Print(rhsfd);
    std::string solfilename = tag + "sol.vec";
    std::ofstream solfd(solfilename.c_str());
    sol_.Print(solfd);
}

void FacetSaddlePoint::SetUpRhs(
    int num_vectors, MultiVector& rhs_view_u, MultiVector& rhs_view_p,
    MultiVector& rhs_view_l)
{
    rhs_.SetSizeAndNumberOfVectors(local_offsets_[3], num_vectors);
    sol_.SetSizeAndNumberOfVectors(local_offsets_[3], num_vectors);
    rhs_ = 0.0;
    rhs_.GetRangeView(local_offsets_[0], local_offsets_[1], rhs_view_u);
    rhs_.GetRangeView(local_offsets_[1], local_offsets_[2], rhs_view_p);
    rhs_.GetRangeView(local_offsets_[2], local_offsets_[3], rhs_view_l);
}

void FacetSaddlePoint::Solve(
    MultiVector& sol_view_u, MultiVector& sol_view_p, MultiVector& sol_view_l)
{
    ldl_solver_.Mult(rhs_, sol_);
    sol_.GetRangeView(local_offsets_[0], local_offsets_[1], sol_view_u);
    sol_.GetRangeView(local_offsets_[1], local_offsets_[2], sol_view_p);
    sol_.GetRangeView(local_offsets_[2], local_offsets_[3], sol_view_l);
}

FacetSaddlePoint::FacetSaddlePoint(
    AgglomeratedTopology::Entity codim_dom,
    const DofAgglomeration& DofAgg,
    const DofAgglomeration& DofAggPlus,
    SparseMatrix * P, const SparseMatrix& AE_PVdof, const SparseMatrix& M_d,
    const SparseMatrix& B_d, const SparseMatrix& W_d, int iAE)
{
    // (1) Compute the offsets
    DofAgg.GetAgglomerateInternalDofRange(codim_dom, iAE, uInternalStart_, uInternalEnd_);
    DofAggPlus.GetAgglomerateInternalDofRange(codim_dom, iAE, pInternalStart_, pInternalEnd_);
    local_offsets_[0] = 0;
    local_offsets_[1] = uInternalEnd_ - uInternalStart_;
    local_offsets_[2] = local_offsets_[1] + pInternalEnd_ - pInternalStart_;
    local_offsets_[3] = local_offsets_[2] + 1;

    // (2) Get local matrices and allocate solver
    Mloc_ = ExtractSubMatrix(M_d, uInternalStart_, uInternalEnd_,
                             uInternalStart_, uInternalEnd_);
    Bloc_ = ExtractSubMatrix(B_d, pInternalStart_, pInternalEnd_,
                             uInternalStart_, uInternalEnd_);
    Wloc_ = ExtractSubMatrix(W_d, pInternalStart_, pInternalEnd_,
                             pInternalStart_, pInternalEnd_);

    Array<int> finePDof;
    DofAggPlus.GetViewAgglomerateInternalDofGlobalNumering(
        codim_dom, iAE, finePDof);

    Array<int> coarsePDof;
    DenseMatrix loc_pv_dof(local_offsets_[2]-local_offsets_[1], 1);
    const int * const i_AE_PVdof = AE_PVdof.GetI();
    const int * const j_AE_PVdof = AE_PVdof.GetJ();
    coarsePDof.MakeRef(const_cast<int *>(j_AE_PVdof+i_AE_PVdof[iAE]), 1);
    P->GetSubMatrix(finePDof, coarsePDof, loc_pv_dof);

    Vector pvloc(loc_pv_dof.Data(), loc_pv_dof.Height());
    Vector tloc(pvloc.Size());
    Wloc_->Mult(pvloc,tloc);
    auto Tloc = createSparseMatrixRepresentationOfScalarProduct(
        tloc.GetData(), tloc.Size());

    Aloc_ = make_unique<DenseMatrix>(local_offsets_[3], local_offsets_[3]);
    *Aloc_ = 0.0;
    Full(*Mloc_, *Aloc_, local_offsets_[0], local_offsets_[0]);
    Full(*Bloc_, *Aloc_, local_offsets_[1], local_offsets_[0]);
    Full(*Tloc, *Aloc_, local_offsets_[2], local_offsets_[1]);

    int test = ldl_solver_.Compute(*Aloc_); // build factorization (in place)
#ifdef ELAG_DEBUG
    if (test != 0)
        // LAPACK documentation suggests this happens only when matrix is
        // "exactly singular"
        std::cerr << "LDL factorization in saddle point solve failed!"
                  << std::endl << "  one reason for this error could be a bad"
                  << " topology, such as a torus-shaped agglomerate."
                  << std::endl;
#endif
    PARELAG_ASSERT(test == 0);

    SparseMatrix * Tloc_rel = Tloc.release();
    destroySparseMatrixRepresentationOfScalarProduct(Tloc_rel);
}

void RidgePeakSaddlePoint::SetUpRhs(
    int num_vectors, MultiVector& rhs_view_u, MultiVector& rhs_view_p)
{
    rhs_.SetSizeAndNumberOfVectors(local_offsets_[2], num_vectors);
    sol_.SetSizeAndNumberOfVectors(local_offsets_[2], num_vectors);
    rhs_ = 0.0;
    rhs_.GetRangeView(local_offsets_[0], local_offsets_[1], rhs_view_u);
    rhs_.GetRangeView(local_offsets_[1], local_offsets_[2], rhs_view_p);
}

void RidgePeakSaddlePoint::Solve(
    MultiVector& sol_view_u, MultiVector& sol_view_p)
{
    ldl_solver_.Mult(rhs_, sol_);
    sol_.GetRangeView(local_offsets_[0], local_offsets_[1], sol_view_u);
    sol_.GetRangeView(local_offsets_[1], local_offsets_[2], sol_view_p);
}

RidgePeakSaddlePoint::RidgePeakSaddlePoint(
    AgglomeratedTopology::Entity codim_dom,
    const DofAgglomeration& DofAgg,
    const DofAgglomeration& DofAggPlus,
    const SparseMatrix& M_d, const SparseMatrix& B_d,
    const SparseMatrix& minusC_d, int iAE)
{
    DofAgg.GetAgglomerateInternalDofRange(
        codim_dom, iAE, uInternalStart_, uInternalEnd_);
    DofAggPlus.GetAgglomerateInternalDofRange(
        codim_dom, iAE, pInternalStart_, pInternalEnd_);

    local_offsets_[0] = 0;
    local_offsets_[1] = uInternalEnd_ - uInternalStart_;
    local_offsets_[2] = local_offsets_[1] + pInternalEnd_ - pInternalStart_;
    local_offsets_[3] = -1;

    // get local matrices for actual operator
    Mloc_ = ExtractSubMatrix(M_d,
                             uInternalStart_, uInternalEnd_,
                             uInternalStart_, uInternalEnd_);
    Bloc_ = ExtractSubMatrix(B_d,
                             pInternalStart_, pInternalEnd_,
                             uInternalStart_, uInternalEnd_);
    mCloc_ = ExtractSubMatrix(minusC_d,
                              pInternalStart_, pInternalEnd_,
                              pInternalStart_, pInternalEnd_);

    Aloc_ = make_unique<DenseMatrix>(local_offsets_[2], local_offsets_[2]);
    *Aloc_ = 0.0;
    Full(*Mloc_, *Aloc_, local_offsets_[0], local_offsets_[0]);
    Full(*Bloc_, *Aloc_, local_offsets_[1], local_offsets_[0]);
    Full(*mCloc_, *Aloc_, local_offsets_[1], local_offsets_[1]);

    if (local_offsets_[1] > 0)
    {
        int test = ldl_solver_.Compute(*Aloc_); // build factorization
        PARELAG_ASSERT(test == 0);
    }
}

}
