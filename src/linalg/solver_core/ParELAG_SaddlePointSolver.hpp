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

#ifndef SADDLEPOINTSOLVER_HPP_
#define SADDLEPOINTSOLVER_HPP_

#include <mfem.hpp>

#include "linalg/dense/ParELAG_LDLCalculator.hpp"
#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "linalg/utilities/ParELAG_SubMatrixExtraction.hpp"

namespace parelag
{

class SaddlePointSolver
{
public:
    SaddlePointSolver() = default;
    virtual ~SaddlePointSolver() = default;

    /// should maybe bounds check in case the real object is RidgePeak
    int GetLocalOffsets(int i) const {return local_offsets_[i];}

    /// you could imagine a generic block(i,j) interface, but I don't
    /// think that's what we want.
    const mfem::SparseMatrix& GetMloc() const {return *Mloc_;}
    const mfem::SparseMatrix& GetBloc() const {return *Bloc_;}

    int GetUInternalStart() {return uInternalStart_;}
    int GetUInternalEnd() {return uInternalEnd_;}
    int GetPInternalStart() {return pInternalStart_;}
    int GetPInternalEnd() {return pInternalEnd_;}

protected:
    mutable parelag::LDLCalculator ldl_solver_; // not very const-friendly
    parelag::MultiVector rhs_, sol_;
    int local_offsets_[4]; // 4? why 4?
    std::unique_ptr<mfem::SparseMatrix> Mloc_, Bloc_;
    std::unique_ptr<mfem::DenseMatrix> Aloc_;
    int uInternalStart_, uInternalEnd_, pInternalStart_, pInternalEnd_;
};

/**
   Takes (large) matrices M_d, B_d, W_d and extracts submatrices from
   them and builds a (dense, direct) solver for the sub-saddle point problem.

   This is an effort to encapsulate and modularize some data/variables
   associated with submatrix extraction and LDLSolver in the
   hEntityExtension routines. It is a work in progress and is quite ugly, but
   arguably better than we had before.

   Now it only works for hFacetExtension, the structure in hRidgeExtension
   and hPeakExtension is just a bit different (2 by 2 block instead of
   3 by 3, slightly different block-nonzero structure).
*/
class FacetSaddlePoint : public SaddlePointSolver
{
public:
    FacetSaddlePoint(
        parelag::AgglomeratedTopology::Entity codim_dom,
        const parelag::DofAgglomeration& DofAgg,
        const parelag::DofAgglomeration& DofAggPlus,
        mfem::SparseMatrix * P, const mfem::SparseMatrix& AE_PVdof,
        const mfem::SparseMatrix& M_d, const mfem::SparseMatrix& B_d,
        const mfem::SparseMatrix& W_d, int iAE);

    /**
       Call this before calling Solve, and then set up your right hand side
       in the returned rhs_view MultiVectors.
    */
    void SetUpRhs(int num_vectors, parelag::MultiVector& rhs_view_u,
                  parelag::MultiVector& rhs_view_p,
                  parelag::MultiVector& rhs_view_l);

    /**
       Using the RHS from SetUpRhs and the matrix from the contructor, return
       solution in three block pieces in these MultiVector references.
    */
    void Solve(parelag::MultiVector& sol_view_u,
               parelag::MultiVector& sol_view_p,
               parelag::MultiVector& sol_view_l);

    const mfem::SparseMatrix& GetWloc() const {return *Wloc_;}

private:
    std::unique_ptr<mfem::SparseMatrix> Wloc_;
};

/**
   Does for hRidgeExtension and hPeakExtension what FacetSaddlePoint does for
   hFacetExtension
*/
class RidgePeakSaddlePoint : public SaddlePointSolver
{
public:
    RidgePeakSaddlePoint(
        parelag::AgglomeratedTopology::Entity codim_dom,
        const parelag::DofAgglomeration& DofAgg,
        const parelag::DofAgglomeration& DofAggPlus,
        const mfem::SparseMatrix& M_d, const mfem::SparseMatrix& B_d,
        const mfem::SparseMatrix& minusC_d, int iAE);

    /**
       Call this before calling Solve, and then set up your right hand side
       in the returned rhs_view MultiVectors.
    */
    void SetUpRhs(int num_vectors, parelag::MultiVector& rhs_view_u,
                  parelag::MultiVector& rhs_view_p);

    /**
       Using the RHS from SetUpRhs and the matrix from the contructor, return
       solution in three block pieces in these parelag::MultiVector references.
    */
    void Solve(parelag::MultiVector& sol_view_u,
               parelag::MultiVector& sol_view_p);

    const mfem::SparseMatrix& GetmCloc() const {return *mCloc_;}

    /// for debugging
    void DumpDetails(std::string tag);

private:
    std::unique_ptr<mfem::SparseMatrix> mCloc_;
};

} // namespace parelag
#endif
