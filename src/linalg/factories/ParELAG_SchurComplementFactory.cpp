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


#include "linalg/factories/ParELAG_SchurComplementFactory.hpp"

#include <algorithm>
#include <numeric>

#include "amge/DeRhamSequence.hpp"
#include "hypreExtension/hypreExtension.hpp"
#include "utilities/elagError.hpp"

namespace parelag
{

SchurComplementFactory::SchurComplementFactory(
    std::string schur_complement_type,
    double scaling)
    : Type_(std::move(schur_complement_type)),
      Alpha_(scaling)
{
    std::transform(Type_.begin(), Type_.end(), Type_.begin(), ::toupper);
}

std::unique_ptr<mfem::Operator>
SchurComplementFactory::BuildOperator(
    MfemBlockOperator& op,SolverState& state) const
{
    PARELAG_ASSERT(op.GetNumBlockRows() == 2);
    PARELAG_ASSERT(op.GetNumBlockCols() == 2);

    if (Type_ == "MASS")
    {
        // Assume the state has just the important form up front
        auto form = state.GetForms().front();
        auto& sequence = state.GetDeRhamSequence();

        return sequence.ComputeTrueM(form);
    }
    else if (Type_ == "DIAGONAL" || Type_ == "ABSROWSUM")
    {
        // Computes A11 - Alpha_*A10*diag(A00)*A10;
        auto A00 = (op.IsZeroBlock(0,0) ? nullptr :
                    dynamic_cast<mfem::HypreParMatrix*>(&op.GetBlock(0,0)));
        auto A01 = (op.IsZeroBlock(0,1) ? nullptr :
                    dynamic_cast<mfem::HypreParMatrix*>(&op.GetBlock(0,1)));
        auto A10 = (op.IsZeroBlock(1,0) ? nullptr :
                    dynamic_cast<mfem::HypreParMatrix*>(&op.GetBlock(1,0)));
        auto A11 = (op.IsZeroBlock(1,1) ? nullptr :
                    dynamic_cast<mfem::HypreParMatrix*>(&op.GetBlock(1,1)));

        // I need three of the blocks to be valid.
        PARELAG_ASSERT(A00 && A01 && A10);

        mfem::Vector diag;
        if (Type_ == "DIAGONAL")
            A00->GetDiag(diag);
        else if (Type_ == "ABSROWSUM")
        {
            const auto num_rows = A00->Height();
            diag.SetSize(num_rows);

            hypre_ParCSRMatrix* pA00 = *A00;

            auto diag_i = hypre_CSRMatrixI(pA00->diag),
                offd_i = hypre_CSRMatrixI(pA00->offd);
            auto diag_d = hypre_CSRMatrixData(pA00->diag),
                offd_d = hypre_CSRMatrixData(pA00->offd);

            // Compute the absolute row sums
            for (auto row = decltype(num_rows){0}; row < num_rows; ++row)
            {
                auto& entry = diag[row] = 0.;

                // Add the diag block contributions
                entry = std::accumulate(
                    diag_d+diag_i[row], diag_d+diag_i[row+1], entry,
                    [](const double& a, const double& b)
                    { return a+std::fabs(b); });

                // Add the offd block contributions
                entry = std::accumulate(
                    offd_d+offd_i[row], offd_d+offd_i[row+1], entry,
                    [](const double& a, const double& b)
                    { return a+std::fabs(b); });
            }
        }

        // FIXME!!!
        hypre_ParCSRMatrix* tmp_h;
        {
            hypre_ParCSRMatrix* a01_h = *A01;

            HYPRE_Int* row_starts = (HYPRE_Int*)malloc(sizeof(HYPRE_Int)*2);
            HYPRE_Int* col_starts = (HYPRE_Int*)malloc(sizeof(HYPRE_Int)*2);
            std::copy_n(a01_h->row_starts,2,row_starts);
            std::copy_n(a01_h->col_starts,2,col_starts);

            tmp_h = hypre_ParCSRMatrixCreate(
                a01_h->comm, a01_h->global_num_rows, a01_h->global_num_cols,
                row_starts, col_starts,
                a01_h->offd->num_cols, a01_h->diag->num_nonzeros,
                a01_h->offd->num_nonzeros);
            PARELAG_ASSERT(tmp_h);

            PARELAG_ASSERT_HYPRE_ERROR_FLAG(
                hypre_ParCSRMatrixInitialize(tmp_h));

            PARELAG_ASSERT_HYPRE_ERROR_FLAG(
                hypre_ParCSRMatrixCopy(a01_h,tmp_h,1));
        }

        // tmp = A01
        mfem::HypreParMatrix tmp(tmp_h);
        // I think this now takes ownership of tmp_h

        // tmp = A00^{-1}*A01
        tmp.InvScaleRows(diag);

        // product = A10*A00^{-1}*A01
        auto product = std::unique_ptr<mfem::HypreParMatrix>{
            mfem::ParMult(A10,&tmp)};

#if MFEM_HYPRE_VERSION <= 22200
        // Fix ownership of stuff
        {
            hypre_ParCSRMatrix* prod_h = *product;
            HYPRE_Int* prod_row_starts =
                (HYPRE_Int*) malloc(sizeof(HYPRE_Int)*2);
            HYPRE_Int* prod_col_starts =
                (HYPRE_Int*) malloc(sizeof(HYPRE_Int)*2);
            std::copy_n(prod_h->row_starts,2,prod_row_starts);
            std::copy_n(prod_h->col_starts,2,prod_col_starts);
            prod_h->row_starts = prod_row_starts;
            prod_h->col_starts = prod_col_starts;
            hypre_ParCSRMatrixOwnsRowStarts(prod_h) = 1;
            hypre_ParCSRMatrixOwnsColStarts(prod_h) = 1;
        }
#endif

        if (A11)
        {
            hypre_ParCSRMatrix* out;
            PARELAG_ASSERT_HYPRE_ERROR_FLAG(
                hypre_ParCSRMatrixAdd2(1.0,*A11,-1.0*Alpha_,*product,&out));
            return make_unique<mfem::HypreParMatrix>(out);
        }
        else
        {
            *product *= -1.0*Alpha_;

            PARELAG_ASSERT_GLOBAL_HYPRE_ERROR_FLAG();
            return std::unique_ptr<mfem::Operator>{std::move(product)};
        }
    }
    else
    {
        const bool Type_Is_Invalid = true;
        PARELAG_TEST_FOR_EXCEPTION(
            Type_Is_Invalid,
            std::runtime_error,
            "Schur complement type = \"" << Type_ << "\" is invalid.\n"
            "Valid types are \"MASS\" and \"DIAGONAL\"");
    }
    return nullptr;
}

}// namespace parelag
