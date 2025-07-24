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


#include "ParELAG_BramblePasciakTransformation.hpp"

#include <numeric>

#include "hypreExtension/hypreExtension.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{

std::unique_ptr<mfem::HypreParMatrix>
BramblePasciakTransformation::BuildMInverse(
    ElementalMatricesContainer& elem_container,
    DofHandler& dof_handler)
{
    // So, rather than implementing something like
    // "DiagonalElementalMatricesContainer", I'm going to just do
    // this with one big stl vector.
    std::vector<double> unassembled_M;

    const auto numElements = elem_container.GetNumEntities();

    size_t unassembled_size = 0, unassembled_width = 0;
    for (const auto& mat : elem_container)
    {
        unassembled_size += mat->Height();
        unassembled_width += mat->Width();
    }
    PARELAG_ASSERT(unassembled_size == unassembled_width);

    unassembled_M.resize(unassembled_size);

    auto m_rng_beg = unassembled_M.begin(), m_rng_end = m_rng_beg;

    SymEigensolver eigs;
    mfem::Vector diag_tmp;
    std::vector<double> eigenvalue(1);
    for (auto elem = 0; elem < numElements; ++elem)
    {
        // Make a copy for the eigensolver to destroy
        mfem::DenseMatrix A(elem_container.GetElementalMatrix(elem));

        A.GetDiag(diag_tmp);
        double * diag = diag_tmp.GetData();

        // Copy into rdof-space vector
        m_rng_end = std::copy(diag,diag+diag_tmp.Size(),
                              m_rng_beg);

        // Coompute sqrt
        std::transform(diag,diag+diag_tmp.Size(),diag,
                       [](const double& a){ return 1./std::sqrt(a); });

        // A <- D^{-1/2}*A*D^{-1/2}
        A.LeftScaling(diag_tmp);
        A.RightScaling(diag_tmp);

        // Find eigs of D*x = \lambda*A*x
        auto info = eigs.ComputeFixedNumber(A,eigenvalue,1,1);
        PARELAG_ASSERT(info == 0);

        // Scale the diagonal appropriately
        const auto& scale_val = 0.5*eigenvalue.front();
        m_rng_beg = std::transform(m_rng_beg,m_rng_end,m_rng_beg,
                                   [&scale_val](const double& a)
                                   {return scale_val*a;});
    }

    PARELAG_ASSERT(m_rng_end == unassembled_M.end());

    // Fake a SparseMatrix...
    std::vector<int> row_ptr(unassembled_M.size()+1);
    std::iota(row_ptr.begin(),row_ptr.end(),0);

    // unassembled_M_inv = diag(unassembled_M);
    mfem::SparseMatrix unassembled_M_inv(
        row_ptr.data(),row_ptr.data(),unassembled_M.data(),
        unassembled_M.size(),unassembled_M.size(),
        false,false,true);

    // Assemble into a processor-based local matrix
    auto M_inv_loc = Assemble(AgglomeratedTopology::ELEMENT,
                              unassembled_M_inv,
                              dof_handler, dof_handler);

    // Assemble into a parallel matrix
    auto M_inv = Assemble(dof_handler.GetDofTrueDof(),*M_inv_loc,
                          dof_handler.GetDofTrueDof());

    // Invert the diagonal matrix
    hypre_ParCSRMatrix* tmp_M = *M_inv;
    double* M_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(tmp_M));
    int nLocRows = hypre_ParCSRMatrixNumRows(tmp_M);

    PARELAG_ASSERT(
        hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(tmp_M)) == nLocRows);

    std::transform(M_data,M_data+nLocRows,M_data,
                   [](const double& a){return 1./a;});

    return M_inv;
}


std::unique_ptr<MfemBlockOperator>
BramblePasciakTransformation::BuildOperator(
    MfemBlockOperator& blo_in, SolverState& state)
{

    if (not invM_)
    {
        auto& seq = state.GetDeRhamSequence();
        auto form = state.GetForms().front();
        auto elem_containers = seq.GetElementalMassMatrices(
            form,AgglomeratedTopology::ELEMENT);
        invM_ = BuildMInverse(*elem_containers,
                              *seq.GetDofHandler(form));
    }

    PARELAG_ASSERT(invM_);

    // [ A*M^{-1}-I   0 ][ A B^T ] = [ A*M^{-1}*A - A   A*M^{-1}*B^T - B^T ]
    // [ B*M^{-1}    -I ][ B  0  ]   [ B*M^{-1}*A - B   B*M^{-1}*B^T + W   ]

    // Verify a few things
    PARELAG_ASSERT(blo_in.GetNumBlockRows() == 2);
    PARELAG_ASSERT(blo_in.GetNumBlockCols() == 2);

    // I'm going to deal with my own casting for now, since I need
    // the pointers for "mfem::ParMult" anyway.
    mfem::HypreParMatrix* A =
        dynamic_cast<mfem::HypreParMatrix*>(&blo_in.GetBlock(0,0));
    PARELAG_ASSERT(A);

    mfem::HypreParMatrix* B =
        dynamic_cast<mfem::HypreParMatrix*>(&blo_in.GetBlock(1,0));
    PARELAG_ASSERT(B);

    mfem::HypreParMatrix* BT =
        dynamic_cast<mfem::HypreParMatrix*>(&blo_in.GetBlock(0,1));
    PARELAG_ASSERT(BT);

    mfem::HypreParMatrix* W = nullptr;
    if (!blo_in.IsZeroBlock(1,1))
    {
        W = dynamic_cast<mfem::HypreParMatrix*>(&blo_in.GetBlock(1,1));
        PARELAG_ASSERT(W);
    }

    MPI_Comm comm = A->GetComm();

    auto blo_out = make_unique<MfemBlockOperator>(
        blo_in.CopyRowOffsets(),blo_in.CopyColumnOffsets());

    // Compute A*M^{-1}-I stuff
    {
        // This does the A*M^{-1} part
        auto AMinv_I = ToUnique(mfem::ParMult(A,invM_.get()));
        {
            hypre_ParCSRMatrix* hyp_am = *AMinv_I;

            HYPRE_Int NumRows = hypre_ParCSRMatrixNumRows(hyp_am);
            HYPRE_Int* I = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(hyp_am));
            HYPRE_Complex* Data = hypre_CSRMatrixData(
                hypre_ParCSRMatrixDiag(hyp_am));

            // This part does the "minus I" bit
            for (HYPRE_Int ii = 0; ii < NumRows; ++ii)
                Data[I[ii]] -= 1;
        }

        blo_out->SetBlock(0, 0, std::move(AMinv_I));
    }

    // Compute B*M^{-1} stuff
    {
        auto BMinv = ToUnique(mfem::ParMult(B,invM_.get()));

        blo_out->SetBlock(1, 0, std::move(BMinv));
    }

    // Create a parallel "-I" block
    {
        hypre_ParCSRMatrix* minus_I = nullptr;

        HYPRE_Int* our_row_starts = (HYPRE_Int*)malloc(sizeof(HYPRE_Int)*2);
        HYPRE_Int* our_col_starts = (HYPRE_Int*)malloc(sizeof(HYPRE_Int)*2);
        size_t num_global_rows,num_global_cols;
        if (W)
        {
            hypre_ParCSRMatrix* tmp = *W;
            std::copy_n(hypre_ParCSRMatrixRowStarts(tmp),2,our_row_starts);
            std::copy_n(hypre_ParCSRMatrixColStarts(tmp),2,our_col_starts);
            num_global_rows = tmp->global_num_rows;
            num_global_cols = tmp->global_num_cols;
        }
        else
        {
            hypre_ParCSRMatrix* tmpB = *B, * tmpBT = *BT;
            std::copy_n(hypre_ParCSRMatrixRowStarts(tmpB) ,2,our_row_starts);
            std::copy_n(hypre_ParCSRMatrixColStarts(tmpBT),2,our_col_starts);
            num_global_rows = tmpB->global_num_rows;
            num_global_cols = tmpBT->global_num_cols;
        }
        PARELAG_ASSERT(num_global_rows == num_global_cols);

        // Square
        if ((our_row_starts[0] == our_col_starts[0]) &&
            (our_row_starts[1] == our_col_starts[1]))
        {
            size_t my_num_rows = our_row_starts[1] - our_row_starts[0];
            minus_I = hypre_ParCSRMatrixCreate(
                comm, num_global_rows, num_global_cols,
                our_row_starts, our_col_starts,
                0, my_num_rows, 0);

            PARELAG_ASSERT_HYPRE_ERROR_FLAG(
                hypre_ParCSRMatrixInitialize(minus_I));

            HYPRE_Int* I = hypre_CSRMatrixI(minus_I->diag),
                * J = hypre_CSRMatrixJ(minus_I->diag);
            HYPRE_Complex* D = minus_I->diag->data;

            for (size_t ii = 0; ii < my_num_rows; ++ii)
            {
                I[ii] = ii;
                J[ii] = ii;
                D[ii] = -1.;
            }
            I[my_num_rows] = my_num_rows;

#if MFEM_HYPRE_VERSION <= 22200
            hypre_ParCSRMatrixOwnsRowStarts(minus_I) = 1;
            hypre_ParCSRMatrixOwnsColStarts(minus_I) = 1;
#endif
        }

        PARELAG_ASSERT(minus_I);

        blo_out->SetBlock(1,1,std::make_shared<mfem::HypreParMatrix>(minus_I));
    }

    return blo_out;
}// Compute transformation

}// namespace parelag
