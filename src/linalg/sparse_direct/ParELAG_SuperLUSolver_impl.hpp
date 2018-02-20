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


#ifndef PARELAG_SUPERLUSOLVER_IMPL_HPP_
#define PARELAG_SUPERLUSOLVER_IMPL_HPP_

#include "ParELAG_SuperLUSolver.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{

template <typename Scalar>
SuperLUSolver<Scalar>::SuperLUSolver()
    : DirectSolver(0,0,false),
      Factored_(false)
{
    // Initialize the default options; no printing though
    SLU::set_default_options(&(Data_.options));
    Data_.options.PrintStat = SLU::NO;

    // Initialize the stats
    SLU::StatInit(&(Data_.stat));

    // Get the optimal parameters:
    Data_.panel_size = SLU::sp_ienv(1);
    Data_.relax = SLU::sp_ienv(2);

    // Initialize the pointer members
    Data_.A.Store = nullptr;
    Data_.L.Store = nullptr;
    Data_.U.Store = nullptr;
    Data_.B.Store = nullptr;
}

template <typename Scalar>
SuperLUSolver<Scalar>::SuperLUSolver(const std::shared_ptr<mfem::Operator>& A)
    : DirectSolver(A->Width(), A->Height(), false),
      Factored_(false),
      A_hypre_global_(nullptr,hypre_CSRMatrixDestroy)
{
    // Initialize the default options; no printing though
    SLU::set_default_options(&(Data_.options));
    Data_.options.PrintStat = SLU::NO;

    // Initialize the stats
    SLU::StatInit(&(Data_.stat));

    // Get the optimal parameters:
    Data_.panel_size = SLU::sp_ienv(1);
    Data_.relax = SLU::sp_ienv(2);

    // Initialize the pointer members
    Data_.A.Store = nullptr;
    Data_.L.Store = nullptr;
    Data_.U.Store = nullptr;
    Data_.B.Store = nullptr;

    SetOperator(A);
}// end SuperLUSover()

template <typename Scalar>
void SuperLUSolver<Scalar>::_do_set_operator(
    const std::shared_ptr<mfem::Operator>& A)
{
    // Reset the operator
    this->width = A->Height();
    this->height = A->Width();

    auto A_par = dynamic_cast<mfem::HypreParMatrix *>(A.get());
    auto A_ser = dynamic_cast<mfem::SparseMatrix *>(A.get());

    PARELAG_TEST_FOR_EXCEPTION(
        (!A_par) && (!A_ser),
        std::runtime_error,
        "SuperLUSolver::SetOperator(): "
        "The operator must be either an mfem::HypreParMatrix or "
        "an mfem::SparseMatrix.");

    A_ = A;

    if (A_par)
    {
        IsParallelMatrix_ = true;

        // Get the parallel hypre ptr
        hypre_ParCSRMatrix * A_par_h =  *A_par;

        // Gather the hypre_ParCSRMatrix into a (globally replicated)
        // hypre_CSRMatrix.
        hypre_CSRMatrix *A_global = hypre_ParCSRMatrixToCSRMatrixAll(A_par_h);
        A_hypre_global_ = csrptr_t(A_global, hypre_CSRMatrixDestroy);
        // No .reset() because there's no way to specify the deleter

        // Now that I have the matrix I want to factor, I need
        // to put it into a format that SLU likes.
        NumRows_ = hypre_CSRMatrixNumRows(A_global);
        NumCols_ = hypre_CSRMatrixNumCols(A_global);

    }
    else if (A_ser)
    {
        IsParallelMatrix_ = false;

        NumRows_ = A_->Height();
        NumCols_ = A_->Width();
    }

    // Resize the permutation vectors. Remember that these
    // are reversed since we're passing in the "transposed
    // matrix"...
    Data_.perm_c.resize(NumRows_);
    Data_.perm_r.resize(NumCols_);

    // Reset factored flag to ensure a refactor
    Factored_ = false;
}


template <typename Scalar>
void SuperLUSolver<Scalar>::_do_factor()
{
    // Check if we have data already (we could be refactoring)
    if (Data_.L.Store)
    {
        SLU::Destroy_SuperNode_Matrix(&(Data_.L));
        SLU::Destroy_CompCol_Matrix(&(Data_.U));
        Data_.L.Store = nullptr;
        Data_.U.Store = nullptr;
    }

    // Custom deleter for SuperLU matrix
    auto delSuperMat = [](SLU::SuperMatrix * supermat)
        {
            Destroy_SuperMatrix_Store(supermat);
            SUPERLU_FREE(supermat);
        };
    std::unique_ptr<SLU::SuperMatrix,decltype(delSuperMat)>
        AA{static_cast<SLU::SuperMatrix *>(
            SLU::SUPERLU_MALLOC(sizeof(SLU::SuperMatrix))),
            delSuperMat};

    constexpr SLU::Dtype_t dtype = slu_traits::Dtype;
    if (IsParallelMatrix_)
    {
        hypre_CSRMatrix *A_global = A_hypre_global_.get();

        const auto nnz = hypre_CSRMatrixNumNonzeros(A_global);

        // We are lying to it here. Our matrix is a *row* matrix;
        // they want a *column* matrix. So we give it the "columns
        // of A^T" (rows of A) and ask it to solve the transpose
        // system during Mult() (essentially solving (A^T)^T*X=B).
        slu_caller::Create_CompCol_Matrix(AA.get(),
                                          NumCols_,
                                          NumRows_,
                                          nnz,
                                          hypre_CSRMatrixData(A_global),
                                          hypre_CSRMatrixJ(A_global),
                                          hypre_CSRMatrixI(A_global),
                                          SLU::SLU_NC,
                                          dtype,
                                          SLU::SLU_GE);

    }
    else
    {
        auto A_ser = dynamic_cast<mfem::SparseMatrix *>(A_.get());

        PARELAG_ASSERT(A_ser);

        const auto nnz = A_ser->NumNonZeroElems();
        slu_caller::Create_CompCol_Matrix(AA.get(),
                                          NumCols_,
                                          NumRows_,
                                          nnz,
                                          A_ser->GetData(),
                                          A_ser->GetJ(),
                                          A_ser->GetI(),
                                          SLU::SLU_NC,
                                          dtype,
                                          SLU::SLU_GE);
    }

    // Get the column permutation
    const int permc_spec = Data_.options.ColPerm;
    SLU::get_perm_c(permc_spec, AA.get(), Data_.perm_c.data());

    // Now compute the preordering
    std::vector<int> etree(NumRows_);
    SLU::sp_preorder(&(Data_.options),AA.get(),Data_.perm_c.data(),
                     etree.data(),&(Data_.A));

    // Factor
    int info = 0;
    slu_caller::gstrf(&(Data_.options),&(Data_.A),
                     Data_.relax,Data_.panel_size,
                     etree.data(),
                     nullptr,0,//lwork = 0
                     Data_.perm_c.data(),Data_.perm_r.data(),
                     &(Data_.L),&(Data_.U),
#ifdef ParELAG_SuperLU_HAVE_VERSION_5
                      &(Data_.glu),
#endif
                     &(Data_.stat),&info);

    // Cleanup the permuted A
    SLU::Destroy_CompCol_Permuted(&(Data_.A));

    PARELAG_TEST_FOR_EXCEPTION(
        info != 0,
        std::runtime_error,
        "SuperLUSolver::Factor(): "
        "GSTRF returned nonzero info = " << info << ".");

    Factored_ = true;
}// end Factor()

template <typename Scalar>
void
SuperLUSolver<Scalar>::
Mult(const mfem::Vector& B, mfem::Vector& X) const
{
    // Actual solve call -- Since we factored A^T, we pass in the
    // TRANS flag to solve (A^T)^T*X = B
    _do_mult(B,X,SLU::TRANS);
}

template <typename Scalar>
void
SuperLUSolver<Scalar>::
MultTranspose(const mfem::Vector& B, mfem::Vector& X) const
{
    // Actual solve call -- Since we factored A^T, we pass in the
    // NOTRANS flag to solve (A^T)*X = B
    _do_mult(B,X,SLU::NOTRANS);
}

template <typename Scalar>
SuperLUSolver<Scalar>::~SuperLUSolver()
{
    SLU::StatFree(&(Data_.stat));

//    if (Data_.A.Store != nullptr)
//        SLU::Destroy_CompCol_Permuted(&(Data_.A));
    if (Data_.L.Store != nullptr)
    {
        Destroy_CompCol_Matrix(&(Data_.U));
        Destroy_SuperNode_Matrix(&(Data_.L));
    }

    // ???
    //Destroy_SuperMatrix_Store(&(Data_.B));
    //Destroy_SuperMatrix_Store(&(Data_.X));
}

template <typename Scalar>
void
SuperLUSolver<Scalar>::
_do_mult(const mfem::Vector& B,mfem::Vector& X,SLU::trans_t trans) const
{
    using vecdestroy_t = decltype(&hypre_SeqVectorDestroy);
    using vecptr_t = std::unique_ptr<hypre_Vector,vecdestroy_t>;

    // Make sure we've factored our matrix.
    PARELAG_ASSERT(Factored_);

    // I think we need to make a copy of the RHS, which is a bummer.
    // The rhs as a hypre_Vector. This guy owns his data.
    vecptr_t rhs_seq{nullptr,hypre_SeqVectorDestroy};

    // I cannot use an std::vector<slu_type> here because it will
    // delete its data, which I can't own without copying it. I don't
    // want to copy it twice... So I'll use an
    // std::unique_ptr<slu_type[],delrhs_t> with a special deleter
    // so that we only delete rhs_data if we own it.
    auto delrhs = [&rhs_seq](slu_type * ptr)
    {
        if (rhs_seq.get() == nullptr)
            delete[] ptr;
    };
    std::unique_ptr<slu_type[],decltype(delrhs)> rhs_data{
        nullptr,delrhs};

    // We need to track the local dof range
    int lower_b, upper_b;
    if (IsParallelMatrix_)
    {
        // I should check if I have a HypreParVector
        auto B_hypre = dynamic_cast<const mfem::HypreParVector *>(&B);

        if (B_hypre)
        {
            hypre_ParVector * tmp = *B_hypre;

            // Gather the whole vector to me
            rhs_seq = vecptr_t{hypre_ParVectorToVectorAll(tmp),
                               hypre_SeqVectorDestroy};
            // Set the data
            rhs_data.reset(hypre_VectorData(rhs_seq));

            lower_b = hypre_ParVectorFirstIndex(tmp);
            upper_b = hypre_ParVectorLastIndex(tmp)+1;
        }
        else
        {
            // Fake the HypreParVector from the "B" we were given
            auto A_hypre = dynamic_cast<mfem::HypreParMatrix *>(A_.get());
            B_hypre = new mfem::HypreParVector(A_hypre->GetComm(),
                                               A_hypre->GetGlobalNumRows(),
                                               B.GetData(),
                                               A_hypre->GetRowStarts());

            hypre_ParVector * tmp = *B_hypre;
            rhs_seq = vecptr_t{hypre_ParVectorToVectorAll(tmp),
                               hypre_SeqVectorDestroy};

            rhs_data.reset(hypre_VectorData(rhs_seq));

            lower_b = hypre_ParVectorFirstIndex(tmp);
            upper_b = hypre_ParVectorLastIndex(tmp)+1;
            delete B_hypre;
        }

        // Sanity check
        PARELAG_ASSERT(hypre_VectorSize(rhs_seq) == NumRows_);
    }
    else
    {
        // Sanity check
        PARELAG_ASSERT(B.Size() == NumRows_);

        rhs_data.reset(new slu_type[B.Size()]);
        std::copy(B.GetData(), B.GetData() + B.Size(), rhs_data.get());

        // Set the bounds
        lower_b = 0;
        upper_b = NumRows_;
    }

    // At this point, rhs_data contains a clean copy of the data
    // that we want to create our SLU DenseMatrix from.
    constexpr int nrhs = 1;
    constexpr SLU::Dtype_t dtype = slu_traits::Dtype;
    slu_caller::Create_Dense_Matrix(&(Data_.B),// SuperMatrix
                                    NumRows_,// Height
                                    nrhs,// NumVectors
                                    rhs_data.get(),// Data
                                    NumRows_,// Leading dimension
                                    SLU::SLU_DN,// Stype_t
                                    dtype,// Dtype_t
                                    SLU::SLU_GE);// Mtype_t

    // Actual SLU solve call
    int info = 0;
    slu_caller::gstrs(trans,&(Data_.L),&(Data_.U),
                      Data_.perm_c.data(),Data_.perm_r.data(),
                      &(Data_.B),&(Data_.stat),&info);

    PARELAG_TEST_FOR_EXCEPTION(
        info != 0,
        std::runtime_error,
        "SuperLUSolver::_do_mult(): "
        "GSTRS returned nonzero info = " << info << ".");

    // So now I have a global solution on each processor. I need to
    // redistribute the local parts to each processor.
    std::copy(rhs_data.get()+lower_b,rhs_data.get()+upper_b,X.GetData());

    // Cleanup the mess
    SLU::Destroy_SuperMatrix_Store(&(Data_.B));
}//end Mult_impl()

}// namespace parelag
#endif /* PARELAG_SUPERLUSOLVER_IMPL_HPP_ */
