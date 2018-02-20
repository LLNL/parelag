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


#ifndef PARELAG_SUPERLUDISTSOLVER_IMPL_HPP_
#define PARELAG_SUPERLUDISTSOLVER_IMPL_HPP_

#include "ParELAG_SuperLUDistSolver.hpp"

#include <cmath>

namespace parelag
{

template <typename Scalar>
void SuperLUDistSolver<Scalar>::_do_create_process_grid()
{
    // Blow up the old one if it exists
    if (AlreadyInitializedOnce_)
    {
        SLUDIST::superlu_gridexit(&(Data_.grid));
        SLUDIST::PStatFree(&(Data_.stat));
    }

    int myrank, mysize;
    MPI_Comm_size(Comm_,&mysize);
    MPI_Comm_rank(Comm_,&myrank);

    SLUDIST::int_t prows = 1, pcols;
    while (prows*prows <= mysize) ++prows;
    --prows;
    pcols = mysize / prows;
    while (prows*pcols != mysize)
    {
        prows -= 1;
        pcols = mysize / prows;
    }
    // Note: might just be a row of procs (mysize == prime)

    // Initialize the SuperLU structs
    SLUDIST::superlu_gridinit(Comm_,prows,pcols,&(Data_.grid));
    // FIXME: This resets all options to defaults
    SLUDIST::set_default_options_dist(&(Data_.options));
    Data_.options.PrintStat = SLUDIST::NO;
    SLUDIST::PStatInit(&(Data_.stat));

    // Symbolic factorization communicator
    if (AlreadyInitializedOnce_ && Data_.symb_comm
        && (Data_.symb_comm != MPI_COMM_NULL)
        && (Data_.symb_comm != MPI_COMM_SELF)
        && (Data_.symb_comm != MPI_COMM_WORLD))
    {
        MPI_Comm_free(&(Data_.symb_comm));
    }
    Data_.symb_comm = MPI_COMM_NULL;
    Data_.nDomains = (int)(std::pow(2,(int) std::log2(prows*pcols)));
    const int color = (myrank < Data_.nDomains) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(Comm_,color,myrank,&(Data_.symb_comm));

    if (Data_.sizes && AlreadyInitializedOnce_)
        free(Data_.sizes);
    if (Data_.fstVtxSep && AlreadyInitializedOnce_)
        free(Data_.fstVtxSep);
    if (Data_.A.Store && AlreadyInitializedOnce_)
        SLUDIST::Destroy_SuperMatrix_Store_dist(&(Data_.A));

    Data_.sizes = nullptr;
    Data_.fstVtxSep = nullptr;
    Data_.A.Store = nullptr;
}

template <typename Scalar>
SuperLUDistSolver<Scalar>::
SuperLUDistSolver(const std::shared_ptr<mfem::Operator>& A,
    bool GloballyReplicated)
    : DirectSolver(A->Width(),A->Height(),false),
      GloballyReplicated_(GloballyReplicated),
      Factored_(false),
      A_hypre_(nullptr,hypre_CSRMatrixDestroy)
{
    SetOperator(A);
}

template <typename Scalar>
SuperLUDistSolver<Scalar>::~SuperLUDistSolver()
{
    if (Data_.A.Store)
        SLUDIST::Destroy_SuperMatrix_Store_dist(&(Data_.A));

    if (Factored_)
        slud_caller::Destroy_LU(this->GlobalNumRows_,&(Data_.grid),&(Data_.LU));
    slud_caller::LUstructFree(&(Data_.LU));

    if (Data_.options.SolveInitialized == SLUDIST::YES)
        slud_caller::SolveFinalize(&(Data_.options),&(Data_.solve));

    if (Data_.symb_comm && (Data_.symb_comm != MPI_COMM_NULL)
        && (Data_.symb_comm != MPI_COMM_SELF)
        && (Data_.symb_comm != MPI_COMM_WORLD))
    {
        MPI_Comm_free(&(Data_.symb_comm));
    }
#ifdef ParELAG_ENABLE_PARMETIS
    if (Data_.sizes)
        free(Data_.sizes);
    if (Data_.fstVtxSep)
        free(Data_.fstVtxSep);
#endif
    SLUDIST::superlu_gridexit(&(Data_.grid));
    SLUDIST::PStatFree(&(Data_.stat));
}

template <typename Scalar>
void
SuperLUDistSolver<Scalar>::_do_set_operator(
    const std::shared_ptr<mfem::Operator>& A)
{
    auto A_par = dynamic_cast<mfem::HypreParMatrix *>(A.get());
    PARELAG_TEST_FOR_EXCEPTION(
        !A_par,
        std::runtime_error,
        "SuperLUDist::SetOperator(): "
        "The operator must be a HypreParMatrix!");

    // Cleanup the old data
    if (Data_.A.Store && AlreadyInitializedOnce_)
    {
        SLUDIST::Destroy_SuperMatrix_Store_dist(&(Data_.A));
        Data_.A.Store = nullptr;
    }

    A_ = A;
    Comm_ = A_par->GetComm();

    _do_create_process_grid();

    // Whether we are doing GloballyReplicated_ or distributed, we
    // will always need the single parallel matrix
    hypre_ParCSRMatrix * A_par_h = *A_par;

    // Merge diag and offd into one matrix (global ids)
    hypre_CSRMatrix *A_local = hypre_MergeDiagAndOffd(A_par_h);
    A_hypre_.reset(A_local);

    GlobalNumRows_ = A_par->GetGlobalNumRows();
    GlobalNumCols_ = A_par->GetGlobalNumCols();

    constexpr SLUDIST::Dtype_t Dtype = slud_traits::Dtype;
    if (GloballyReplicated_)
    {
        // Now I need to copy the matrix once more... Yippee.
        // NOTE: These functions won't "throw", of course, but they
        // will ABORT, generally if memory fails to allocate.

        // This is the global matrix in distributed row format.
        SLUDIST::SuperMatrix tmp;
        slud_caller::Create_CompRowLoc_Matrix(
            &tmp,GlobalNumRows_,GlobalNumCols_,
            hypre_CSRMatrixNumNonzeros(A_local),
            hypre_CSRMatrixNumRows(A_local),
            hypre_ParCSRMatrixFirstRowIndex(A_par_h),
            hypre_CSRMatrixData(A_local),
            hypre_CSRMatrixJ(A_local),hypre_CSRMatrixI(A_local),
            SLUDIST::SLU_NR,Dtype,SLUDIST::SLU_GE);

        // Now convert the hypre matrix to the SLU_NC matrix
        slud_caller::CompRow_loc_to_CompCol_global(
            1,&tmp,&(Data_.grid),&(Data_.A) );
    }
    else
    {
        slud_caller::Create_CompRowLoc_Matrix(
            &(Data_.A),GlobalNumRows_,GlobalNumCols_,
            hypre_CSRMatrixNumNonzeros(A_local),
            hypre_CSRMatrixNumRows(A_local),
            hypre_ParCSRMatrixFirstRowIndex(A_par_h),
            hypre_CSRMatrixData(A_local),
            hypre_CSRMatrixJ(A_local),hypre_CSRMatrixI(A_local),
            SLUDIST::SLU_NR,Dtype,SLUDIST::SLU_GE);
    }

    Data_.perm_r.resize(GlobalNumRows_);
    Data_.perm_c.resize(GlobalNumCols_);

    // Set these here in case the raw pointer is invalidated by the resize.
    Data_.scale_perm.perm_r = Data_.perm_r.data();
    Data_.scale_perm.perm_c = Data_.perm_c.data();

    if (Factored_)
    {
        slud_caller::Destroy_LU(this->GlobalNumRows_,&(Data_.grid),&(Data_.LU));
#ifdef ParELAG_ENABLE_PARMETIS
        if (Data_.sizes)
            free(Data_.sizes);
        if (Data_.fstVtxSep)
            free(Data_.fstVtxSep);
#endif
        slud_caller::LUstructFree(&(Data_.LU));
    }

    // Reinitialize this
    slud_caller::LUstructInit(this->GlobalNumCols_,
                              &(Data_.LU));

    Factored_ = false;
    AlreadyInitializedOnce_ = true;
}

template <typename Scalar>
void SuperLUDistSolver<Scalar>::_do_factor()
{
    // This will do the preordering and both the symbolic and the
    // numeric factorizations

    /* CLEANUP */
    if (Data_.options.SolveInitialized == SLUDIST::YES)
        slud_caller::SolveFinalize(&(Data_.options),&(Data_.solve));

    /* PREORDERING */

    // Use the NATURAL ordering
    for (int ii = 0; ii < GlobalNumRows_; ++ii)
        Data_.perm_r[ii] = ii;

    PARELAG_TEST_FOR_EXCEPTION(
        Data_.sizes || Data_.fstVtxSep,
        std::runtime_error,
        "SuperLUDistSolver::Factor(): "
        "Tom made a mistake. This shouldn't be possible.");

    float info = 0.0;
#ifdef ParELAG_ENABLE_PARMETIS
    info = SLUDIST::get_perm_c_parmetis(
        &(Data_.A),Data_.perm_r.data(),Data_.perm_c.data(),
        Data_.grid.nprow*Data_.grid.npcol,Data_.nDomains,
        &(Data_.sizes),&(Data_.fstVtxSep),&(Data_.grid),
        &(Data_.symb_comm));
#else
    std::vector<SLUDIST::int_t> sizevec(2*Data_.nDomains,0),
        fstVtxSepvec(2*Data_.nDomains,0);
    sizevec[2*Data_.nDomains-2] = Data_.A.nrow;
    Data_.sizes = sizevec.data();
    Data_.fstVtxSep = fstVtxSepvec.data();

    for (int ii = 0; ii < GlobalNumCols_; ++ii)
        Data_.perm_c[ii] = ii;
#endif

    PARELAG_TEST_FOR_EXCEPTION(
        info > 0.0,
        std::runtime_error,
        "SuperLUDistSolver::Factor(): "
        "SuperLU_DIST preordering ran out of memory after " <<
        info << " bytes.");

    /* SYMBOLIC FACTORIZATION */

    info = SLUDIST::symbfact_dist(
        Data_.grid.nprow*Data_.grid.npcol,Data_.nDomains,
        &(Data_.A),Data_.perm_c.data(),Data_.perm_r.data(),
        Data_.sizes, Data_.fstVtxSep,&(Data_.freeable),
        &(Data_.grid.comm),&(Data_.symb_comm),&(Data_.mem_use));

#ifdef ParELAG_ENABLE_PARMETIS
    Data_.sizes = nullptr;
    Data_.fstVtxSep = nullptr;
#endif

    PARELAG_TEST_FOR_EXCEPTION(
        info > 0.0,
        std::runtime_error,
        "SuperLUDistSolver::Factor(): "
        "SuperLU_DIST symbolic factorization ran out of memory after " <<
        info << " bytes.");

    /* NUMERIC FACTORIZATION */
    auto Astore = (SLUDIST::NRformat_loc *)Data_.A.Store;
    SLUDIST::int_t nnz_loc = Astore->nnz_loc;
    for (SLUDIST::int_t ii = 0; ii < nnz_loc; ++ii)
        Astore->colind[ii] = Data_.perm_c[Astore->colind[ii]];

    info = slud_caller::dist_psymbtonum(
        SLUDIST::DOFACT,GlobalNumRows_,&(Data_.A),&(Data_.scale_perm),
        &(Data_.freeable),&(Data_.LU), &(Data_.grid));

    PARELAG_TEST_FOR_EXCEPTION(
        info > 0.0,
        std::runtime_error,
        "SuperLUDistSolver::Factor(): "
        "SuperLU_DIST distribute ran out of memory after " <<
        info << " bytes.");

    double anorm = slud_caller::plangs((char *)"I",&(Data_.A),&(Data_.grid));
    int info_int = 0;
    slud_caller::gstrf(
        &(Data_.options),GlobalNumRows_,GlobalNumCols_,anorm,
        &(Data_.LU),&(Data_.grid),&(Data_.stat),&info_int);

    PARELAG_TEST_FOR_EXCEPTION(
        info_int > 0,
        std::runtime_error,
        "SuperLUDistSolver::Factor(): "
        "SuperLU_DIST numeric factorization (gstrf) failed! "
        "U(" << info_int << "," << info_int << ") = 0.");

    Factored_ = true;
}

template <typename Scalar>
void
SuperLUDistSolver<Scalar>::
Mult(const mfem::Vector& B, mfem::Vector& X) const
{
    if (GloballyReplicated_)
        _do_mult_global(B,X);
    else
        _do_mult(B,X);
}

template <typename Scalar>
void
SuperLUDistSolver<Scalar>::
_do_mult_global(mfem::Vector const&,mfem::Vector&) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        true,
        not_implemented_error,
        "SuperLUDistSolver::_do_mult_global(): Function not implemented!");
}

template <typename Scalar>
void
SuperLUDistSolver<Scalar>::
_do_mult(const mfem::Vector& B,mfem::Vector& X) const
{
    const int nEntries = B.Size();
    constexpr int nrhs = 1;
    auto Astore = (SLUDIST::NRformat_loc*) Data_.A.Store;
    PARELAG_ASSERT(nEntries == Astore->m_loc);

    // Copy the data for SLU, which overwrites
    std::vector<Scalar> sluX(B.GetData(),B.GetData()+nEntries);

    if (Data_.options.SolveInitialized == SLUDIST::NO)
        slud_caller::SolveInit(
            &(Data_.options),&(Data_.A),Data_.perm_r.data(),Data_.perm_c.data(),
            nrhs,&(Data_.LU),&(Data_.grid),&(Data_.solve));

    int info = 0;
    slud_caller::gstrs(
        GlobalNumRows_,&(Data_.LU),&(Data_.scale_perm),&(Data_.grid),
        sluX.data(),nEntries,Astore->fst_row,nEntries,nrhs,
        &(Data_.solve),&(Data_.stat),&info);

    PARELAG_TEST_FOR_EXCEPTION(
        info,
        std::runtime_error,
        "SuperLUDistSolver::_do_mult(): "
        "Argument " << -info << " had an illegl value.");

    slud_caller::Permute_Dense_Matrix(
        Astore->fst_row,Astore->m_loc,Data_.solve.row_to_proc,
        Data_.solve.inv_perm_c,sluX.data(),Astore->m_loc,
        X.GetData(),Astore->m_loc,nrhs,&(Data_.grid));
}

}// namespace parelag
#endif /* PARELAG_SUPERLUDISTSOLVER_IMPL_HPP_ */
