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


#ifndef PARELAG_SUPERLUDISTSOLVER_HPP_
#define PARELAG_SUPERLUDISTSOLVER_HPP_

#include <memory>

#include <mfem.hpp>

#include "ParELAG_Config.h"
#include "linalg/solver_core/ParELAG_DirectSolver.hpp"
#include "linalg/sparse_direct/ParELAG_SuperLUDist_Helpers.hpp"

namespace parelag
{

/** \class SuperLUDistSolver
 *  \brief This class manages a direct solve with SuperLUDist.
 *
 *  \todo Currently, only the default SuperLUDist options are
 *        supported for simplicity.
 */
template <typename Scalar = double>
class SuperLUDistSolver : public DirectSolver
{
public:

    /** \name Constructors and destructor */
    ///@{

    /** \brief Default constructor. */
    SuperLUDistSolver() = default;

    /** \brief Construct from an operator. */
    SuperLUDistSolver(const std::shared_ptr<mfem::Operator>& A,
                      bool GloballyReplicated=false);

    /** \brief Destructor handles deallocation of SuperLUDist types. */
    ~SuperLUDistSolver();

    ///@}
    /** \name The mfem::Solver interface */
    ///@{

    /** \brief Solves the system A*X = B.
     *
     *  \note Contents of X are overwritten.
     *
     *  \warning This will call Factor() if the matrix hasn't been factored.
     *
     *  \param[in]     B  The system right-hand side
     *  \param[in,out] X  The system initial guess and output solution.
     */
    void Mult(const mfem::Vector& B, mfem::Vector& X) const override;

    ///@}

private:

    /** \brief Handle the actual SLUDIST function call for the
     *         solve.
     */
    void _do_mult(const mfem::Vector& B, mfem::Vector& X) const;

    /** \brief Handle the actual SLU function call for the solve in
     *         the globally replicated case.
     */
    void _do_mult_global(const mfem::Vector& B, mfem::Vector& X) const;

    /** \brief Create the process grid for the matrix. */
    void _do_create_process_grid();

    /** \name Special things for this class */
    ///@{

    /** \brief Computes the LU decomposition of A.
     *
     *  \warning This method essentially forces a refactor. No
     *           checking is done as to whether the matrix is already
     *           factored. As factoring is quite expensive, this
     *           should only be called once.
     */
    void _do_factor() override;

    ///@}

    /** \name The parelag::Solver interface */
    ///@{

    /** \brief Handle the setting of the operator.
     *
     *  \note This resets the solver's state. If the previous operator
     *        was factored, a refactorization will be forced (this
     *        requires a separate call to Factor()).
     */
    void _do_set_operator(const std::shared_ptr<mfem::Operator>& A) override;

    ///@}

private:
    // The type and function caller for this class
    using slud_traits = SLUDIST_TypeTraits<Scalar>;
    using slud_type = typename slud_traits::type;
    using slud_caller = SLUDIST_Caller<slud_type>;

    std::shared_ptr<mfem::Operator> A_;
    MPI_Comm Comm_;

    mutable struct SLUDIST_Data
    {
        // The problem matrices
        SLUDIST::SuperMatrix A;
        typename slud_traits::LUstruct_t LU;
        // A = Row-permuted A_ (column-permuted A^T)
        // B = SLU Dense Matrix version of RHS
        // L = L factor of A^T
        // U = U factor of A^T

        // Special SLU types
#ifdef ParELAG_SuperLUDist_HAVE_VERSION_5
        SLUDIST::superlu_dist_options_t options;
        SLUDIST::superlu_dist_mem_usage_t mem_use;
#else
        SLUDIST::superlu_options_t options;
        SLUDIST::mem_usage_t mem_use;
#endif
        SLUDIST::SuperLUStat_t stat;
        SLUDIST::gridinfo_t grid;
        SLUDIST::ScalePermstruct_t scale_perm;
        typename slud_traits::SOLVEstruct_t solve;
        SLUDIST::Pslu_freeable_t freeable;

        MPI_Comm symb_comm;
        int nDomains;

        // Permutation vectors
        std::vector<SLUDIST::int_t> perm_r;
        std::vector<SLUDIST::int_t> perm_c;

        SLUDIST::int_t *sizes, *fstVtxSep;

        // Special things we need
        int relax;
        int panel_size;

    } Data_;

    // Problem things
    bool GloballyReplicated_ = false;
    mutable bool Factored_ = false;
    mutable bool AlreadyInitializedOnce_ = false;

    // These are the dimensions of the matrix used by SuperLU. They
    // are either the dimension of the mfem::SparseMatrix or of the
    // *gathered* hypre_CSRMatrix. "int" because this is what
    // SuperLU_DIST wants.
    int GlobalNumRows_;
    int GlobalNumCols_;

    // If GloballyReplicated_, this will be the global CSR matrix;
    // else this is the CSR matrix describing the locally owned rows
    // in terms of global column ids.
    using delcsr_t = decltype(&hypre_CSRMatrixDestroy);
    using csrptr_t = std::unique_ptr<hypre_CSRMatrix, delcsr_t>;
    csrptr_t A_hypre_ = csrptr_t{nullptr,hypre_CSRMatrixDestroy};

};// class SuperLUDistSolver
}// namespace parelag
#endif /* PARELAG_SUPERLUDISTSOLVER_HPP_ */
