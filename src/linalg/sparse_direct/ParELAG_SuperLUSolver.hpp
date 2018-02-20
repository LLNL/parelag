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


#ifndef PARELAG_SUPERLUSOLVER_HPP_
#define PARELAG_SUPERLUSOLVER_HPP_

#include <memory>

#include <mfem.hpp>

#include "ParELAG_Config.h"
#include "linalg/solver_core/ParELAG_DirectSolver.hpp"
#include "linalg/sparse_direct/ParELAG_SuperLU_Helpers.hpp"

namespace parelag
{

/** \class SuperLUSolver
 *  \brief This class manages a direct solve with SuperLU.
 *
 *  \todo Currently, only the default SuperLU options are supported
 *  for simplicity.
 *
 *  \warning In parallel, this class will globally replicate the
 *  ParCSRMatrix on each processor. Then the factorization and solve
 *  will happen simultaneously on each processor. This is certainly
 *  flop-intensive, but these processors have nothing better to
 *  do. And it saves me the trouble of redistributing
 *  afterwards. Additionally, if one is running in serial but uses a
 *  HyperParMatrix as the operator type, the "globally replicated
 *  CSRMatrix" will still be created, though it isn't
 *  necessary. Please keep that in mind.
 */
// FIXME (trb 03/08/16): Should include a ParameterList interaction to
// deviate from the standard SuperLU options. This shouldn't affect
// the interface.
template <typename Scalar = double>
class SuperLUSolver : public DirectSolver
{
public:
    /** \name Constructors and destructor */
    ///@{

    /** \brief Default constructor. */
    SuperLUSolver();

    /** \brief Construct from an operator.
     *
     *  \param A  The system matrix to be solved.
     */
    SuperLUSolver(const std::shared_ptr<mfem::Operator>& A);

    /** \brief Destructor.
     *
     *  Handles deallocation of SuperLU types.
     */
    ~SuperLUSolver();

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

    /** \brief Solves the system A^T*X=B.
     *
     *  \note Contents of X are overwritten.
     *
     *  \warning This will call Factor() if the matrix hasn't been factored.
     *
     *  \param[in]     B  The system right-hand side
     *  \param[in,out] X  The system initial guess and output solution.
     */
    void MultTranspose(const mfem::Vector& B, mfem::Vector& X) const override;

    ///@}

private:

    /** \brief Handle the actual SLU function call for the solve. */
    void _do_mult(const mfem::Vector& B,
                  mfem::Vector& X,
                  SLU::trans_t trans) const;

    /** \name DirectSolver functions */
    ///@{

    /** \brief Computes the LU decomposition of A.
     *
     *  \warning This method essentially forces a refactor. No
     *  checking is done as to whether the matrix is already
     *  factored. As factoring is quite expensive, this should only be
     *  called once.
     */
    void _do_factor() override;

    ///@}

    /** \name The parelag::Solver interface */
    ///@{

    /** \brief Handle the setting of the operator.
     *
     *  \note This resets the solver's state. If the previous operator was
     *        factored, a refactorization will be forced (this requires a
     *        separate call to Factor()).
     */
    void _do_set_operator(const std::shared_ptr<mfem::Operator>& A) override;

    ///@}


private:
    // The type and function caller for this class
    using slu_traits = SLU_TypeTraits<Scalar>;
    using slu_type = typename slu_traits::type;
    using slu_caller = SLU_Caller<slu_type>;

    std::shared_ptr<mfem::Operator> A_;

    mutable struct SLU_Data
    {
        // The problem matrices
        SLU::SuperMatrix A, B, L, U;
        // A = Row-permuted A_ (column-permuted A^T)
        // B = SLU Dense Matrix version of RHS
        // L = L factor of A^T
        // U = U factor of A^T

        // Special SLU types
        SLU::superlu_options_t options;
        SLU::SuperLUStat_t stat;
#ifdef ParELAG_SuperLU_HAVE_VERSION_5
        SLU::GlobalLU_t glu;
#endif

        // Permutation vectors
        std::vector<int> perm_r;// row permutations from partial pivoting
        std::vector<int> perm_c;// column permutation vector

        // Special things we need
        int relax;
        int panel_size;

    } Data_;

    // Problem things
    bool IsParallelMatrix_;
    mutable bool Factored_;

    // These are the dimensions of the matrix used by SuperLU. They
    // are either the dimension of the mfem::SparseMatrix or of the
    // *gathered* hypre_CSRMatrix.
    int NumRows_;
    int NumCols_;

    // In case we're using Hypre objects, store the gathered guy
    using delcsr_t = decltype(&hypre_CSRMatrixDestroy);
    using csrptr_t = std::unique_ptr<hypre_CSRMatrix, delcsr_t>;
    csrptr_t A_hypre_global_ = csrptr_t(nullptr,hypre_CSRMatrixDestroy);

};// class SuperLUSolver
}// namespace parelag
#endif /* PARELAG_SUPERLUSOLVER_HPP_ */
