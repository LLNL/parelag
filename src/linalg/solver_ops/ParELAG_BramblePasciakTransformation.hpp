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


#ifndef PARELAG_BRAMBLEPASCIAKTRANSFORMATION_HPP_
#define PARELAG_BRAMBLEPASCIAKTRANSFORMATION_HPP_

#include "amge/DeRhamSequence.hpp"
#include "amge/ElementalMatricesContainer.hpp"

#include "linalg/dense/ParELAG_Eigensolver.hpp"
#include "linalg/solver_core/ParELAG_SolverState.hpp"
#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"

namespace parelag
{

/** \class BramblePasciakTransformation
 *  \brief Computes the Bramble-Pasciak transformation operator.
 */
class BramblePasciakTransformation
{
    using local_index_type = HYPRE_Int;
    using globa_index_type = HYPRE_Int;
    using value_type = HYPRE_Complex;

public:
    /** \name Constructors and destructor */
    ///@{

    /** \brief The default constructor. */
    BramblePasciakTransformation() = default;

    /** \brief Destructor. */
    ~BramblePasciakTransformation() = default;

    ///@}
    /** \name Factory methods */
    ///@{

    /** \brief Build the transformation operator.
     *
     *  \note This does method does *not* assemble the transformation
     *        operator into a monolithic matrix. That is left as a
     *        separate step for added solver flexibility.
     *
     *  \param blo_in The main saddle-point block operator. Must be
     *                block 2x2.
     *  \param state The SolverState object for this system.
     *
     *  \return The transformation operator.
     */
    std::unique_ptr<MfemBlockOperator> BuildOperator(
        MfemBlockOperator& blo_in, SolverState& state);

    ///@}

private:

    /** \name Private factory methods */
    ///@{

    /** \brief This builds the local-to-a-process \f$M^{-1}\f$. */
    std::unique_ptr<mfem::SparseMatrix> BuildLocalMInverse(
        ElementalMatricesContainer& elem_container);

    /** \brief This builds the global (parallel) \f$M^{-1}\f$. */
    std::unique_ptr<mfem::HypreParMatrix> BuildMInverse(
        ElementalMatricesContainer& elem_container,
        DofHandler& dof_handler);

    ///@}

private:

    /** \brief The \f$M^{-1}\f$ operator. */
    std::shared_ptr<mfem::HypreParMatrix> invM_;

};// class BramblePasciakTransformation

/** \class BramblePasciakTransformation
 *
 *  The Bramble-Pasciak transformation converts a
 *  saddle-point matrix to one that is positive-definite. Suppose
 *
 *  \f[
 *      \mathcal{F} = \begin{bmatrix}
 *          A & B^T \\ B & -C
 *      \end{bmatrix},
 *  \f]
 *
 *  where \f$A\f$ is symmetric positive-definite. Given a symmetric
 *  positive-definite matrix, \f$M\f$, such that \f$A-M\f$ is
 *  positive-definite, we can transform \f$\mathcal{F}\f$ to a
 *  symmetric positive-definite matrix with the following
 *  transformation:
 *
 *  \f[
 *      \mathcal{F}_{\text{spd}} = \begin{bmatrix}
 *          AM^{-1}-I & 0 \\ BM^{-1} & -I
 *      \end{bmatrix}\begin{bmatrix}
 *          A & B^T \\ B & -C
 *      \end{bmatrix} = \begin{bmatrix}
 *          AM^{-1}A-A & (AM^{-1}-I)B^{T} \\ B(M^{-1}A-I) & C+BM^{-1}B^{T}
 *      \end{bmatrix}.
 *  \f]
 *
 *  This class computes a diagonal \f$M\f$ by solving generalized
 *  eigenproblems on each agglomerate and performing a finite-element
 *  assembly type operation on agglomerate-level diagonal
 *  matrices. For each agglomerate, \f$\tau\in\mathcal{T}_{h}\f$, let
 *  \f$A_{\tau}\f$ be the agglomerate mass matrix and let \f$D_{\tau}
 *  = text{diag}(A_{\tau})\f$. We solve the generalized eigenproblems:
 *
 *  \f[
 *      D_{\tau}q = \lambda_{\text{max}} A_{\tau} q
 *  \f]
 *
 *  We then assemble \f$M = \{M_{\tau} =
 *  \tfrac{1}{2\lambda_{\text{max}}}D_{\tau}\}_{tau\in\mathcal{T}_{h}}\f$
 *  to get the global matrix, \f$M\f$.
 *
 *  \todo Implement Chebyshev refinement of \f$M^{-1}\f$.
 */

}// namespace parelag
#endif
