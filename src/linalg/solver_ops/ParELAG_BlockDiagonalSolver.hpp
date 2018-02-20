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


#ifndef PARELAG_BLOCKDIAGONALSOLVER_HPP_
#define PARELAG_BLOCKDIAGONALSOLVER_HPP_

#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"
#include "linalg/solver_core/ParELAG_Solver.hpp"

namespace parelag
{

/** \class BlockDiagonalSolver
 *  \brief Block Diagonal solver that preserves lifetimes of
 *         associated objects.
 *
 *  This class approximates the action of the inverse of the operator
 *
 *  \f[
 *     A = \begin{bmatrix}
 *         A_{00} & \dots & A_{0n} \\
 *         \vdots & \ddots & \vdots \\
 *         A_{n0} & \dots & A_{nn}
 *     \end{bmatrix}
 *  \f]
 *
 *  using the following block-diagonal solver:
 *
 *  \f[
 *     \hat{A} = \begin{bmatrix}
 *         \hat{A}_{00} & & \\ & \ddots & \\ & & \hat{A}_{nn}
 *     \end{bmatrix}
 *  \f]
 *
 *  where \f$\hat{A}_{ii}\f$ is some solver or precondtioner for
 *  \f$A_{ii}\f$ (i.e., \f$\hat{A}_{ii}^{-1}\f$ approximates the
 *  action of \f$A_{ii}^{-1}\f$). The \f$\hat{A}_{ii}^{-1}\f$ may be
 *  specified to be any (appropriate) parelag::Solver object.
 *
 *  MFEM has a similar class, but does not preserve the lifetime of
 *  the blocks. This fixes that issue.
 */
class BlockDiagonalSolver : public Solver
{
public:

    /** \brief Construct from an operator.
     *
     *  Inverse operators must be set using SetInverseDiagonal() or
     *  SetInverseDiagonals().
     *
     *  \param op The underlying operator.
     */
    BlockDiagonalSolver(std::shared_ptr<mfem::Operator> op);

    /** \brief Construct from an operator and a collection of
     *         inverses.
     *
     *  \param op The underlying operator.
     *  \param inv_ops The diagonal inverses. \c inv_ops[i] is the
     *                 inverse for \f$A_{ii}\f$.
     *  \param aux_ops Any additional operators that need to stay
     *                 alive. E.g., if using a Schur-complement-based
     *                 solver, this will ensure that the Schur
     *                 complement operator is not destroyed.
     *
     *  \todo \c aux_ops can probably be factored out if we restrict
     *        subsolvers to be parelag::Solver objects only and
     *        require that such objects keep their operator alive if
     *        necessary.
     */
    BlockDiagonalSolver(
        std::shared_ptr<mfem::Operator> op,
        std::vector<std::shared_ptr<mfem::Solver>> inv_ops,
        std::vector<std::shared_ptr<mfem::Operator>> aux_ops =
        std::vector<std::shared_ptr<mfem::Operator>>());

    /** \name Solver interface functions */
    ///@{

    /** \brief Apply the operator to a vector. */
    void Mult(const mfem::Vector& rhs, mfem::Vector& sol) const override;

    /** \brief Apply the transpose of the operator to a vector. */
    void MultTranspose(
        const mfem::Vector& rhs, mfem::Vector& sol) const override;

    ///@}
    /** \name Extra goodies. */
    ///@{

    /** \brief Set the diagonal inverses.
     *
     *  It is assumed that \c inv_ops[i] is the inverse for
     *  \f$A_{ii}\f$. However, no checking is done that \c
     *  inv_ops.size() == \c num_blocks or that \c size(inv_ops[i]) ==
     *  \c size(\f$A_{ii}\f$).
     *
     *  \param inv_ops The inverses for the diagonal blocks.
     *
     *  \todo Size checking? Of both the vector and the operators?
     *        Debug mode only?
     */
    void SetInverseDiagonals(
        std::vector<std::shared_ptr<mfem::Solver>> inv_ops);

    /** \brief Set a diagonal inverse.
     *
     *  It is assumed that \c inv_op is the inverse for \f$A_{ii}\f$,
     *  where \c i = \c block. However, no checking is done that \c
     *  size(inv_op) == \c size(\f$A_{ii}\f$).
     *
     *  \param block The index of the diagonal block being set.
     *  \param inv_op The inverses for the diagonal block.
     *
     *  \todo Size checking? Debug mode only?
     */
    void SetInverseDiagonal(
        size_t block, std::shared_ptr<mfem::Solver> inv_op);

    ///@}

    /** \name Deleted special functions. */
    ///@{
    /** \brief Disable default construction. */
    BlockDiagonalSolver() = delete;
    ///@}

private:

    void _do_set_operator(const std::shared_ptr<mfem::Operator>& op) override;

private:

    /** \brief The underlying operator as an MfemBlockOperator. */
    std::shared_ptr<MfemBlockOperator> A_;

    /** \brief The collection of inverses for diagonal blocks. */
    std::vector<std::shared_ptr<mfem::Solver>> inv_ops_;

    /** \brief The collection of other operators that this keeps alive. */
    std::vector<std::shared_ptr<mfem::Operator>> aux_ops_;

};// class BlockDiagonalSolver
}// namespace parelag
#endif /* PARELAG_BLOCKDIAGONALSOLVER_HPP_ */
