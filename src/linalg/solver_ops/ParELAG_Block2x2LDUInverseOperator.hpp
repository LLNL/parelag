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


#ifndef PARELAG_BLOCK2X2LUINVERSEOPERATOR_HPP_
#define PARELAG_BLOCK2X2LUINVERSEOPERATOR_HPP_

#include "utilities/MemoryUtils.hpp"
#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"
#include "linalg/solver_core/ParELAG_Solver.hpp"

namespace parelag
{

/** \class Block2x2LDUInverseOperator
 *  \brief Applies a Block2x2 block-factorization inverse of a blocked
 *         operator.
 *
 *  This class approximates the action of the inverse of the operator
 *
 *  \f[\begin{bmatrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{bmatrix}\f]
 *
 *  using the following factorization of the matrix:
 *
 *  \f[
 *     \begin{bmatrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{bmatrix} =
 *     \begin{bmatrix} I & 0 \\ A_{10}A_{00}^{-1} & I \end{bmatrix}
 *     \begin{bmatrix} A_{00} & 0 \\ 0 & S \end{bmatrix}
 *     \begin{bmatrix} I & A_{00}^{-1}A_{01} \\ 0 & I \end{bmatrix}
 *  \f]
 *
 *  where \f$S\f$ is the Schur complement or some matrix to be treated
 *  equivalently. The Schur complement is:
 *
 *  \f[
 *     S = A_{11} - A_{10}*A_{00}^{-1}*A_{01},
 *  \f]
 *
 *  but some approximation may be used (see SchurComplementFactory).
 *
 *  This requires the specification of 4 inverse operations: 3
 *  applications of \f$A_{00}^{-1}\f$ and 1 application of
 *  \f$S^{-1}\f$.
 *
 *  The method is applied as follows. Given all inverses and a vector
 *  \f$(f, g)^{T}\f$, the vector \f$(u, p)^{T} = M (f, g)^{T}\f$ is
 *  computed as:
 *
 *    1. \f$\hat{u} = \hat{A}_{00}^{-1}f\f$
 *    2. \f$p = S^{-1} (g - A_{10}\bar{A}_{00}^{-1}f)\f$
 *    3. \f$u = \hat{u} - \tilde{A}_{00}^{-1}A_{01}p\f$
 *
 *  There are shortcuts possible if \f$\hat{A}_{00}^{-1} = \bar{A}_{00}^{-1}\f$ or if
 *  \f$\hat{A}_{00}^{-1} = \tilde{A}_{00}^{-1}\f$. These save 1 inverse apply.
 *
 *  If \f$\hat{A}_{00}^{-1} = \bar{A}_{00}^{-1}\f$:
 *
 *    1. \f$\hat{u} = \hat{A}_{00}^{-1}f\f$
 *    2. \f$p = S^{-1} (g - A_{10}\hat{u})\f$
 *    3. \f$u = \hat{u} - \tilde{A}_{00}^{-1}A_{01}p\f$
 *
 *  If \f$\hat{A}_{00}^{-1} = \tilde{A}_{00}^{-1}\f$:
 *
 *    1. \f$p = S^{-1} (g - A_{10}\bar{A}_{00}^{-1}f)\f$
 *    2. \f$u = \hat{A}_{00}^{-1}(f - A_{01}p)\f$
 *
 *  It is not possible to use fewer than 2 applications of
 *  \f$A_{00}^{-1}\f$.
 *
 *  The operator, A, upon which this is built *must* be block-2x2. Of
 *  course, these blocks may themselves be BlockOperators of arbitrary
 *  (but compatible) block-dimension.
 *
 *  Block2x2LDUSolverFactory gives an example of creating this
 *  operator.
 *
 *  \todo Implement the aforementioned shortcuts if applicable.
 */
class Block2x2LDUInverseOperator : public Solver
{
public:
    /** \name Constructors and destructor */
    ///@{

    /** \brief Constructor.
     *
     *  \param A The blocked operator that has been factored.
     *  \param invA00_1 The inverse for the first application of
     *                  \f$A_{00}^{-1}\f$ (i.e., \f$\hat{A}_{00}^{-1}\f$).
     *  \param invA00_2 The inverse for the second application of
     *                  \f$A_{00}^{-1}\f$ (i.e., \f$\bar{A}_{00}^{-1}\f$).
     *  \param invA00_3 The inverse for the third application of
     *                  \f$A_{00}^{-1}\f$ (i.e., \f$\tilde{A}_{00}^{-1}\f$).
     *  \param invS The inverse for \f$S\f$.
     *  \param S The (approximate) Schur complement operator.
     *  \param DampingFactor The damping weight on the residual
     *                       update.
     */
    Block2x2LDUInverseOperator(
        std::shared_ptr<mfem::Operator> const& A,
        std::shared_ptr<mfem::Solver> invA00_1,
        std::shared_ptr<mfem::Solver> invA00_2,
        std::shared_ptr<mfem::Solver> invA00_3,
        std::shared_ptr<mfem::Solver> invS,
        std::shared_ptr<mfem::Operator> S,
        double DampingFactor = 1.0);

    /** \brief Destructor. */
    ~Block2x2LDUInverseOperator() = default;

    /** \brief Default constructor -- deleted. */
    Block2x2LDUInverseOperator() = delete;

    ///@}
    /** \name mfem::Operator interface */
    ///@{

    /** \brief Apply operator to a vector. */
    void Mult(const mfem::Vector& rhs, mfem::Vector& sol) const override;

    /** \brief Apply transpose of operator to a vector.
     *
     *  \warning Each of the operators given to the constructor of
     *  this operator must have MultTranspose() defined or this will
     *  fail at runtime.
     */
    void MultTranspose(
        const mfem::Vector& rhs, mfem::Vector& sol) const override;

    ///@}

private:

    /** \brief Implemenation of SetOperator().
     *
     *  New operator must be of the same dimension, and it must be a
     *  BlockOperator.
     */
    void _do_set_operator(const std::shared_ptr<mfem::Operator>& A) override;

private:

    /** \brief The blocked operator that has been factored. */
    std::shared_ptr<MfemBlockOperator> A_;

    /** \brief The inverse for the first application of
     *         \f$A_{00}^{-1}\f$ (i.e., \f$\hat{A}_{00}^{-1}\f$).*/
    std::shared_ptr<mfem::Solver> invA00_1_;

    /** \brief The inverse for the second application of
     *         \f$A_{00}^{-1}\f$ (i.e., \f$\bar{A}_{00}^{-1}\f$).*/
    std::shared_ptr<mfem::Solver> invA00_2_;

    /** \brief The inverse for the third application of
     *         \f$A_{00}^{-1}\f$ (i.e., \f$\tilde{A}_{00}^{-1}\f$). */
    std::shared_ptr<mfem::Solver> invA00_3_;

    /** \brief The inverse for \f$S\f$. */
    std::shared_ptr<mfem::Solver> invS_;

    /** \brief The (approximate) Schur complement operator. */
    std::shared_ptr<mfem::Operator> S_;

    /** \brief The damping weight on the residual update. */
    double DampingFactor_;

    ///@{
    /** \brief Helper vectors. */
    mutable std::unique_ptr<mfem::BlockVector> Residual_, Correction_, Tmp_;
    ///@}
};// class Block2x2LDUInverseOperator
}// namespace parelag
#endif /* PARELAG_BLOCK2X2LDUINVERSEOPERATOR_HPP_ */
