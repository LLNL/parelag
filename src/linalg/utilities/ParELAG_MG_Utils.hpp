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


#ifndef PARELAG_MG_UTILS_HPP_
#define PARELAG_MG_UTILS_HPP_

#include <mfem.hpp>

#include "hypreExtension/hypreExtension.hpp"
#include "utilities/elagError.hpp"
#include "utilities/MemoryUtils.hpp"

#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"
#include "linalg/solver_core/ParELAG_Solver.hpp"

namespace parelag
{
/** \namespace mg_utils
 *  \brief A collection of utilities specific to functionality for the
 *         MG solvers code.
 *
 *  This namespace contains routines for finding communicators in
 *  matrix objects, unifying the interface between the raw
 *  mfem::BlockOperator class and parelag::MfemBlockOperator, unifying
 *  the interface between mfem::SparseMatrix and mfem::HypreParMatrix,
 *  and computing residuals.
 */
namespace mg_utils
{

/** \brief Check if the solver is symmetric. */
bool CheckSymmetric(const Solver& solver, double tol = 1e-10);

/** \brief Get a communicator from a given operator.
 *
 *  \warning This is basically an if-elseif on dynamic_casts.
 */
MPI_Comm GetComm(mfem::Operator const& op);

/** \brief Get a communicator from a given HypreParMatrix. */
MPI_Comm GetComm(mfem::HypreParMatrix const& op);

/** \brief Get a communicator from a given HypreParMatrix.
 *
 *  \return MPI_COMM_SELF
 */
MPI_Comm GetComm(mfem::SparseMatrix const& op);

/** \brief Get a communicator from a given MfemBlockOperator. */
MPI_Comm GetComm(MfemBlockOperator const& op);


/** \class MatrixMatrixHelper
 *  \brief Facilitate matrix-matrix operations for two matrices of the same
 *         matrix type.
 *
 *  \tparam MatrixType The type of the matrices being multiplied.
 *                     Specialized for mfem::SparseMatrix and
 *                     mfem::HypreParMatrix.
 */
template <typename MatrixType>
struct MatrixMatrixHelper
{
    /** \brief Compute the matrix-matrix product of the input matrices.
     *
     *  \param A The left-hand matrix.
     *  \param B The right-hand matrix.
     *
     *  \return The product \f$AB\f$.
     */
    static std::unique_ptr<MatrixType> Multiply(MatrixType* A, MatrixType* B);

    /** \brief Compute the matrix-matrix sum of the input matrices.
     *
     *  \param A The left-hand matrix.
     *  \param B The right-hand matrix.
     *
     *  \return The sum \f$A+B\f$.
     */
    static std::unique_ptr<MatrixType> Add(MatrixType* A, MatrixType* B);
};// struct MatrixMatrixHelper

/** \class MatrixMatrixHelper<mfem::SparseMatrix>
 *  \brief Specialization of MatrixMatrixHelper for mfem::SparseMatrix.
 */
template <>
struct MatrixMatrixHelper<mfem::SparseMatrix>
{

    /** \brief Compute the matrix-matrix product of the input matrices.
     *
     *  \param A The left-hand matrix.
     *  \param B The right-hand matrix.
     *
     *  \return The product \f$AB\f$.
     */
    static std::unique_ptr<mfem::SparseMatrix>
    Multiply(mfem::SparseMatrix* A, mfem::SparseMatrix* B)
    { return ToUnique(mfem::Mult(*A,*B)); }

    /** \brief Compute the matrix-matrix sum of the input matrices.
     *
     *  \param A The left-hand matrix.
     *  \param B The right-hand matrix.
     *
     *  \return The sum \f$A+B\f$.
     */
    static std::unique_ptr<mfem::SparseMatrix>
    Add(mfem::SparseMatrix* A, mfem::SparseMatrix* B)
    { return ToUnique(mfem::Add(*A,*B)); }

};// struct MatrixMatrixHelper<mfem::SparseMatrix>

/** \class MatrixMatrixHelper<mfem::HypreParMatrix>
 *  \brief Specialization of MatrixMatrixHelper for mfem::HypreParMatrix.
 */
template <>
struct MatrixMatrixHelper<mfem::HypreParMatrix>
{

    /** \brief Compute the matrix-matrix product of the input matrices.
     *
     *  \param A The left-hand matrix.
     *  \param B The right-hand matrix.
     *
     *  \return The product \f$AB\f$.
     */
    static std::unique_ptr<mfem::HypreParMatrix>
    Multiply(mfem::HypreParMatrix* A, mfem::HypreParMatrix* B)
    { return ToUnique(mfem::ParMult(A,B)); }

    /** \brief Compute the matrix-matrix sum of the input matrices.
     *
     *  \param A The left-hand matrix.
     *  \param B The right-hand matrix.
     *
     *  \return The sum \f$A+B\f$.
     */
    static std::unique_ptr<mfem::HypreParMatrix>
    Add(mfem::HypreParMatrix* A, mfem::HypreParMatrix* B)
    {
        hypre_ParCSRMatrix * A_hyp = *A, * B_hyp = *B, * C_hyp;
        auto ierr = parelag_ParCSRMatrixAdd(A_hyp,B_hyp,&C_hyp);

        PARELAG_ASSERT_HYPRE_ERROR_FLAG(ierr);

        // TODO: CHECK OWNERSHIP AND DO APPROPRIATE COPIES

        return make_unique<mfem::HypreParMatrix>(C_hyp);
    }

};// struct MatrixMatrixHelper<mfem::HypreParMatrix>


/** \class BlockOpHelper
 *  \brief Provide uniform interface betwixt mfem::BlockOperator and
 *  parelag::MfemBlockOperator
 *
 *  This mostly exists because I had reasons for making the interfaces
 *  differ, but I don't want to write them down. Feel free to ask me
 *  in person.
 *
 *  \tparam BlockOperatorType The type of blocked operator being
 *              used. Specialized for mfem::BlocKOperator and
 *              parelag::MfemBlockOperator.
 */
template <typename BlockOperatorType>
struct BlockOpHelper
{
    /** \brief Type used for expressing sizes. */
    using size_type = MfemBlockOperator::size_type;

    /** \brief Type used for offset counters. */
    using offset_type = MfemBlockOperator::offset_type;

    /** \brief Get the number of block rows the operator has. */
    static size_type GetNumBlockRows(const BlockOperatorType&) noexcept;

    /** \brief Get the number of block columns the operator has. */
    static size_type GetNumBlockCols(const BlockOperatorType&) noexcept;

    /** \brief Return a clean copy of the row offsets. */
    static std::vector<offset_type>
    CopyRowOffsets(const BlockOperatorType&) noexcept;

    /** \brief Return a clean copy of the column offsets. */
    static std::vector<offset_type>
    CopyColumnOffsets(const BlockOperatorType&) noexcept;

    /** \brief Get a (raw) pointer to a block of the underlying
     *         operator.
     *
     *  \tparam MatrixType The assumed underlying type of a block in
     *              the operator.
     *
     *  \return A pointer to a block of the BlockOperator casted to
     *          MatrixType. If the block exists but is not of type
     *          MatrixType, an exception is thrown. If the block is
     *          null, a nullptr is returned.
     */
    template <typename MatrixType>
    static MatrixType* GetBlockPtr(
        const BlockOperatorType&,
        MfemBlockOperator::size_type, MfemBlockOperator::size_type);

};// struct BlockOpHelper


/** \class BlockOpHelper<MfemBlockOperator>
 *  \brief Template specialization of BlockOpHelper for
 *         parelag::MfemBlockOperator.
 */
template <>
struct BlockOpHelper<MfemBlockOperator>
{
    /** \brief Type used for expressing sizes. */
    using size_type = MfemBlockOperator::size_type;

    /** \brief Type used for offset counters. */
    using offset_type = MfemBlockOperator::offset_type;

    /** \brief Get the number of block rows the operator has */
    static size_type GetNumBlockRows(const MfemBlockOperator& A) noexcept
    { return A.GetNumBlockRows(); }

    /** \brief Get the number of block columns the operator has */
    static size_type GetNumBlockCols(const MfemBlockOperator& A) noexcept
    { return A.GetNumBlockCols(); }

    /** \brief Return a clean copy of the row offsets */
    static std::vector<offset_type>
    CopyRowOffsets(const MfemBlockOperator& A) noexcept
    { return A.CopyRowOffsets(); }

    /** \brief Return a clean copy of the column offsets */
    static std::vector<offset_type>
    CopyColumnOffsets(const MfemBlockOperator& A) noexcept
    { return A.CopyColumnOffsets(); }

    /** \brief Get a (raw) pointer to a block of the underlying
     *         operator.
     *
     *  \tparam MatrixType The assumed underlying type of a block in
     *              the operator.
     *
     *  \param A The blocked operator.
     *  \param blk_i The block-row index of the desired block.
     *  \param blk_j The block-column index of the desired block.
     *
     *  \return A pointer to a block of the BlockOperator casted to
     *          MatrixType. If the block exists but is not of type
     *          MatrixType, an exception is thrown. If the block is
     *          null, a nullptr is returned.
     */
    template <typename MatrixType>
    static MatrixType* GetBlockPtr(
        const MfemBlockOperator& A, size_type blk_i, size_type blk_j)
    {
        auto op_ptr = A.GetBlockPtr(blk_i,blk_j);

        // No block pointer available; return null
        if (!op_ptr)
            return nullptr;

        auto mat_ptr = dynamic_cast<MatrixType*>(op_ptr.get());

        PARELAG_TEST_FOR_EXCEPTION(
            !mat_ptr,
            std::runtime_error,
            "BlockOpHelper::GetBlockPtr<MatrixType>(...): "
            "Operator (" << blk_i << "," << blk_j << ") "
            "is not the specified type!");

        return mat_ptr;
    }
};// struct BlockOpHelper<MfemBlockOperator>


/** \class BlockOpHelper<mfem::BlockOperator>
 *  \brief Template specialization of BlockOpHelper for
 *         mfem::BlockOperator.
 */
template <>
struct BlockOpHelper<mfem::BlockOperator>
{
    using size_type = MfemBlockOperator::size_type;
    using offset_type = MfemBlockOperator::offset_type;

    /** \brief Get the number of block rows the operator has */
    static size_type GetNumBlockRows(const mfem::BlockOperator& A) noexcept
    { return A.NumRowBlocks(); }

    /** \brief Get the number of block columns the operator has */
    static size_type GetNumBlockCols(const mfem::BlockOperator& A) noexcept
    { return A.NumColBlocks(); }

    /** \brief Return a clean copy of the row offsets */
    static std::vector<offset_type>
    CopyRowOffsets(const mfem::BlockOperator& A) noexcept
    {
        auto& Aref = const_cast<mfem::BlockOperator&>(A);
        return std::vector<offset_type>(
            Aref.RowOffsets().GetData(),
            Aref.RowOffsets().GetData()+Aref.RowOffsets().Size());
    }

    /** \brief Return a clean copy of the column offsets */
    static std::vector<offset_type>
    CopyColumnOffsets(const mfem::BlockOperator& A) noexcept
    {
        auto& Aref = const_cast<mfem::BlockOperator&>(A);
        return std::vector<offset_type>(
            Aref.ColOffsets().GetData(),
            Aref.ColOffsets().GetData()+Aref.ColOffsets().Size());
    }

    /** \brief Get a (raw) pointer to a block of the underlying
     *         operator.
     *
     *  \tparam MatrixType The assumed underlying type of a block in
     *              the operator.
     *
     *  \param A The blocked operator.
     *  \param blk_i The block-row index of the desired block.
     *  \param blk_j The block-column index of the desired block.
     *
     *  \return A pointer to a block of the BlockOperator casted to
     *          MatrixType. If the block is null, a nullptr is
     *          returned.
     *
     *  \throws std::runtime_error If the block exists but is not of
     *          type MatrixType.
     */
    template <typename MatrixType>
    static MatrixType* GetBlockPtr(
        const mfem::BlockOperator& A, size_type blk_i, size_type blk_j)
    {
        auto& Aref = const_cast<mfem::BlockOperator&>(A);

        if (A.IsZeroBlock(blk_i,blk_j))
            return nullptr;

        auto& op_ref = Aref.GetBlock(blk_i,blk_j);

        auto mat_ptr = dynamic_cast<MatrixType*>(&op_ref);

        PARELAG_TEST_FOR_EXCEPTION(
            !mat_ptr,
            std::runtime_error,
            "BlockOpHelper<mfem::BlockOperator>::GetBlockPtr<MatrixType>(...): "
            "Operator (" << blk_i << "," << blk_j << ") "
            "is not the specified type!");

        return mat_ptr;
    }
};// struct BlockOpHelper<mfem::BlockOperator>


/** \brief Compute an explicit blocked operator-operator product,
 *         where the operators have underlying type MatrixType.
 *
 *  This computes the product \f$\text{out} = AB\f$, where \f$A\f$ and
 *  \f$B\f$ are blocked operators. The operator \f$A\f$ is hereafter
 *  referred to as the "left" operator and \f$b\f$ as the "right
 *  operator.
 *
 *  \tparam MatrixType The assumed type of the underlying blocks.
 *  \tparam BlockOperatorTypeA The deduced type of the left operator.
 *  \tparam BlockOperatorTypeB The deduced type of the right operator.
 *
 *  \param A The left operator in the product.
 *  \param B The right operator in the product.
 *
 *  \return The operator representing the explicit operator-operator
 *          product of the input operators.
 */
template <typename MatrixType, typename BlockOperatorTypeA,
          typename BlockOperatorTypeB>
std::unique_ptr<MfemBlockOperator> BlockedMatrixMult(
    const BlockOperatorTypeA& A, const BlockOperatorTypeB& B);


/** \brief Compute a residual given the operator, RHS, and iterate.
 *
 *  Computes \f$r = B - \text{op}(X)\f$ for a given system.
 *
 *  \param Op The system operator.
 *  \param X The current solution vector.
 *  \param B The system right-hand side vector.
 *  \param transpose If \c true, use \f$\text{Op}^T\f$ instead.
 *
 *  \return A newly allocated vector containing the residual data.
 */
inline std::unique_ptr<mfem::Vector> ComputeResidual(
    const mfem::Operator& op, const mfem::Vector& X, const mfem::Vector& B,
    bool transpose = false)
{
    // Create a temporary vector in the range of op
    auto tmp = make_unique<mfem::Vector>(B.Size());
    *tmp = 0.;

    // tmp = op*x or tmp = op^T*x
    if (transpose)
        op.MultTranspose(X,*tmp);
    else
        op.Mult(X,*tmp);

    // tmp = A*X - B
    *tmp -= B;

    // tmp = B - A*X
    *tmp *= -1;

    return tmp;
}

/** \brief Compute the residual of a blocked system.
 *
 *  Computes \f$r = B - \text{op}(X)\f$ for a given blocked system.
 *
 *  \param Op The system operator.
 *  \param X The current solution vector.
 *  \param B The system right-hand side vector.
 *  \param transpose If \c true, use \f$\text{Op}^T\f$ instead.
 *
 *  \return A new BlockVector containing the residual.
 *
 *  \warning The output BlockVector is not guaranteed to be valid
 *           beyond the lifetime of "op" as BlockVector does not
 *           (and, in fact, cannot) own its offsets.
 */
inline std::unique_ptr<mfem::BlockVector> ComputeResidual(
    MfemBlockOperator& op,
    const mfem::BlockVector& X, const mfem::BlockVector& B,
    bool transpose = false)
{
    // Create a temporary vector in the range of op
    int * offsets = const_cast<int*>(op.ViewRowOffsets().data());
    auto tmp = make_unique<mfem::BlockVector>(
        mfem::Array<int>(offsets, op.ViewRowOffsets().size()));
    *tmp = 0.;

    // tmp = op*x or tmp = op^T*x
    if (transpose)
        op.MultTranspose(X,*tmp);
    else
        op.Mult(X,*tmp);

    // tmp = A*X - B
    *tmp -= B;

    // tmp = B - A*X
    *tmp *= -1;

    return tmp;
}

}// namespace mg_utils
}// namespace parelag
#endif /* PARELAG_MG_UTILS_HPP_ */
