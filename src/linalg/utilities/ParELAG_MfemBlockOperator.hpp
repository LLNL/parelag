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


#ifndef PARELAG_MFEMBLOCKOPERATOR_HPP_
#define PARELAG_MFEMBLOCKOPERATOR_HPP_

#include <vector>

#include "mfem.hpp"

#include "utilities/MemoryUtils.hpp"

namespace parelag
{

/** \class MfemBlockOperator
 *  \brief Memory-managing wrapper for mfem::BlockOperator.
 *
 *  This allows for preservation of operators through shared_ptrs
 *  instead of the rather the rather ad hoc flag-based approach in the
 *  mfem::BlockOperator.
 *
 *  Zero-blocks are stored as \c nullptrs.
 */
class MfemBlockOperator : public mfem::Operator
{
public:

    /** \brief The type used to store offsets. */
    using offset_type = int;

    /** \brief The type used to store sizes. */
    using size_type = typename std::vector<offset_type>::size_type;

public:

    /** \name Constructors and destructor. */
    ///@{

    /** \brief Create a square BlockOperator from given offsets.
     *
     *  \param block_offsets The block offset in "CSR"-like format:
     *                       [0, blk0_max, blk1_max,...,total_num_rows/cols].
     */
    MfemBlockOperator(std::vector<offset_type> block_offsets);

    /** \brief Create a square BlockOperator from given offsets (mfem
     *         version).
     *
     *  \param block_offsets The block offset in "CSR"-like format:
     *                       [0, blk0_max, blk1_max,...,total_num_rows/cols].
     */
    MfemBlockOperator(mfem::Array<offset_type> const& block_offsets);

    /** \brief Create a generally rectangular BlockOperator from given
     *         row and column offsets.
     *
     *  \param row_offsets The block row offsets in "CSR"-like format:
     *                     [0, blk0_max, blk1_max,...,total_num_rows].
     *  \param col_offsets The block column offsets in "CSR"-like format:
     *                     [0, blk0_max, blk1_max,...,total_num_cols].
     */
    MfemBlockOperator(std::vector<offset_type> row_offsets,
                      std::vector<offset_type> col_offsets);

    /** \brief Create a generally rectangular BlockOperator from given
     *         row and column offsets (mfem version).
     *
     *  \param row_offsets The block row offsets in "CSR"-like format:
     *                     [0, blk0_max, blk1_max,...,total_num_rows].
     *  \param col_offsets The block column offsets in "CSR"-like format:
     *                     [0, blk0_max, blk1_max,...,total_num_cols].
     */
    MfemBlockOperator(mfem::Array<offset_type> const& row_offsets,
                      mfem::Array<offset_type> const& col_offsets);

    /** \brief Destructor. */
    ~MfemBlockOperator() = default;

    ///@}
    /** \name mfem::Operator interface */
    ///@{

    /** \brief Apply the operator to a vector. */
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /** \brief Apply the operator's transpose to a vector. */
    void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

    ///@}
    /** \name Get/Set methods */
    ///@{

    /** \brief Return whether a given block is zero (i.e. nullptr)
     *
     *  \param block_row The block row index.
     *  \param block_col The block column index.
     *
     *  \return \c true if the block is null.
     */
    bool IsZeroBlock(size_type block_row, size_type block_col) const
    { return A_.IsZeroBlock(block_row,block_col); }

    /** \brief Set a block of the operator.
     *
     *  \param block_row The block row index.
     *  \param block_col The block column index.
     *  \param op The operator to add to the blocked operator.
     */
    void SetBlock(size_type block_row, size_type block_col,
                  std::shared_ptr<mfem::Operator> op);


    /** \brief Get a member block of the operator by reference.
     *
     *  \param block_row The block row index.
     *  \param block_col The block column index.
     *
     *  \return Reference to the requested block.
     */
    mfem::Operator& GetBlock(size_type block_row, size_type block_col);

    /** \brief Get a member block of the operator by reference (const
     *         version).
     *
     *  \param block_row The block row index.
     *  \param block_col The block column index.
     *
     *  \return const reference to the requested block.
     */
    const mfem::Operator& GetBlock(
        size_type block_row, size_type block_col) const;

    /** \brief Get a member block of the operator by shared_ptr.
     *
     *  \param block_row The block row index.
     *  \param block_col The block column index.
     *
     *  \return Shared_ptr to the requested block.
     */
    std::shared_ptr<mfem::Operator> GetBlockPtr(
        size_type block_row, size_type block_col) const noexcept;

    /** \brief Get the number of block rows.
     *
     *  \return The number of block rows in this operator.
     */
    size_type GetNumBlockRows() const noexcept { return NumBlockRows_; }

    /** \brief Get the number of block columns.
     *
     *  \return The number of block columns in this operator.
     */
    size_type GetNumBlockCols() const noexcept { return NumBlockCols_; }

    /** \brief Get a view of the row offsets.
     *
     *  \return const reference to the row offests.
     */
    const std::vector<offset_type>& ViewRowOffsets() const noexcept
    { return RowOffsets_; }

    /** \brief Get a view of the column offsets.
     *
     *  \return const reference to the column offests.
     */
    const std::vector<offset_type>& ViewColumnOffsets() const noexcept
    { return (ColOffsets_.size() ? ColOffsets_ : RowOffsets_); }

    /** \brief Get a clean copy of the row offests.
     *
     *  \return A vector containing the row offsets.
     */
    std::vector<offset_type> CopyRowOffsets() const { return RowOffsets_; }

    /** \brief Get a clean copy of the column offests.
     *
     *  \return A vector containing the column offsets.
     */
    std::vector<offset_type> CopyColumnOffsets() const
    { return (ColOffsets_.size() ? ColOffsets_ : RowOffsets_); }

    /** \brief Copy the row offsets into the provided array.
     *
     *  \param[out] row_offsets Array into which the offsets will be copied.
     */
    void CopyRowOffsetsAsMfemArray(
        mfem::Array<offset_type>& row_offsets) const;

    /** \brief Copy the column offsets into the provided array.
     *
     *  \param[out] col_offsets Array into which the offsets will be copied.
     */
    void CopyColumnOffsetsAsMfemArray(
        mfem::Array<offset_type>& col_offsets) const;

    ///@}

private:

    /** \brief The number of block rows in this operator. */
    size_type NumBlockRows_;

    /** \brief The number of block columns in this operator. */
    size_type NumBlockCols_;

    /** \brief The row offsets. */
    std::vector<offset_type> RowOffsets_;

    /** \brief The column offsets. */
    std::vector<offset_type> ColOffsets_;

    /** \brief The memory-managed operators, stored block-row-major. */
    std::vector<std::shared_ptr<mfem::Operator>> Ops_;

    /** \brief The underlying BlockOperator object. */
    mfem::BlockOperator A_;

};// class MfemBlockOperator
}// namespace parelag
#endif /* PARELAG_MFEMBLOCKOPERATOR_HPP_ */
