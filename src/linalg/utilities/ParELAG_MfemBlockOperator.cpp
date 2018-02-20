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


#include "ParELAG_MfemBlockOperator.hpp"

#include "utilities/elagError.hpp"
#include "utilities/ParELAG_Meta.hpp"

namespace parelag
{

MfemBlockOperator::MfemBlockOperator(std::vector<offset_type> block_offsets)
    : mfem::Operator{block_offsets.back()},
      NumBlockRows_{block_offsets.size() ? block_offsets.size()-1 : 0},
      NumBlockCols_{NumBlockRows_},
      RowOffsets_{std::move(block_offsets)},
      ColOffsets_{RowOffsets_},
      Ops_{NumBlockRows_*NumBlockCols_},
      A_{mfem::Array<offset_type>{
              RowOffsets_.data(),narrowing_cast<int>(RowOffsets_.size())}}
{
    A_.owns_blocks = false;
}


MfemBlockOperator::MfemBlockOperator(
    const mfem::Array<offset_type>& block_offsets)
    : mfem::Operator{block_offsets.Last()},
    NumBlockRows_{narrowing_cast<size_type>(
            block_offsets.Size() ? block_offsets.Size()-1 : 0)},
      NumBlockCols_{NumBlockRows_},
      RowOffsets_{block_offsets.GetData(),
                  block_offsets.GetData()+block_offsets.Size()},
      ColOffsets_{RowOffsets_},
      Ops_{NumBlockRows_*NumBlockCols_},
      A_{mfem::Array<offset_type>{
              RowOffsets_.data(),narrowing_cast<int>(RowOffsets_.size())}}
{
    A_.owns_blocks = false;
}


MfemBlockOperator::MfemBlockOperator(std::vector<offset_type> row_offsets,
                                     std::vector<offset_type> col_offsets)
    : mfem::Operator{row_offsets.back(),col_offsets.back()},
      NumBlockRows_{row_offsets.size() ? row_offsets.size()-1 : 0},
      NumBlockCols_{col_offsets.size() ? col_offsets.size()-1 : 0},
      RowOffsets_{std::move(row_offsets)},
      ColOffsets_{std::move(col_offsets)},
      Ops_{NumBlockRows_*NumBlockCols_},
      A_{mfem::Array<offset_type>{
             RowOffsets_.data(),narrowing_cast<int>(RowOffsets_.size())},
         mfem::Array<offset_type>{
             ColOffsets_.data(),narrowing_cast<int>(ColOffsets_.size())}}
{
    A_.owns_blocks = false;
}


MfemBlockOperator::MfemBlockOperator(
    const mfem::Array<offset_type>& row_offsets,
    const mfem::Array<offset_type>& col_offsets)
    : mfem::Operator{row_offsets.Last(),col_offsets.Last()},
      NumBlockRows_{narrowing_cast<size_type>(
              row_offsets.Size() ? row_offsets.Size()-1 : 0)},
      NumBlockCols_{narrowing_cast<size_type>(
              col_offsets.Size() ? col_offsets.Size()-1 : 0)},
      RowOffsets_{row_offsets.GetData(),
                  row_offsets.GetData()+row_offsets.Size()},
      ColOffsets_{col_offsets.GetData(),
                  col_offsets.GetData()+col_offsets.Size()},
      Ops_{NumBlockRows_*NumBlockCols_},
      A_{mfem::Array<offset_type>{
             RowOffsets_.data(),narrowing_cast<int>(RowOffsets_.size())},
         mfem::Array<offset_type>{
             ColOffsets_.data(),narrowing_cast<int>(ColOffsets_.size())}}
{
    A_.owns_blocks = false;
}


void MfemBlockOperator::Mult(const mfem::Vector& x, mfem::Vector& b) const
{
    A_.Mult(x,b);
}


void MfemBlockOperator::MultTranspose(const mfem::Vector& x,
                                      mfem::Vector& b) const
{
    A_.MultTranspose(x,b);
}


void MfemBlockOperator::SetBlock(size_type block_row, size_type block_col,
                                 std::shared_ptr<mfem::Operator> op)
{
#ifdef ELAG_DEBUG
    PARELAG_TEST_FOR_EXCEPTION(
        (op->Width() != (ColOffsets_[block_col+1] - ColOffsets_[block_col])) ||
        (op->Height() != (RowOffsets_[block_row+1] - RowOffsets_[block_row])),
        std::runtime_error,
        "MfemBlockOperator::SetBlock(...):\n"
        "Input operator has wrong size (size = " << op->Height() << "x" <<
        op->Width() << "; should be " <<
        (RowOffsets_[block_row+1] - RowOffsets_[block_row]) << "x" <<
        (ColOffsets_[block_col+1] - ColOffsets_[block_col]) << ").");
#endif

    PARELAG_ASSERT(block_row < NumBlockRows_);
    PARELAG_ASSERT(block_col < NumBlockCols_);

    A_.SetBlock(block_row,block_col,op.get());
    op.swap(Ops_[block_row*NumBlockCols_+block_col]);
}


mfem::Operator& MfemBlockOperator::GetBlock(
    size_type block_row, size_type block_col)
{
    return A_.GetBlock(block_row,block_col);
}


const mfem::Operator& MfemBlockOperator::GetBlock(
    size_type block_row, size_type block_col) const
{
    return const_cast<mfem::BlockOperator&>(A_).GetBlock(block_row,block_col);
}


std::shared_ptr<mfem::Operator> MfemBlockOperator::GetBlockPtr(
    size_type block_row, size_type block_col) const noexcept
{
    return Ops_[block_col+block_row*NumBlockCols_];
}


void MfemBlockOperator::CopyRowOffsetsAsMfemArray(
    mfem::Array<offset_type>& row_offsets) const
{
    row_offsets.SetSize(RowOffsets_.size());
    std::copy(RowOffsets_.begin(),RowOffsets_.end(),row_offsets.GetData());
}


void MfemBlockOperator::CopyColumnOffsetsAsMfemArray(
    mfem::Array<offset_type>& col_offsets) const
{
    auto& offsets = (ColOffsets_.size() ? ColOffsets_ : RowOffsets_);

    col_offsets.SetSize(offsets.size());
    std::copy(offsets.begin(),offsets.end(),col_offsets.GetData());
}

}// namespace parelag
