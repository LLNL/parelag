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


#ifndef PARELAG_MG_UTILS_IMPL_HPP_
#define PARELAG_MG_UTILS_IMPL_HPP_

#include "ParELAG_MG_Utils.hpp"

namespace parelag
{
namespace mg_utils
{

// Compute a blocked operator-operator product, where the operators
// have underlying type MatrixType
template <typename MatrixType, typename BlockOperatorTypeA,
          typename BlockOperatorTypeB>
std::unique_ptr<MfemBlockOperator> BlockedMatrixMult(
    const BlockOperatorTypeA& A, const BlockOperatorTypeB& B)
{
    using idx_type = MfemBlockOperator::size_type;

    using helperA = BlockOpHelper<BlockOperatorTypeA>;
    using helperB = BlockOpHelper<BlockOperatorTypeB>;
    using mm_helper = MatrixMatrixHelper<MatrixType>;

    const auto num_block_rows_A = helperA::GetNumBlockRows(A),
        num_block_cols_A = helperA::GetNumBlockCols(A),
        num_block_rows_B = helperB::GetNumBlockRows(B);

    PARELAG_ASSERT(num_block_cols_A == num_block_rows_B);

    // The result will have the row offsets of A and the column offsets of B
    auto out = make_unique<MfemBlockOperator>(
        helperA::CopyRowOffsets(A),helperB::CopyColumnOffsets(B));

    // Fill the blocks. Unfortunately, this process takes a lot of
    // dynamic_casting, but c'est la vie.
    //
    // If a block in either operator is null, it is treated as a ZERO
    // block and the result of the multiplication for that block is a
    // null (i.e. ZERO) block.
    std::unique_ptr<MatrixType> tmp = nullptr;
    for (auto blk_i = idx_type{0}; blk_i < num_block_rows_A; ++blk_i)
    {
        for (auto blk_j = idx_type{0}; blk_j < num_block_cols_A; ++blk_j)
        {
            // Reset the tmp ptr to null, just in case
            tmp = nullptr;
            for (auto blk_k = idx_type{0}; blk_k < num_block_cols_A; ++blk_k)
            {
                // I could do a thing with references here, but that
                // requires me to check validity.
                auto Aik = helperA::template GetBlockPtr<MatrixType>(
                    A,blk_i,blk_k);
                auto Bkj = helperB::template GetBlockPtr<MatrixType>(
                    B,blk_k,blk_j);

                if (!Aik || !Bkj)
                    continue;

                auto this_contrib = mm_helper::Multiply(Aik,Bkj);

                if (tmp)
                    tmp = mm_helper::Add(tmp.get(), this_contrib.get());
                else
                    tmp = std::move(this_contrib);
            }

            // Set the block if it's not null
            if (tmp)
                out->SetBlock(blk_i,blk_j,std::move(tmp));
        }
    }

    return out;
}

}// namespace mg_utils
}// namespace parelag

#endif /* PARELAG_MG_UTILS_IMPL_HPP_ */
