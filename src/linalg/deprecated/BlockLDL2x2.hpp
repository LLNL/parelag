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

#ifndef BLOCKLDL2X2_HPP_
#define BLOCKLDL2X2_HPP_

#include <mfem.hpp>

namespace parelag
{
class BlockLDL2x2 : public mfem::Solver
{
public:
    BlockLDL2x2(const mfem::Array<int> & offsets);

    //! Add a square block op in the block-entry (iblock, iblock).
    /**
     * iblock: The block will be inserted in location (iblock, iblock).
     * op: the Operator to be inserted.
     */
    void SetDiagonalBlock(int iblock, mfem::Operator *op);
    //! Add a block op in the block-entry (iblock, jblock).
    /**
     * irow, icol: The block will be inserted in location (irow, icol).
     * op: the Operator to be inserted.
     */
    void SetBlock(int iRow, int iCol, mfem::Operator *op);
    //! This method is present since required by the abstract base class Solver
    virtual void SetOperator(const mfem::Operator &){ }

    /// Operator application
    virtual void Mult (const mfem::Vector & x, mfem::Vector & y) const;

    /// Action of the transpose operator
    virtual void MultTranspose (const mfem::Vector & x, mfem::Vector & y) const;

    virtual ~BlockLDL2x2();

    //! Controls the ownership of the blocks: if nonzero, BlockOperator will
    //! delete all blocks that are set (non-NULL); the default value is zero.
    int owns_blocks;

private:

    //! Number of block rows
    int nBlocks;
    //! Row offsets for the starting position of each block
    mfem::Array<int> offsets;
    //! 2D array that stores each block of the operator.
    mfem::Array2D<mfem::Operator *> op;

    //! Temporary Vectors used to efficiently apply the Mult and MultTranspose methods.
    mutable mfem::BlockVector xblock;
    mutable mfem::BlockVector yblock;
    mutable mfem::BlockVector u;
    mutable mfem::BlockVector v;
};
}//namespace parelag
#endif /* BLOCKLDL2X2_HPP_ */
