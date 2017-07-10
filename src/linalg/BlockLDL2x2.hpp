/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#ifndef BLOCKLDL2X2_HPP_
#define BLOCKLDL2X2_HPP_

class BlockLDL2x2 : public Solver
{
public:
	BlockLDL2x2(const Array<int> & offsets);

	//! Add a square block op in the block-entry (iblock, iblock).
	/**
	 * iblock: The block will be inserted in location (iblock, iblock).
	 * op: the Operator to be inserted.
	 */
	void SetDiagonalBlock(int iblock, Operator *op);
	//! Add a block op in the block-entry (iblock, jblock).
	/**
	 * irow, icol: The block will be inserted in location (irow, icol).
	 * op: the Operator to be inserted.
	 */
	void SetBlock(int iRow, int iCol, Operator *op);
	//! This method is present since required by the abstract base class Solver
	virtual void SetOperator(const Operator &op){ }

	/// Operator application
	virtual void Mult (const Vector & x, Vector & y) const;

	/// Action of the transpose operator
	virtual void MultTranspose (const Vector & x, Vector & y) const;

	virtual ~BlockLDL2x2();

	//! Controls the ownership of the blocks: if nonzero, BlockOperator will
	//! delete all blocks that are set (non-NULL); the default value is zero.
	int owns_blocks;

private:

	//! Number of block rows
	int nBlocks;
	//! Row offsets for the starting position of each block
	Array<int> offsets;
	//! 2D array that stores each block of the operator.
	Array2D<Operator *> op;

	//! Temporary Vectors used to efficiently apply the Mult and MultTranspose methods.
	mutable BlockVector xblock;
	mutable BlockVector yblock;
	mutable BlockVector u;
	mutable BlockVector v;
};

#endif /* BLOCKLDL2X2_HPP_ */
