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

#include "BlockLDL2x2.hpp"

#include "utilities/elagError.hpp"

namespace parelag
{
using namespace mfem;

BlockLDL2x2::BlockLDL2x2(const Array<int> & offsets_)
    : Solver(offsets_.Last()),
      owns_blocks(0),
      nBlocks(offsets_.Size() - 1),
      offsets(0),
      op(nBlocks, nBlocks),
      u(offsets_),
      v(offsets_)
{
    elag_assert(offsets_.Size()==3);
    op = static_cast<Operator *>(NULL);
    offsets.MakeRef(offsets_);

    u = 0.0;
    v = 0.0;
}

void BlockLDL2x2::SetDiagonalBlock(int iblock, Operator *op)
{
    SetBlock(iblock, iblock, op);
}

void BlockLDL2x2::SetBlock(int iRow, int iCol, Operator *opt)
{
    op(iRow, iCol) = opt;

    MFEM_VERIFY(offsets[iRow+1] - offsets[iRow] == opt->NumRows() &&
                offsets[iCol+1] - offsets[iCol] == opt->NumCols(),
                "incompatible Operator dimensions");
}

void BlockLDL2x2::Mult (const Vector & x, Vector & y) const
{
    MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
    MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

    yblock.Update(y.GetData(),offsets);
    xblock.Update(x.GetData(),offsets);

    u.GetBlock(0) = 0.;
    op(0,0)->Mult(xblock.GetBlock(0), u.GetBlock(0));
    op(1,0)->Mult(u.GetBlock(0), v.GetBlock(1));
    add(1, xblock.GetBlock(1), -1., v.GetBlock(1), v.GetBlock(1));

    yblock = 0.;
    op(1,1)->Mult(v.GetBlock(1), yblock.GetBlock(1));
    yblock.GetBlock(1) *= -1.;

    op(0,1)->Mult(yblock.GetBlock(1), v.GetBlock(0));
    op(0,0)->Mult(v.GetBlock(0), yblock.GetBlock(0));
    add(1., u.GetBlock(0), -1., yblock.GetBlock(0), yblock.GetBlock(0));

}

void BlockLDL2x2::MultTranspose (const Vector &, Vector &) const
{
    mfem_error("");
}

BlockLDL2x2::~BlockLDL2x2()
{
    if (owns_blocks)
        for (int iRow=0; iRow < nBlocks; ++iRow)
            for (int jCol=0; jCol < nBlocks; ++jCol)
                delete op(jCol,iRow);
}

}//namespace parelag
