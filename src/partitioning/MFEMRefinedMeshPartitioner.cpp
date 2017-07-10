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

#include "elag_partitioning.hpp"

MFEMRefinedMeshPartitioner::MFEMRefinedMeshPartitioner(int nDimensions):
	nDim(nDimensions)
{

}

void copyXtimes(int x, const Array<int> & orig, Array<int> & xcopy)
{
	int osize = orig.Size();
	int xsize = xcopy.Size();
	elag_assert(osize*x == xsize);

	const int * odata = orig.GetData();
	int * xdata = xcopy.GetData();

	int d;
	for(int i = 0; i < osize; ++i)
	{
		d = odata[i];
		for(int j = 0; j < x; ++j, ++xdata)
			*xdata = d;
	}
}


void MFEMRefinedMeshPartitioner::Partition(int nElements, int nParts, Array<int> & partitioning)
{
	int nsplits = 1<<nDim;// nsplit = 2^nDim.

	elag_assert(nElements > nParts);
	elag_assert(partitioning.Size() == nElements);
	elag_assert(nElements % nParts == 0);

	int coarseringFactor = nElements / nParts;
	elag_assert(coarseringFactor % nsplits == 0);

	int npass = 0;
	int tmp = coarseringFactor;

	while(tmp > 1)
	{
		tmp /= nsplits;
		++npass;
	}

	int * p = partitioning.GetData();

	for(int i = 0; i < nParts; ++i)
		p[i] = i;

	int len = nParts;
	for(int i = 0; i < npass; ++i)
	{
		Array<int> orig(p, len);
		Array<int> xcopy(p+len, len * (nsplits-1) );
		copyXtimes(nsplits-1, orig, xcopy);
		len *= nsplits;
	}
}

MFEMRefinedMeshPartitioner::~MFEMRefinedMeshPartitioner()
{

}

