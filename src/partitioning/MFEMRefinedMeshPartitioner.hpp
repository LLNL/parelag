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

#ifndef MFEMREFINEDMESHPARTITIONER_HPP_
#define MFEMREFINEDMESHPARTITIONER_HPP_

//! @class
/*!
 * @brief Essentially undoes one level of MFEM's mesh refinement
 *
 */
class MFEMRefinedMeshPartitioner
{
public:
	MFEMRefinedMeshPartitioner(int nDimensions);
	void Partition(int nElements, int nParts, Array<int> & partitioning);
	~MFEMRefinedMeshPartitioner();
private:
	int nDim;
};

#endif /* MFEMREFINEDMESHPARTITIONER_HPP_ */
