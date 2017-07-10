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

#include"elag_structures.hpp"

/**
 * findMinimalIntersectionSets
 * Inputs:
 * \param Z the intersection matrix (should be square).
 * \param skipDiagEntryLessThan if the diagonal entry is less that this quantity, the entity will not belong to any MIS.
 * Ouput: the entiy_MIS relationship
 * \param entity_MIS: SparseMatrix where entry i,j is non-zero if entity j belongs to MIS j.
 * The numerical value of the matrix is + or - one according to the relative orientation of MIS and entity.
 */

SerialCSRMatrix * findMinimalIntersectionSets(SerialCSRMatrix & Z, double skipDiagEntryLessThan)
{
	elag_assert( Z.Size() == Z.Width() );

	int nEntities( Z.Size() );
	double toll(1e-10);

	int * i_entity_MIS = new int[nEntities+1];

	Vector diagZ(nEntities);
	Z.GetDiag(diagZ);

	int nnz(0);
	for(int i = 0; i < nEntities; ++i)
	{
		i_entity_MIS[i] = nnz;
		if( diagZ(i)  - skipDiagEntryLessThan > -toll )
			++nnz;
	}
	i_entity_MIS[nEntities] = nnz;

	int * j_entity_MIS = new int[nnz];
	std::fill(j_entity_MIS, j_entity_MIS+nnz, -1);
	double * a_entity_MIS = new double[nnz];
	int current_MIS(0);

	const int * Z_I = Z.GetI();
	const int * Z_J = Z.GetJ();
	const double * Z_D = Z.GetData();

	int iEntity(0), jEntity(0);
	double Z_ii(0), Z_jj(0), Z_ij(0);
	for ( ; iEntity<nEntities; ++iEntity)
	{
		Z_ii = diagZ(iEntity);
		/* NOTE: in the if statement first we check whenever iEntity belongs to a minimal intersection set (Z_ii >= skipDiagEntryLessThan)
		 * then we check if we have already found the MIS which contains iEntity ( j_entity_MIS[ i_entity_MIS[iEntity] ] == -1 ).
		 * If by mistake one invert the ordering of the two conditions valgrind may complain: in fact if the last entity (iEntity = nEntities-1)
		 * does not belong to any minimal intersection set (i.e. the first condition is false), then i_entity_MIS[iEntity] will be equal to nnz and therefore outside the
		 * area of memory pointed by j_entity_MIS.
		 */

		if ( Z_ii - skipDiagEntryLessThan > -toll && j_entity_MIS[ i_entity_MIS[iEntity] ] == -1 )
		{
			for (int ind=Z_I[iEntity]; ind<Z_I[iEntity+1]; ++ind)
				// note that we do not skip self-interaction; that is, at some point jEntity will be iEntity
			{
				jEntity = Z_J[ind];
				Z_ij = Z_D[ind];
				Z_jj = diagZ(jEntity);
				if (fabs(Z_jj - Z_ii) < toll)     //  Z_ij (=Z_ji)
				{
					if(fabs(Z_ij - Z_ii) < toll || fabs(Z_ij + Z_ii) < toll)
					{
						j_entity_MIS[i_entity_MIS[jEntity] ]=current_MIS;
						a_entity_MIS[i_entity_MIS[jEntity] ] = Z_ij / Z_ii;
						++nnz;
					}
				}
			}
			++current_MIS;
		}
	}

	return new SparseMatrix(i_entity_MIS, j_entity_MIS, a_entity_MIS, nEntities, current_MIS);
}


#if 0
SerialCSRMatrix * findMinimalIntersectionSets(SerialCSRMatrix & Z, double skipDiagEntryLessThan)
{
	elag_assert( Z.Size() == Z.Width() );

	int nEntities( Z.Size() );
	double toll(1e-10);

	Array<int> entity_MIS(nEntities);
	entity_MIS = -1;
	Vector orientation(nEntities);
	int current_MIS(0);
	int nnz(0);

	const int * Z_I = Z.GetI();
	const int * Z_J = Z.GetJ();
	const double * Z_D = Z.GetData();

	Vector diagZ(nEntities);
	Z.GetDiag(diagZ);

	int iEntity(0), jEntity(0);
	double Z_ii(0), Z_jj(0), Z_ij(0);
	for ( ; iEntity<nEntities; ++iEntity)
	{
		Z_ii = diagZ(iEntity);
		if (entity_MIS[iEntity] == -1 && Z_ii - skipDiagEntryLessThan > -toll)
		{
			for (int ind=Z_I[iEntity]; ind<Z_I[iEntity+1]; ++ind)
				// note that we do not skip self-interaction; that is, at some point jEntity will be iEntity
			{
				jEntity = Z_J[ind];
				Z_ij = Z_D[ind];
				Z_jj = diagZ[jEntity];
				if (fabs(Z_jj - Z_ii) < toll)     //  Z_ij (=Z_ji)
				{
					if(fabs(Z_ij - Z_ii) < toll || fabs(Z_ij + Z_ii) < toll)
					{
						entity_MIS[jEntity]=current_MIS;
						orientation(jEntity) = Z_ij / Z_ii;
						++nnz;
					}
				}
			}
			++current_MIS;
		}
	}

	Array<int> i_MIS_entities(current_MIS+2);
	Array<int> j_MIS_entities(nnz);
	SerialCSRMatrix * MIS_entity = transpose(entity_MIS, orientation, current_MIS);

	return MIS_entity;
}
#endif

inline int getMin(HYPRE_Int * it, HYPRE_Int * it_end, HYPRE_Int * map, HYPRE_Int defaultValue)
{
	int min = defaultValue;

	for( ; it != it_end; ++it)
		if(map[*it] < min)
			min = map[*it];

	return min;
}

/* ZZ is a symmetric matrix, such that ZZ(i,i) != 0, and if ZZ(i,j) = +- ZZ(i,i) if i and j correspond to the same
 trueZ, 0 otherwise. NOTE hypre_diag( ZZ ) should be a diagonal matrix.
 At the output we construct Z_trueZ, where Z_true(i, any_fix_k ) and Z_true(j, any_fix_k) are both non zeros if
 i and j correspond to the same trueZ k. The orientation in Z_trueZ depends on the reciprocal orientation of ZZ(i,j).
 We assign the trueZ to the processor of lower Pid. The orientation of trueZ is the same Z for the master processor.
 */
hypre_ParCSRMatrix * ParUnique(hypre_ParCSRMatrix * ZZ, Array<int> & trueZ_start)
{
	hypre_CSRMatrix * diag_ZZ = hypre_ParCSRMatrixDiag(ZZ);
	hypre_CSRMatrix * offd_ZZ = hypre_ParCSRMatrixOffd(ZZ);

	// Check that diag_ZZ is actually diagonal
	elag_assert( hypre_CSRMatrixNumRows(diag_ZZ) == hypre_CSRMatrixNumCols(diag_ZZ) );
	elag_assert( hypre_CSRMatrixNumRows(diag_ZZ) == hypre_CSRMatrixNumNonzeros(diag_ZZ) );
#ifdef ELAG_DEBUG
	{
		HYPRE_Int * i_diag_ZZ = hypre_CSRMatrixI(diag_ZZ);
		HYPRE_Int * j_diag_ZZ = hypre_CSRMatrixJ(diag_ZZ);
		double    * a_diag_ZZ = hypre_CSRMatrixData(diag_ZZ);
		int localSize        = hypre_CSRMatrixNumRows(diag_ZZ);
		int nnz = hypre_CSRMatrixNumNonzeros(diag_ZZ);

		for(int irow = 0; irow < localSize+1; ++irow)
			elag_assert( i_diag_ZZ[irow] == irow );

		for(int innz = 0; innz < nnz; ++innz)
			elag_assert( j_diag_ZZ[innz] == innz);

		for(int innz = 0; innz < nnz; ++innz)
			elag_assert( fabs( a_diag_ZZ[innz] - 1. ) < 1e-10 );
	}
#endif

	int localSize        = hypre_CSRMatrixNumRows(diag_ZZ);
//	int myFirstGlobalZ   = hypre_ParCSRMatrixFirstRowIndex(ZZ);
	int myLastGlobalZ    = hypre_ParCSRMatrixLastRowIndex(ZZ);
	HYPRE_Int * offdGlobalZmap  = hypre_ParCSRMatrixColMapOffd(ZZ);

	HYPRE_Int * i_offd_ZZ = hypre_CSRMatrixI(offd_ZZ);
	HYPRE_Int * j_offd_ZZ = hypre_CSRMatrixJ(offd_ZZ);

	// Create a selector matrix S of dimension (localSize, localTrueSize).
	// nnz(S) = localTrueSize;
	// S(i,j) != 0 if the local Z i is matched with the local TrueZ j.
	// TrueZ belongs to this processor if this processor owns the Z with lowest i.d. that is matched to TrueZ.

	int localTrueSize = 0;
	int * i_S = new int[localSize+1];

	for(int i = 0; i < localSize; ++i)
	{
		i_S[i] = localTrueSize;
		if( myLastGlobalZ < getMin(j_offd_ZZ + i_offd_ZZ[i], j_offd_ZZ + i_offd_ZZ[i+1], offdGlobalZmap, myLastGlobalZ+1) )
			localTrueSize++;
	}
	i_S[localSize] = localTrueSize;

	int * j_S = new int[localTrueSize];
	double * a_S = new double[localTrueSize];

	for(int i = 0; i < localTrueSize; ++i)
		j_S[i] = i;

	std::fill(a_S, a_S+localTrueSize, 1.);

	SerialCSRMatrix S(i_S, j_S, a_S, localSize, localTrueSize);

	MPI_Comm comm = hypre_ParCSRMatrixComm(ZZ);
	ParPartialSums_AssumedPartitionCheck( comm , localTrueSize, trueZ_start);
	ParallelCSRMatrix Sd(comm, hypre_ParCSRMatrixGlobalNumRows(ZZ), trueZ_start.Last(),
			hypre_ParCSRMatrixRowStarts(ZZ), trueZ_start.GetData(), &S);

	hypre_ParCSRMatrix * Z_trueZ = hypre_ParMatmul(ZZ, Sd);

	return Z_trueZ;
}
