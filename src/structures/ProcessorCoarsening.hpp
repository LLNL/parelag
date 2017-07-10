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

#ifndef PROCESSORCOARSENING_HPP_
#define PROCESSORCOARSENING_HPP_

/*
 * @class ProcessorCoarsening
 * @brief Class to manage reducing the number of processors as we move to coarser grids
 */

class ProcessorCoarsening
{
public:
	//! very basic constructor
	ProcessorCoarsening(MPI_Comm comm);
	//! creates the identity map for going from square matrices with one partitioning to another
	void SetUpSameOrder(Array<int> & current_global_offsets_, Array<int> & desired_global_offsets_);
	//! creates reordering matrix, given an element-to-processor partitioning vector (probably from Parmetis)
	//  eventually we may combine this with SetUp()
	void SetUpReorder(Array<int> & partitioning, int new_num_procs,
							Array<int> & current_global_offsets_);

	//! Destructor.
	virtual ~ProcessorCoarsening();

	//! the main routine, redistributes a square matrix to a (probably smaller) partition of processors
	friend HypreParMatrix * RepartitionMatrix(ProcessorCoarsening &pc, HypreParMatrix &A);

	//! returns the "right" identity matrix with a kind of rectangular partition
	hypre_ParCSRMatrix * GetRightIdentity() {return right_identity;}
	SharingMap * GetCoarseSharingMap() {return coarse;}
	SharingMap * GetFineSharingMap() {return fine;}

private:
	MPI_Comm comm;
	//! current_global_offsets used as row_starts for right_identity
	//  (this should probably belong to SharingMap, not here)
	Array<int> current_global_offsets;
	//! desired_global_offsets used as col_starts for right_identity
	//  (this should probably belong to SharingMap, not here)
	Array<int> desired_global_offsets;
	// TODO: rename _identity to _permutation throughout
	hypre_ParCSRMatrix * right_identity;
	//! square permutation matrix that maps old ordering to new ordering
	hypre_ParCSRMatrix * reordering; // == left_identity

	SharingMap * coarse;
	SharingMap * fine;

	hypre_ParCSRMatrix * transpose_partitioning(Array<int> & partitioning, 
															  int global_num_entities, int new_num_procs,
															  int * entity_starts);
	hypre_ParCSRMatrix * explode_processor_entity(hypre_ParCSRMatrix * processor_entity);
	hypre_ParCSRMatrix * create_reordering(Array<int> & partitioning, int global_num_entities,
														int new_num_procs, int * entity_starts);

};

#endif /* PROCESSORCOARSENING_HPP_ */
