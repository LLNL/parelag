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

#ifndef SHARINGMAP_HPP_
#define SHARINGMAP_HPP_

class DofHandler;

/*
 * @class SharingMap
 * @brief Class to manage the relationship between the local (processor based) and the global numbering.
 *
 * The internal data structure consists of two HypreParMatrix objects:
 * -- entity_trueEntity: for each local id (processor based numbering) provide the global id (plus the orientation)
 * -- entity_trueEntity_entity = (entity_trueEntity) * (entity_trueEntity)^T for each local id (processor based numbering) provide the local id on the other processors
 *                               sharing the same id.
 *
 * TODO This class should be refactored w.r.t. orientation.
 */

class SharingMap
{
public:
	//! Empty Constructor. Set everything to NULL.
	SharingMap(MPI_Comm comm);
	//! Set up the SharingMap for entity type codim from a ParMesh.
	/**
	 * For codim = 0 we allocate a DG0 finite element space and then we use the dof_TrueDof relationship (this will be an unsigned relationship)
	 * For codim = 1 we allocate a RT0 finite element space and then we use the dof_TrueDof relationship (this will be a signed relationship)
	 * For codim = 2 we allocate a ND0 finite element space and then we use the dof_TrueDof relationship (this will be an unsigned relationship)
	 */
	void SetUp(ParMesh * pmesh, int codim);
	//! Set up the SharingMap from a ParFiniteElementSpace.
	/** SharingMap will invalidate the Dof_TrueDof matrix of feSpace.
	 */
	void SetUp(ParFiniteElementSpace * fes, int useDofSign);
	//! Set up the SharingMap from its own basic properties.
	void SetUp(Array<int> & entityStart, Array<int> & trueEntityStart, hypre_ParCSRMatrix * entity_trueEntity);
	//! Set up a SharingMap when entity and trueEntity coincide. In this case we just pass the local (True) size.
	void SetUp(int localSize);
	/*! Set up a SharingMap for the agglomerated entities given the Transpose of an agglomeration table,
	 and the SharingMap for the fine entities */
	void SetUp(SerialCSRMatrix & e_AE, SharingMap & e_Te);
	//! SetUp a SharingMap when a DofHandler is provided.
	void SetUp(DofHandler & dof);
	//! SetUp a SharingMap from a fine grid sharing map and the local to the processor interpolator P and cochain projector Pi. Pi*P == Id.
	void SetUp(const SerialCSRMatrix & Pi, const SharingMap & fdof_fdof, const SerialCSRMatrix & P);
	//! Build a SharingMap from an entity_TrueEntity map and an entity_interiordofs matrix.
	void SetUp(SharingMap & e_Te, SparseMatrix & e_idof);
	//! Destructor.
	virtual ~SharingMap();

	//@name Distribute operations. All these methods return 0 if successfull.
	//@{
	/**
	 * The master will broadcast its value to all slaves
	 */
	int Synchronize(Array<int> & data) const;
	/**
	 * Distribute shared data (int type) from owner to ghosts.
	 */
	int Distribute(const Array<int> & trueData, Array<int> & data) const;
	/**
	 * Distribute shared data (int double) from owner to ghosts.
	 */
	int Distribute(const Vector & trueData, Vector & data) const;
	//@}
	//@name Reduction operations. All these methods return 0 if successfull.
	//@{
	/**
	 * From a vector containing ghosts form the unique vector by dropping non local entries. (Integer version)
	 */
	int IgnoreNonLocal(const Array<int> & data, Array<int> & trueData) const;
	/**
	 * From a vector containing ghosts form the unique vector by dropping non local entries. (double version)
	 */
	int IgnoreNonLocal(const Vector & data, Vector & trueData) const;
	/**
	 * From a vector containing ghosts form the unique vector by adding local and non-local entries. (Integer version)
	 */
	int Assemble(const Array<int> & data, Array<int> & trueData) const;
	/**
	 * From a vector containing ghosts form the unique vector by adding local and non-local entries. (double version)
	 */
	int Assemble(const Vector & data, Vector & trueData) const;
	//@}

	//! out = A * entity_TrueEntity_entity * B.
	hypre_ParCSRMatrix * ParMatmultAB(const SerialCSRMatrix & A, const SerialCSRMatrix & B, const Array<int> & row_starts_A, const Array<int> & col_starts_B) const;
	//! out = At * entity_TrueEntity_entity * B.
	hypre_ParCSRMatrix * ParMatmultAtB(SerialCSRMatrix & A, SerialCSRMatrix & B, const Array<int> & col_starts_A, const Array<int> & col_starts_B) const;
	//! out = At * entity_TrueEntity_entity * A.
	hypre_ParCSRMatrix * ParMatmultAtA(SerialCSRMatrix & A, const Array<int> & col_starts_A) const;

	//@name friend functions
	//@{
	friend SerialCSRMatrix * Distribute(SharingMap & range, ParallelCSRMatrix & A, SharingMap & domain);
	friend ParallelCSRMatrix * IgnoreNonLocalRange(const SharingMap & range, SerialCSRMatrix & A, const SharingMap & domain);
	friend ParallelCSRMatrix * Assemble(const SharingMap & range, SerialCSRMatrix & A, const SharingMap & domain);
	friend SerialCSRMatrix * AssembleNonLocal(SharingMap & range, SerialCSRMatrix & A, SharingMap & domain);
	//@}

	//@name Getters
	//@{
	inline int GetTrueGlobalSize() const { return trueEntity_start.Last(); }
	inline int GetTrueLocalSize() const { return trueEntity_start[assumedPID+1] - trueEntity_start[assumedPID]; }
	inline int GetGlobalSize() const { return entity_start.Last(); }
	inline int GetLocalSize() const { return entity_start[assumedPID+1] - entity_start[assumedPID]; }
	inline int MyGlobalOffset() const { return entity_start[assumedPID]; }
	MPI_Comm GetComm() const {return comm; }
	//@}

	int DebugCheck();

	/*!This function returns:
	 * -1 if the entity is Shared but not owned by the processor
	 *  0 if the entity is not shared with other processors
	 *  1 if the entity is Shared and it is owned by the processor
	 */
	int IsShared(int localId) const;
	//! Number of shared entities.
	int GetNumberShared() const;
	//! Number of shared owned entities
	int GetNumberSharedOwned() const;
	//! Number of shared not owned entities
	int GetNumberSharedNotOwned() const;
	/*! Ids off all SharedEntities.
	 *  Entities for which this process is the owner are listed first.
	 */
	const Array<int> & SharedEntitiesId() const;
	void ViewOwnedSharedEntitiesId(Array<int> & ownedSharedId);
	void ViewNotOwnedSharedEntitiesId(Array<int> & notOwnedSharedId);


private:

	void resetHypreParVectors();
	void storeSharedEntitiesIds();
	void round(const Vector & d, Array<int> & a) const;

	//! Mpi communicator.
	MPI_Comm comm;
	/**
	 * assumedPID assumed processor id to address the arrays entity_start and trueEntity_start.
	 * assumedPID = pid if HYPRE_AssumedPartitionCheck() is false
	 * assumedPID = 0 HYPRE_AssumedPartitionCheck() is true.
	 */
	int assumedPID;
	/**
	 * assumedNumProc assumed number of processors to determine the size of the arrays entity_start and trueEntity_start.
	 * assumedNumProc = MPI_Comm_size if HYPRE_AssumedPartitionCheck() is false
	 * assumedNumProc = 2 if HYPRE_AssumedPartitionCheck() is true.
	 */
	int assumedNumProc;

	/** entity_start is an array that represents the global offsets for the entity in this processors
	 * when shared entities/dofs are counted twice.
	 * It is used as the row_start array of the entity_trueEntity matrix.
	 * It has dimension
	 * - n_procs if HYPRE_AssumedPartitionCheck() is false
	 * - 3 if HYPRE_AssumedPartitionCheck() is true.
	*/
	Array<int> entity_start;
	/** trueEntity_start is an array that represents the global offsets for the entity in this processors
	 * when shared entities/dofs are counted only once (we will assign them to the process with lowest id).
	 * It is used as the col_start array of the entity_trueEntity matrix.
	 * It has dimension
	 * - n_procs if HYPRE_AssumedPartitionCheck() is false
	 * - 3 if HYPRE_AssumedPartitionCheck() is true.
	 */
	Array<int> trueEntity_start;
	/**
	 * ParMatrix to manage the communications.
	 */
	hypre_ParCSRMatrix * entity_trueEntity;
	/**
	 * ParMatrix to manage the communications between overlapping serial objects.
	 */
	hypre_ParCSRMatrix * entity_trueEntity_entity;

	//! Array containing the Ids of all Shared Entities, Owned entities are listed first.
	Array<int> sharedEntityIds;
	//! Number of owned SharedEntities
	int nOwnedSharedEntities;

	//@name help Vectors to convert array int into doubles. Not too efficient :(, but ok for now.
	//@{
	mutable Vector helpData;
	mutable Vector helpTrueData;
	hypre_ParVector * xtrue;
	hypre_ParVector * x;
	//@}
};

#endif /*SHARINGMAP_HPP_ */
