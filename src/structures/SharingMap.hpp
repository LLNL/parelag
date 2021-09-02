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

#ifndef SHARINGMAP_HPP_
#define SHARINGMAP_HPP_

#include <memory>
#include <mfem.hpp>

#include "ParELAG_Config.h"
#include "elag_typedefs.hpp"
#include "utilities/HypreTraits.hpp"

namespace parelag
{

/// \class SharingMap
/// \brief Class to manage the relationship between the local
/// (processor based) and the global numbering.
///
/// The internal data structure consists of two HypreParMatrix objects:
///
/// -- entity_trueEntity: for each local id (processor based numbering)
///    provide the global id (plus the orientation)
///
/// -- entity_trueEntity_entity = (entity_trueEntity)*(entity_trueEntity)^T
///    for each local id (processor based numbering) provide the local
///    id on the other processors sharing the same id.
///
/// TODO This class should be refactored w.r.t. orientation.
class SharingMap
{
    friend class GhostMap;

public:
    /// Empty Constructor. Set everything to NULL.
    SharingMap(MPI_Comm comm);

    /// Deep copy constructor. x and xtrue will be reset
    SharingMap(const SharingMap & map);

    /// Set up the SharingMap for entity type codim from a ParMesh.
    ///
    /// For codim = 0 we allocate a DG0 finite element space and then
    /// we use the dof_TrueDof relationship (this will be an unsigned
    /// relationship)
    ///
    /// For codim = 1 we allocate a RT0 finite element space and then
    /// we use the dof_TrueDof relationship (this will be a signed
    /// relationship)
    ///
    /// For codim = 2 we allocate a ND0 finite element space and then
    /// we use the dof_TrueDof relationship (this will be an unsigned
    /// relationship)
    ////
    void SetUp(mfem::ParMesh & pmesh, int codim);

    /// Set up the SharingMap from a ParFiniteElementSpace.
    ///
    /// SharingMap will invalidate the Dof_TrueDof matrix of feSpace.
    void SetUp(mfem::ParFiniteElementSpace * fes, int useDofSign);

    /// Set up the SharingMap from its own basic properties.
    void SetUp(mfem::Array<int> & entityStart,
               mfem::Array<int> & trueEntityStart,
               std::unique_ptr<ParallelCSRMatrix> entity_trueEntity);

    /// entity_trueEntity is assumed to own starts arrays
    void SetUp(std::unique_ptr<ParallelCSRMatrix> entity_trueEntity);

    /// Set up a SharingMap when entity and trueEntity coincide. In
    /// this case we just pass the local (True) size.
    void SetUp(int localSize);

    /// Set up a SharingMap for the agglomerated entities given the
    /// Transpose of an agglomeration table, and the SharingMap for
    /// the fine entities
    void SetUp(SerialCSRMatrix & e_AE, SharingMap & e_Te);

    /// SetUp a SharingMap when a DofHandler is provided.
    //void SetUp(DofHandler & dof);

    /// SetUp a SharingMap from a fine grid sharing map and the local
    /// to the processor interpolator P and cochain projector Pi. Pi*P
    /// == Id.
    void SetUp(const SerialCSRMatrix & Pi,
               const SharingMap & fdof_fdof,
               const SerialCSRMatrix & P);

    /// Build a SharingMap from an entity_TrueEntity map and an
    /// entity_interiordofs matrix.
    void SetUp(SharingMap & e_Te, mfem::SparseMatrix & e_idof);

    /// Destructor.
    virtual ~SharingMap();


    /// \name Distribute operations. All these methods return 0 if successful.
    //@{

    /// The master will broadcast its value to all slaves
    int Synchronize(mfem::Array<int> & data) const;

    /// Distribute shared data (int type) from owner to ghosts.
    int Distribute(const mfem::Array<int> & trueData,
                   mfem::Array<int> & data) const;

    /// Distribute shared data (int double) from owner to ghosts.
    int Distribute(const mfem::Vector & trueData,
                   mfem::Vector & data) const;
    //@}
    /// \name Reduction operations. All these methods return 0 if successful.
    //@{

    /// From a vector containing ghosts form the unique vector by
    /// dropping non local entries. (Integer version)
    int IgnoreNonLocal(const mfem::Array<int> & data,
                       mfem::Array<int> & trueData) const;

    /// From a vector containing ghosts form the unique vector by
    /// dropping non local entries. (double version)
    int IgnoreNonLocal(const mfem::Vector & data, mfem::Vector & trueData) const;

    /// From a vector containing ghosts form the unique vector by
    /// adding local and non-local entries. (Integer version)
    int Assemble(const mfem::Array<int> & data, mfem::Array<int> & trueData) const;

    /// From a vector containing ghosts form the unique vector by
    /// adding local and non-local entries. (double version)
    int Assemble(const mfem::Vector & data, mfem::Vector & trueData) const;
    //@}

    /// out = A * entity_TrueEntity_entity * B.
    std::unique_ptr<ParallelCSRMatrix> ParMatmultAB(
        const SerialCSRMatrix & A,
        const SerialCSRMatrix & B,
        const mfem::Array<int> & row_starts_A,
        const mfem::Array<int> & col_starts_B) const;

    /// out = At * entity_TrueEntity_entity * B.
    std::unique_ptr<ParallelCSRMatrix> ParMatmultAtB(
        SerialCSRMatrix & A,
        SerialCSRMatrix & B,
        const mfem::Array<int> & col_starts_A,
        const mfem::Array<int> & col_starts_B) const;

    /// out = At * entity_TrueEntity_entity * A.
    std::unique_ptr<ParallelCSRMatrix> ParMatmultAtA(
        SerialCSRMatrix & A,
        const mfem::Array<int> & col_starts_A ) const;

    /// \name friend functions
    //@{
    friend std::unique_ptr<SerialCSRMatrix> Distribute(
        SharingMap & range,
        ParallelCSRMatrix & A,
        SharingMap & domain);

    friend std::unique_ptr<ParallelCSRMatrix> IgnoreNonLocalRange(
        const SharingMap & range,
        SerialCSRMatrix & A,
        const SharingMap & domain );

#ifdef ParELAG_ENABLE_PETSC
    /// tid should be PETSC_MATAIJ or PETSC_MATIS
    friend std::unique_ptr<mfem::PetscParMatrix> AssemblePetsc(
        const SharingMap & range,
        SerialCSRMatrix & A,
        const SharingMap & domain,
        mfem::Operator::Type tid);
#endif

    friend std::unique_ptr<ParallelCSRMatrix> Assemble(
        const SharingMap & range,
        SerialCSRMatrix & A,
        const SharingMap & domain);

    /// Pulls in the contributions from other processes and sums them
    /// into A. Outputs result as new SerialCSRMatrix.
    friend std::unique_ptr<SerialCSRMatrix> AssembleNonLocal(
        SharingMap & range,
        SerialCSRMatrix & A,
        SharingMap & domain );

    //@}
    /// \name Getters
    //@{

    inline int GetTrueGlobalSize() const { return trueEntity_start.Last(); }

    inline int GetTrueLocalSize() const
    { return trueEntity_start[AssumedPID_+1] - trueEntity_start[AssumedPID_]; }

    inline int GetGlobalSize() const { return entity_start.Last(); }

    inline int GetLocalSize() const
    { return entity_start[AssumedPID_+1] - entity_start[AssumedPID_]; }

    inline int MyGlobalOffset() const { return entity_start[AssumedPID_]; }

    MPI_Comm GetComm() const noexcept { return Comm_; }
    //@}

    int DebugCheck();

    /// This function returns:
    ///  -1 if the entity is Shared but not owned by the processor
    ///   0 if the entity is not shared with other processors
    ///   1 if the entity is Shared and it is owned by the processor
    int IsShared(int localId) const;

    /// Number of shared entities.
    int GetNumberShared() const;
    /// Number of shared owned entities
    int GetNumberSharedOwned() const;
    /// Number of shared not owned entities
    int GetNumberSharedNotOwned() const;

    /// Ids off all SharedEntities. Entities for which this process is
    /// the owner are listed first.
    const mfem::Array<int> & SharedEntitiesId() const;
    void ViewOwnedSharedEntitiesId(mfem::Array<int> & ownedSharedId);
    void ViewNotOwnedSharedEntitiesId(mfem::Array<int> & notOwnedSharedId);

    ParallelCSRMatrix * get_entity_trueEntity() const noexcept
    {
        return entity_trueEntity.get();
    }

    std::unique_ptr<mfem::HypreParVector>
    TrueDataToParTrueData(mfem::Vector trueData);

private:

    void resetHypreParVectors();
    void storeSharedEntitiesIds();
    void round(const mfem::Vector & d, mfem::Array<int> & a) const;

    /// Mpi communicator.
    MPI_Comm Comm_;

    /// assumedPID assumed processor id to address the arrays
    /// entity_start and trueEntity_start.
    ///
    /// assumedPID = pid if HYPRE_AssumedPartitionCheck() is false
    /// assumedPID = 0 HYPRE_AssumedPartitionCheck() is true.
    int AssumedPID_;

    /// assumedNumProc assumed number of processors to determine the
    /// size of the arrays entity_start and trueEntity_start.
    ///
    /// assumedNumProc = MPI_Comm_size if HYPRE_AssumedPartitionCheck() is false
    /// assumedNumProc = 2 if HYPRE_AssumedPartitionCheck() is true.
    int AssumedNumProc_;

    /// entity_start is an array that represents the global offsets
    /// for the entity in this processors when shared entities/dofs are
    /// counted twice.
    ///
    /// It is used as the row_start array of the entity_trueEntity matrix.
    ///
    /// It has dimension
    /// - n_procs if HYPRE_AssumedPartitionCheck() is false
    /// - 3 if HYPRE_AssumedPartitionCheck() is true.
    mfem::Array<int> entity_start;

    /// trueEntity_start is an array that represents the global
    /// offsets for the entity in this processors when shared
    /// entities/dofs are counted only once (we will assign them to the
    /// process with lowest id).
    ///
    /// It is used as the col_start array of the entity_trueEntity matrix.
    ///
    /// It has dimension
    /// - n_procs if HYPRE_AssumedPartitionCheck() is false
    /// - 3 if HYPRE_AssumedPartitionCheck() is true.
    mfem::Array<int> trueEntity_start;

    /// ParCSRMatrix to manage the communications.
    std::unique_ptr<ParallelCSRMatrix> entity_trueEntity;

    /// ParCSRMatrix to manage the communications between overlapping
    /// serial objects.
    std::unique_ptr<ParallelCSRMatrix> entity_trueEntity_entity;

    /// mfem::Array containing the Ids of all Shared Entities, Owned
    /// entities are listed first.
    mfem::Array<int> sharedEntityIds;
    /// Number of owned SharedEntities
    int nOwnedSharedEntities_;

    /// \name help Vectors to convert array int into doubles. Not too
    //efficient :(, but ok for now.

    //@{
    mutable mfem::Vector HelpData_;
    mutable mfem::Vector HelpTrueData_;
    HypreTraits<hypre_ParVector>::unique_ptr_t xTrue_;
    HypreTraits<hypre_ParVector>::unique_ptr_t x_;
    //@}
};

#ifdef ParELAG_ENABLE_PETSC
std::unique_ptr<mfem::PetscParMatrix> AssemblePetsc(
    const SharingMap & range,
    SerialCSRMatrix & A,
    const SharingMap & domain,
    mfem::Operator::Type tid);
#endif

std::unique_ptr<ParallelCSRMatrix> Assemble(
    const SharingMap & range,SerialCSRMatrix & A,const SharingMap & domain);

}//namespace parelag
#endif /* SHARINGMAP_HPP_ */
