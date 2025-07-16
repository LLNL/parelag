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

#ifndef _REDISTRIBUTOR_HPP_
#define _REDISTRIBUTOR_HPP_

#include <memory>
#include <vector>

#include <mfem.hpp>

//#include "topology/Topology.hpp"
//#include "structures/BooleanMatrix.hpp"
//#include "structures/SharingMap.hpp"
//#include "topology/TopologyTable.hpp"
//#include "utilities/elagError.hpp"
//#include "partitioning/MetisGraphPartitioner.hpp"
//#include "amge/DofHandler.hpp"
#include "amge/DeRhamSequence.hpp"


#include "matred.hpp"

namespace parelag
{

unique_ptr<ParallelCSRMatrix> Move(matred::ParMatrix& A);

void Mult(const ParallelCSRMatrix& A, const mfem::Array<int>& x, mfem::Array<int>& Ax);

/// From the parallel proc-to-proc connectivity table,
/// get a copy of the global matrix as a serial matrix locally (via permutation),
/// and then call METIS to "partition processors" in each processor locally
std::vector<int> RedistributeElements(
      ParallelCSRMatrix& elem_face, int& num_redist_procs, bool geometric_partitioning = false);

/// A helper to redistribute AgglomeratedTopology, DofHandler, DeRhamSequence
class Redistributor
{
    using ParMatrix = matred::ParMatrix;

   // Enumeration convention follows the ones in AgglomeratedTopology/DofHandler
   std::vector<unique_ptr<ParallelCSRMatrix> > redTrueEntity_trueEntity;
   std::vector<unique_ptr<ParallelCSRMatrix> > redEntity_trueEntity;
   std::vector<unique_ptr<ParallelCSRMatrix> > redTrueDof_trueDof;
   std::vector<unique_ptr<ParallelCSRMatrix> > redDof_trueDof;

   shared_ptr<AgglomeratedTopology> redist_topo;

   std::shared_ptr<AgglomeratedTopology> Redistribute(
         const AgglomeratedTopology& topo);

   std::unique_ptr<DofHandler> Redistribute(const DofHandler& dof);

   unique_ptr<ParallelCSRMatrix> BuildRedEntToTrueEnt(
         const ParallelCSRMatrix& elem_trueEntity) const;

   unique_ptr<ParallelCSRMatrix> BuildRedEntToRedTrueEnt(
         const ParallelCSRMatrix& redEntity_trueEntity) const;

   unique_ptr<ParallelCSRMatrix> BuildRedTrueEntToTrueEnt(
         const ParallelCSRMatrix& redEntity_redTrueEntity,
         const ParallelCSRMatrix& redEntity_trueEntity) const;

   unique_ptr<ParallelCSRMatrix>
   BuildRepeatedDofToTrueDof(const DofHandler& dof, int codim);

   unique_ptr<ParallelCSRMatrix>
   BuildRepeatedDofRedistribution(const AgglomeratedTopology& topo,
                                  const DofHandler& dof,
                                  const DofHandler& redist_dof,
                                  int codim, int jform);
public:

   /// Constructor for Redistributor
   /// A redistributed topology will be constructed and stored in the class
   /// @param topo topology in the original data distribution
   /// @param elem_redist_procs an array of size number of local elements.
   /// elem_redist_procs[i] indicates which processor the i-th local element
   /// will be redistributed to. Other entities are redistributed accordingly.
   Redistributor(const AgglomeratedTopology& topo,
                 const std::vector<int>& elem_redist_procs);

   /// @param num_redist_procs number of processors to be redistributed to
   Redistributor(const AgglomeratedTopology& topo, int& num_redist_procs, bool geometric_redistribution = false);

   void Init(const AgglomeratedTopology& topo,
             const std::vector<int>& elem_redist_procs);

   const ParallelCSRMatrix& TrueEntityRedistribution(int codim) const
   {
      return *(redTrueEntity_trueEntity[codim]);
   }

   const ParallelCSRMatrix& TrueDofRedistribution(int jform) const
   {
      return *(redTrueDof_trueDof[jform]);
   }

   const ParallelCSRMatrix* TrueDofRedistributionPtr(int jform) const
   {
      return redTrueDof_trueDof[jform].get();
   }

   AgglomeratedTopology& GetRedistributedTopology() { return *redist_topo; }

   std::shared_ptr<DeRhamSequenceAlg> Redistribute(const DeRhamSequence& seq);

   friend class MultiRedistributor;
};

class MultiRedistributor
{
protected:
   MPI_Comm parent_comm_;
   // FIXME (aschaf 09/13/22): find a better name than copy/copies?
   int num_copies_;
   std::vector<std::shared_ptr<Redistributor>> redistributors_;
   int mycopy_;
   MPI_Comm child_comm_;

   bool geometric_redistribution_;

   std::vector<std::vector<unique_ptr<ParallelCSRMatrix>>> trueDof_redTrueDof;

   void Init(const AgglomeratedTopology& topo, const int num_current_procs, int& num_redist_procs);

public:

   /// Constructor for MultiRedistributor
   /// A list of redistributed topologies will be constructed and stored in the class
   /// @param num_current_procs number of parent processors from which to redistribute
   /// @param num_redist_procs number of processors to be redistributed to
   MultiRedistributor(const AgglomeratedTopology& topo, const int num_current_procs, int& num_redist_procs, bool geometric_redistribution = false);

   /// @brief Constructor for MultiRedistributor
   /// A list of redistributed topologies will be constructed and stored in the class
   /// @param num_current_procs number of parent processors from which to redistribute
   /// @param num_redist_procs number of processors to be redistributed to
   /// @param other_redistributors another MultiRedistributor we copy the Communicator from
   MultiRedistributor(const AgglomeratedTopology& topo, const int num_current_procs, int& num_redist_procs, const MultiRedistributor &other_redistributors);

   std::shared_ptr<Redistributor> GetRedistributor() const
   {
      return redistributors_[mycopy_];
   }

   std::vector<std::shared_ptr<Redistributor>> GetRedistributors() const
   {
      return redistributors_;
   }

   std::shared_ptr<Redistributor> GetRedistributor(int copy) const
   {
      PARELAG_ASSERT(copy >= 0 && copy < num_copies_)
      return redistributors_[copy];
   }

   std::vector<std::shared_ptr<AgglomeratedTopology>> GetRedistributedTopologies() const;

   std::vector<std::shared_ptr<DeRhamSequenceAlg>> Redistribute(const std::shared_ptr<DeRhamSequence>& seq);

   // std::shared_ptr<AgglomeratedTopology> RedistributeComm(const AgglomeratedTopology& redist_topo);

   // std::shared_ptr<DeRhamSequenceAlg> RedistributeComm(const DeRhamSequenceAlg& redist_seq);

   // Access to the transpose of Redistributor::TrueDofRedistribution(jform)
   // If not set, we build it.
   ParallelCSRMatrix* TrueDofRedistributedTrueDofPtr(int group, int jform);

   /// Get the split communicator object
   MPI_Comm GetChildComm() const noexcept
   {
      return child_comm_;
   }

   int GetNumCopies() const
   {
      return num_copies_;
   }

   int GetMyCopy() const
   {
      return mycopy_;
   }

};

} // namespace parelag

#endif // _REDISTRIBUTOR_HPP_
