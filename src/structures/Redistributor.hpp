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

#include "topology/Topology.hpp"
//#include "structures/BooleanMatrix.hpp"
//#include "structures/SharingMap.hpp"
//#include "topology/TopologyTable.hpp"
//#include "utilities/elagError.hpp"
//#include "partitioning/MetisGraphPartitioner.hpp"
#include "amge/DofHandler.hpp"

#include "matred.hpp"

namespace parelag
{

unique_ptr<ParallelCSRMatrix> Move(matred::ParMatrix& A);

class Redistributor
{
    using ParMatrix = matred::ParMatrix;

   // Enumeration convention follows the ones in AgglomeratedTopology/DofHandler
   std::vector<unique_ptr<ParallelCSRMatrix> > redTrueEntity_trueEntity;
   std::vector<unique_ptr<ParallelCSRMatrix> > redEntity_trueEntity;
   std::vector<unique_ptr<ParallelCSRMatrix> > redTrueDof_TrueDof;

   unique_ptr<ParallelCSRMatrix> BuildRedEntToTrueEnt(
         const ParallelCSRMatrix& elem_trueEntity) const;

   unique_ptr<ParallelCSRMatrix> BuildRedEntToRedTrueEnt(
         const ParallelCSRMatrix& redEntity_trueEntity) const;

   unique_ptr<ParallelCSRMatrix> BuildRedTrueEntToTrueEnt(
         const ParallelCSRMatrix& redEntity_redTrueEntity,
         const ParallelCSRMatrix& redEntity_trueEntity) const;
public:
   Redistributor(const AgglomeratedTopology& topo,
                 const std::vector<int>& elem_redist_procs);

   const ParallelCSRMatrix& TrueEntityRedistribution(int codim) const
   {
      return *(redTrueEntity_trueEntity[codim]);
   }

   const ParallelCSRMatrix& TrueDofRedistribution(int jform) const
   {
      return *(redTrueDof_TrueDof[jform]);
   }

//   const ParallelCSRMatrix& Redistributed_EntityTrueEntity(int codim) const
//   {
//      return *(redE_redTE_helper[codim]);
//   }

   std::shared_ptr<AgglomeratedTopology> Redistribute(
         const AgglomeratedTopology& topo);

   std::unique_ptr<DofHandler> Redistribute(
           const DofHandler& dof,
           const std::shared_ptr<AgglomeratedTopology>& redist_topo);
};

void Mult(const ParallelCSRMatrix& A, const std::vector<int>& x, std::vector<int>& Ax);

} // namespace parelag

#endif // _REDISTRIBUTOR_HPP_
