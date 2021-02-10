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
#include "structures/BooleanMatrix.hpp"
#include "structures/SharingMap.hpp"
#include "topology/TopologyTable.hpp"
#include "utilities/elagError.hpp"
#include "partitioning/MetisGraphPartitioner.hpp"
#include "amge/DofHandler.hpp"

#include "matred.hpp"

namespace parelag
{

class Redistributor
{
   // Enumeration convention follows the one in AgglomeratedTopology
   vector<matred::ParMatrix> redTrueEntity_trueEntity;
   vector<unique_ptr<ParallelCSRMatrix> > redTE_TE_helper;

   vector<matred::ParMatrix> redEntity_redTrueEntity;
   vector<unique_ptr<ParallelCSRMatrix> > redE_redTE_helper;

   matred::ParMatrix BuildRedEntToTrueEnt(const SerialCSRMatrix& elem_trueEntity);
   matred::ParMatrix BuildRedEntToTrueEnt(const ParallelCSRMatrix& elem_trueEntity);
   matred::ParMatrix BuildNewEntTrueEnt(const matred::ParMatrix& redEntity_trueEntity);
public:
   Redistributor(const AgglomeratedTopology& topo,
                 const std::vector<int>& elem_redist_procs);

   const ParallelCSRMatrix& TrueEntityRedistribution(int codim) const
   {
      return *(redTE_TE_helper[codim]);
   }

   const ParallelCSRMatrix& Redistributed_EntityTrueEntity(int codim) const
   {
      return *(redE_redTE_helper[codim]);
   }

   std::shared_ptr<AgglomeratedTopology> Redistribute(
         const AgglomeratedTopology& topo) const;


   std::unique_ptr<DofHandler> Redistribute(
           const DofHandler& dof,
           const std::shared_ptr<AgglomeratedTopology>& redist_topo) const;
};

} // namespace parelag

#endif // _REDISTRIBUTOR_HPP_
