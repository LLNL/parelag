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

#ifndef COARSENMETISMATERIALID_HPP_
#define COARSENMETISMATERIALID_HPP_

#include <metis.h>
#include <mfem.hpp>

#include "partitioning/MetisGraphPartitioner.hpp"
#include "topology/Topology.hpp"

namespace parelag
{

struct MetisMaterialId
{
    int metis_partition;
    int materialId;
};

inline bool operator==(const MetisMaterialId & lhs,
                       const MetisMaterialId & rhs)
{
    return lhs.metis_partition ==
        rhs.metis_partition && lhs.materialId == rhs.materialId;
}

class CoarsenMetisMaterialId
{
public:
    CoarsenMetisMaterialId(
        MetisGraphPartitioner & partitioner,
        AgglomeratedTopology & topo,
        int & num_partitions,
        mfem::Array<MetisMaterialId> & info)
    {
        partitioner.doPartition(*(topo.LocalElementElementTable()),
                                topo.Weight(0),num_partitions,partitioning);
        info.SetSize(partitioning.Size());
        for (int i(0); i < partitioning.Size(); ++i)
            info[i].metis_partition = partitioning[i];
    }

    void FillFinestMetisMaterialId(const mfem::ParMesh & pmesh,
                                   mfem::Array<MetisMaterialId> & info)
    {
        elag_assert(partitioning.Size() == pmesh.GetNE());
        info.SetSize(pmesh.GetNE());
        for (int i(0); i < pmesh.GetNE(); ++i)
            info[i].materialId = pmesh.GetElement(i)->GetAttribute();
    }

    mfem::Array<int> & GetMetisPartitioning() { return partitioning; }

    inline void operator()(const MetisMaterialId & fine,
                           MetisMaterialId & coarse) const
    {
        coarse.metis_partition = fine.metis_partition;
        coarse.materialId = fine.materialId;
    }
private:
    mfem::Array<int> partitioning;
};

}//namespace parelag
#endif /* COARSENMETISMATERIALID_HPP_ */
