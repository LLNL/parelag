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

#ifndef MFEMREFINEDMESHPARTITIONER_HPP_
#define MFEMREFINEDMESHPARTITIONER_HPP_

#include <mfem.hpp>
#include "topology/Topology.hpp"
#include "utilities/ParELAG_TimeManager.hpp"

namespace parelag
{
//! @class
/*!
 * @brief Essentially undoes one level of MFEM's mesh refinement
 *
 */
class MFEMRefinedMeshPartitioner
{
public:
    MFEMRefinedMeshPartitioner(int nDimensions);
    void Partition(int nElements, int nParts, mfem::Array<int> & partitioning);
    void PermutedPartition(int nElements, int nParts, const mfem::Array<int>& permutation, mfem::Array<int> & partitioning);
    ~MFEMRefinedMeshPartitioner();
private:
    int nDim;
};

struct MFEMMaterialId
{
    int MFEM_partition;
    int materialId;
};

inline bool operator==(const MFEMMaterialId & lhs,
                       const MFEMMaterialId & rhs)
{
    return (lhs.MFEM_partition == rhs.MFEM_partition &&
            lhs.materialId == rhs.materialId);
}

class CoarsenMFEMMaterialId
{
public:
    CoarsenMFEMMaterialId(
        MFEMRefinedMeshPartitioner & partitioner,
        AgglomeratedTopology & topo,
        int & num_partitions,
        mfem::Array<MFEMMaterialId> & info)
    {
        partitioning.SetSize(topo.GetNumberLocalEntities(AgglomeratedTopology::ELEMENT));
        partitioner.Partition(topo.GetNumberLocalEntities(AgglomeratedTopology::ELEMENT),
                num_partitions,partitioning);
        info.SetSize(partitioning.Size());
        for(int i(0); i < partitioning.Size(); ++i)
            info[i].MFEM_partition = partitioning[i];
    }

    void FillFinestMFEMMaterialId(const mfem::ParMesh & pmesh,
                                   mfem::Array<MFEMMaterialId> & info)
    {
        elag_assert(partitioning.Size() == pmesh.GetNE());
        info.SetSize(pmesh.GetNE());
        for(int i(0); i < pmesh.GetNE(); ++i)
            info[i].materialId = pmesh.GetElement(i)->GetAttribute();
    }

    mfem::Array<int> & GetMFEMPartitioning() { return partitioning; }

    inline void operator()(const MFEMMaterialId & fine,
                           MFEMMaterialId & coarse) const
    {
        coarse.MFEM_partition = fine.MFEM_partition;
        coarse.materialId = fine.materialId;
    }
private:
    mfem::Array<int> partitioning;
};

inline std::shared_ptr<mfem::ParMesh> BuildParallelMesh(MPI_Comm &comm, mfem::Mesh &mesh, mfem::Array<int> &partitioning_permutation, bool use_metis = true)
{
    auto timer = TimeManager::AddTimer("Mesh : Build Parallel Mesh");
    int num_ranks, myid;
    MPI_Comm_size(comm, &num_ranks);
    MPI_Comm_rank(comm, &myid);
    mfem::Array<int> par_partitioning;
    if (!use_metis)
    {
        int cartesian[3] = {1, 1, 1};
        if (num_ranks > 1)
        {
            int num = num_ranks;
            cartesian[2] = int(ceil(pow(num, 1. / 3)));
            num /= cartesian[2];
            cartesian[1] = int(ceil(sqrt(num)));
            num /= cartesian[1];
            cartesian[0] = int(ceil(num));
            if (myid == 0)
            {
                printf("-- Cartesian partioning : %dx%dx%d\n", cartesian[0], cartesian[1], cartesian[2]);
                std::cout << std::flush;
            }
        }
        par_partitioning.MakeRef(mesh.CartesianPartitioning(cartesian), mesh.GetNE());
    }
    else
    {
        par_partitioning.MakeRef(mesh.GeneratePartitioning(num_ranks), mesh.GetNE());
    }
    if (partitioning_permutation.Size())
    {
        auto timer = TimeManager::AddTimer("Mesh : build permutation map");
        mfem::Array<mfem::Array<int>*> tmp1(num_ranks);
        for (auto && arr : tmp1)
        {
            arr = new mfem::Array<int>(0);
            arr->Reserve(par_partitioning.Size() / num_ranks * 15 / 10);
        }

        for (int i = 0; i < par_partitioning.Size(); i++)
            tmp1[par_partitioning[i]]->Append(i);
        
        mfem::Array<int> tmp2;
        tmp2.Reserve(par_partitioning.Size());
        for (auto && arr : tmp1)
            tmp2.Append(*arr);

        // assert partitioning_permutation.Size() == tmp2.Size()

        for (int i = 0; i < partitioning_permutation.Size(); i++)
            partitioning_permutation[tmp2[i]] = i;
        tmp1.DeleteAll();
    }

    return std::make_shared<mfem::ParMesh>(comm, mesh, par_partitioning);
}

}//namespace parelag
#endif /* MFEMREFINEDMESHPARTITIONER_HPP_ */
