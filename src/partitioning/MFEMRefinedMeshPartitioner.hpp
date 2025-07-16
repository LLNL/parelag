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
#include "utilities/ParELAG_ParameterList.hpp"

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

// TODO (aschaf 2023-12-19) : replace with just num_redist_proc
struct SerialRefinementInfo
{
    int num_redist_proc = 0;
    std::vector<int> elem_redist_procs{};
    int num_elems = 0;
    mfem::Array<int> partition{};
    mfem::Array<int> reordering{};
};

/// @brief Generates a partitioning and distributes the serial mesh across some communicator
/// @param comm MPI communicator to distribute the mesh across
/// @param mesh serial mesh
/// @param serial_refinements
/// @param parameter_list parameter list
/// @return local part of the parallel mesh, if serial_refinements had non-zero length it contains the number of processors each serial refinement level can be redistributed
std::shared_ptr<mfem::ParMesh> BuildParallelMesh(MPI_Comm &comm, mfem::Mesh &mesh, std::vector<SerialRefinementInfo> &serial_refinements, ParameterList &parameter_list);

}//namespace parelag
#endif /* MFEMREFINEDMESHPARTITIONER_HPP_ */
