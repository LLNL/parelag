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

#include <amge/SequenceHierarchy.hpp>
#include <amge/DeRhamSequenceFE.hpp>
#include <structures/Redistributor.hpp>
#include <partitioning/MFEMRefinedMeshPartitioner.hpp>
#include <partitioning/MetisGraphPartitioner.hpp>
#include <utilities/MemoryUtils.hpp>

namespace parelag
{
using namespace mfem;
using std::make_shared;

void PrintCoarseningTime(int level, double time)
{
    std::cout << "SequenceHierarchy: level-" << level << " DeRhamSequence "
              << "coarsened in " << time << " seconds.\n";
}

SequenceHierarchy::SequenceHierarchy(const std::shared_ptr<ParMesh>& mesh,
                                     int num_levels,
                                     const std::vector<int>& num_elements,
                                     int elem_coarsening_factor,
                                     int proc_coarsening_factor,
                                     int num_local_elems_threshold,
                                     int num_global_elems_threshold,
                                     bool verbose)
    : topo_(num_levels), seq_(num_levels), comm_(mesh->GetComm()), verbose_(verbose)
{
    PARELAG_ASSERT(mesh->GetNE() == num_elements[0]);

    const int dim = mesh->Dimension();
    const int feorder = 0;
    const int upscalingOrder = 0;
    const int jFormStart = dim-1;
    const int uform = dim - 1;
    const int pform = dim;

    ConstantCoefficient coeffL2(1.);
    ConstantCoefficient coeffHdiv(1.);

    if (verbose)
    {
        std::cout << "SequenceHierarchy: building a hierarchy of DeRhamSequence"
                  << " of at most " << num_levels << " levels ...\n";
    }

    topo_[0] = make_shared<AgglomeratedTopology>(mesh, 1);

    if (dim == 3)
    {
        seq_[0] = make_shared<DeRhamSequence3D_FE>(topo_[0], mesh.get(), feorder);
    }
    else
    {
        seq_[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(topo_[0], mesh.get(), feorder);
    }

    seq_[0]->SetjformStart(jFormStart);

    DeRhamSequenceFE * DRSequence_FE = seq_[0]->FemSequence();
    DRSequence_FE->ReplaceMassIntegrator(
                elem_t_, pform, make_unique<MassIntegrator>(coeffL2), false);
    DRSequence_FE->ReplaceMassIntegrator(
                elem_t_, uform, make_unique<VectorFEMassIntegrator>(coeffHdiv), true);
    DRSequence_FE->SetUpscalingTargets(dim, upscalingOrder, jFormStart);

    // first num_elements.size()-1 levels of topo_ are constructed geometrically
    GeometricPartitionings(num_elements, dim);

    int num_redist_procs;
    MPI_Comm_size(comm_, &num_redist_procs);

    StopWatch chrono;
    MetisGraphPartitioner partitioner;
    partitioner.setParELAGDefaultMetisOptions();

    for (unsigned int l = num_elements.size()-1; l < num_levels-1; ++l)
    {
        chrono.Clear();
        chrono.Start();

        int num_global_elems = topo_[l]->GetNumberGlobalTrueEntities(elem_t_);
        if (num_global_elems < num_global_elems_threshold)
        {
            if (verbose)
            {
                std::cout << "SequenceHierarchy: global number of elements ("
                          << num_global_elems << ") on level " << l << " is "
                          << "below threshold (" << num_global_elems_threshold
                          << "), terminating the coarsening process earlier.\n";
            }
            topo_.resize(l+1);
            seq_.resize(l+1);
            break;
        }

        seq_[l]->SetSVDTol(1e-6);

        const int num_local_elems = topo_[l]->GetNumberLocalEntities(elem_t_);
        const int min_num_local_elems =
                MinNonzeroNumLocalElements(l, num_local_elems_threshold);

        if (min_num_local_elems < num_local_elems_threshold)
        {
            num_redist_procs /= proc_coarsening_factor;
            if (verbose)
            {
                std::cout << "SequenceHierarchy: minimal nonzero number of "
                          << "local elements (" << min_num_local_elems << ") on"
                          << " level " << l << " is below threshold ("
                          << num_local_elems_threshold << "), redistributing the"
                          << " level to " << num_redist_procs << " processors\n";
            }

            Redistributor redistributor(*topo_[l], num_redist_procs);
            topo_[l+1] = topo_[l]->Coarsen(redistributor, partitioner,
                                           elem_coarsening_factor, 0, 0);
            seq_[l+1] = seq_[l]->Coarsen(redistributor);
        }
        else
        {
            Array<int> partition(num_local_elems);
            if (num_local_elems > 0)
            {
                int num_aggs = ceil(((double)num_local_elems) / elem_coarsening_factor);
                auto elem_elem = topo_[l]->LocalElementElementTable();
                partitioner.setParELAGDefaultFlags(num_aggs);
                partitioner.doPartition(*elem_elem, num_aggs, partition);
            }

            topo_[l+1] = topo_[l]->CoarsenLocalPartitioning(partition, 0, 0, 2);
            seq_[l+1] = seq_[l]->Coarsen();
        }

        if (verbose) { PrintCoarseningTime(l, chrono.RealTime()); }
    }


    if (verbose)
    {
        std::cout << "SequenceHierarchy:\n";
        for (int i = 0; i < topo_.size(); ++i)
        {
            std::cout << "\tNumber of global elements on level " << i << " is "
                      << topo_[i]->GetNumberGlobalTrueEntities(elem_t_) <<".\n";
        }
        std::cout << "\n";
    }
}

void SequenceHierarchy::GeometricPartitionings(
        const std::vector<int>& num_elems, int dim)
{
    MFEMRefinedMeshPartitioner partitioner(dim);
    StopWatch chrono;

    for (unsigned int l = 0; l < num_elems.size()-1; ++l)
    {
        chrono.Clear();
        chrono.Start();

        Array<int> partition(topo_[l]->GetNumberLocalEntities(elem_t_));
        partitioner.Partition(num_elems[l], num_elems[l+1], partition);
        topo_[l+1] = topo_[l]->CoarsenLocalPartitioning(partition, 0, 0,
                                                        dim == 3 ? 2 : 0);
        seq_[l+1] = seq_[l]->Coarsen();

        if (verbose_) { PrintCoarseningTime(l, chrono.RealTime()); }
    }
}

int SequenceHierarchy::MinNonzeroNumLocalElements(int level, int zero_replace)
{
    int num_local_elems = topo_[level]->GetNumberLocalEntities(elem_t_);
    num_local_elems = num_local_elems == 0 ? zero_replace : num_local_elems;
    int out;
    MPI_Allreduce(&num_local_elems, &out, 1, MPI_INT, MPI_MIN, comm_);
    return out;
}

}
