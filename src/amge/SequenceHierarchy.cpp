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

enum partition_type { MFEMRefined, METIS };

void PrintCoarseningTime(int level, double time, partition_type type)
{
    string msg = type == METIS ? "(using METIS)" : "(using MFEM refinement info)";
    std::cout << "SequenceHierarchy: level-" << level << " DeRhamSequence is "
              << "coarsened " << msg << " in " << time << " seconds.\n";
}

SequenceHierarchy::SequenceHierarchy(shared_ptr<ParMesh> mesh, ParameterList params, bool verbose)
    : comm_(mesh->GetComm()), mesh_(mesh), params_(move(params)),
      verbose_(verbose), mass_is_assembled_(false)
{
    auto num_levels = params_.Get("Hierarchy levels", 2);
    auto fe_order = params_.Get("Finite element order", 0);
    const int start_form = mesh_->Dimension()-1; // TODO: read from ParameterList

    topo_.resize(num_levels);
    seq_.resize(num_levels);

    topo_[0] = make_shared<AgglomeratedTopology>(mesh, 1);

    if (mesh_->Dimension() == 3)
    {
        seq_[0] = make_shared<DeRhamSequence3D_FE>(
                    topo_[0], mesh.get(), fe_order, true, false);
    }
    else
    {
        seq_[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(
                    topo_[0], mesh.get(), fe_order, true, false);
    }
    seq_[0]->SetjformStart(start_form);
}

void SequenceHierarchy::Build(const Array<int>& num_elements)
{
    PARELAG_ASSERT(mesh_->GetNE() == num_elements[0]);

    auto num_levels = params_.Get("Hierarchy levels", 2);
    auto elem_coarsening_factor = params_.Get("Hierarchy coarsening factor", 8);
    auto proc_coarsening_factor = params_.Get("Processor coarsening factor", 2);
    auto num_local_elems_threshold = params_.Get("Local elements threshold", 80);
    auto num_global_elems_threshold = params_.Get("Global elements threshold", 10);
    auto upscale_order = params_.Get("Upscaling order", 0);
    auto SVD_tol = params_.Get("SVD tolerance", 1e-6);

    const int dim = mesh_->Dimension();
    const int start_form = dim-1; // TODO: read from ParameterList

    if (verbose_)
    {
        std::cout << "SequenceHierarchy: building a hierarchy of DeRhamSequence"
                  << " of at most " << num_levels << " levels ...\n";
    }

    if (mass_is_assembled_ == false)
    {
        ReplaceMassIntegrator(dim, make_unique<MassIntegrator>(), true);
    }
    seq_[0]->FemSequence()->SetUpscalingTargets(dim, upscale_order, start_form);

    // first num_elements.size()-1 levels of topo_ are constructed geometrically
    GeometricCoarsenings(num_elements, dim);

    int num_nonempty_procs;
    MPI_Comm_size(comm_, &num_nonempty_procs);
    int num_redist_procs = num_nonempty_procs;
    level_redist_procs.SetSize(num_levels);
    level_redist_procs[0]=num_redist_procs;
    for (int l = 0; l < num_elements.Size()-1; l++)
        level_redist_procs[l+1]=num_redist_procs;
    StopWatch chrono;
    MetisGraphPartitioner partitioner;
    partitioner.setParELAGDefaultMetisOptions();

    for (int l = num_elements.Size()-1; l < num_levels-1; ++l)
    {
        chrono.Clear();
        chrono.Start();

        int num_global_elems = topo_[l]->GetNumberGlobalTrueEntities(elem_t_);
        if (num_global_elems < num_global_elems_threshold)
        {
            if (verbose_)
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

        seq_[l]->SetSVDTol(SVD_tol);

        const int num_local_elems = topo_[l]->GetNumberLocalEntities(elem_t_);
#ifdef ParELAG_ENABLE_MATRED
        const int min_num_local_elems =
                MinNonzeroNumLocalElements(l, num_local_elems_threshold);

        if (min_num_local_elems < num_local_elems_threshold)
        {
            num_redist_procs = max(num_nonempty_procs/proc_coarsening_factor, 1);
        }

        if (num_redist_procs < num_nonempty_procs)
        {
            Redistributor redistributor(*topo_[l], num_redist_procs);

            if (verbose_)
            {
                std::cout << "SequenceHierarchy: minimal nonzero number of "
                          << "local elements (" << min_num_local_elems << ") on"
                          << " level " << l << " is below threshold ("
                          << num_local_elems_threshold << "), redistributing the"
                          << " level to " << num_redist_procs << " processors\n";
            }

            topo_[l+1] = topo_[l]->Coarsen(redistributor, partitioner,
                                           elem_coarsening_factor, 0, 0);
            seq_[l+1] = seq_[l]->Coarsen(redistributor);
            num_nonempty_procs = num_redist_procs;
        }
        else
#endif // ParELAG_ENABLE_MATRED
        {
            Array<int> partition(num_local_elems);
            if (num_local_elems > 0)
            {
                int num_aggs = ceil(((double)num_local_elems) / elem_coarsening_factor);
                auto elem_elem = topo_[l]->LocalElementElementTable();

                PARELAG_ASSERT_DEBUG(IsConnected(*elem_elem));

                partitioner.setParELAGDefaultFlags(num_aggs);
                partitioner.doPartition(*elem_elem, num_aggs, partition);
            }

            topo_[l+1] = topo_[l]->CoarsenLocalPartitioning(partition, 0, 0, 2);
            seq_[l+1] = seq_[l]->Coarsen();
        }

        if (verbose_) { PrintCoarseningTime(l, chrono.RealTime(), METIS); }
        level_redist_procs[l+1]=num_redist_procs;
    }

    if (verbose_)
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

void SequenceHierarchy::ReplaceMassIntegrator(
        int form,
        unique_ptr<BilinearFormIntegrator> integ,
        bool recompute_mass)
{
    mass_is_assembled_ = recompute_mass;
    DeRhamSequenceFE* seq = seq_[0]->FemSequence();
    seq->ReplaceMassIntegrator(elem_t_, form, move(integ), recompute_mass);
}

void SequenceHierarchy::GeometricCoarsenings(const Array<int>& num_elems, int dim)
{
    const int num_levels = params_.Get("Hierarchy levels", 2);
    const int num_geo_levels = min(num_levels, num_elems.Size());

    MFEMRefinedMeshPartitioner partitioner(dim);
    StopWatch chrono;

    for (int l = 0; l < num_geo_levels-1; ++l)
    {
        chrono.Clear();
        chrono.Start();

        Array<int> partition(topo_[l]->GetNumberLocalEntities(elem_t_));
        partitioner.Partition(num_elems[l], num_elems[l+1], partition);
        topo_[l+1] = topo_[l]->CoarsenLocalPartitioning(partition, 0, 0, dim == 3 ? 2 : 0);
        seq_[l+1] = seq_[l]->Coarsen();

        if (verbose_) { PrintCoarseningTime(l, chrono.RealTime(), MFEMRefined); }
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
