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
#include <utilities/ParELAG_TimeManager.hpp>

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
      verbose_(verbose), mass_is_assembled_(false), is_redistributed_(false)
{
    auto num_levels = params_.Get("Hierarchy levels", 2);
    auto fe_order = params_.Get("Finite element order", 0);
    const int start_form = mesh_->Dimension()-1; // TODO: read from ParameterList

    topo_.resize(1);
    topo_[0].resize(num_levels);
    seq_.resize(1);
    seq_[0].resize(num_levels);
    redistributors_.resize(num_levels);
    // redist_parent_topo_.resize(num_levels);
    // redist_parent_seq_.resize(num_levels);
    // redist_topo_.resize(num_levels);
    // redist_seq_.resize(num_levels);
    mycopy_.resize(num_levels);
    level_is_redistributed_.resize(num_levels, false);
    comms_.resize(1, mesh->GetComm());
    num_global_copies_.resize(1, 1);
    num_copies_.resize(num_levels, 1);
    redistribution_index.resize(num_levels, 0);

    topo_[0][0] = make_shared<AgglomeratedTopology>(mesh, 1);

    if (mesh_->Dimension() == 3)
    {
        seq_[0][0] = make_shared<DeRhamSequence3D_FE>(
                    topo_[0][0], mesh.get(), fe_order, true, false);
    }
    else
    {
        seq_[0][0] = make_shared<DeRhamSequence2D_Hdiv_FE>(
                    topo_[0][0], mesh.get(), fe_order, true, false);
    }
    seq_[0][0]->SetjformStart(start_form);
}

void SequenceHierarchy::Build(const Array<int>& num_elements, const SequenceHierarchy &other_sequence_hierarchy)
{
    PARELAG_ASSERT(mesh_->GetNE() == num_elements[0]);

    auto num_levels = params_.Get("Hierarchy levels", 2);
    auto elem_coarsening_factor = params_.Get("Hierarchy coarsening factor", 8);
    auto proc_coarsening_factor = params_.Get("Processor coarsening factor", 2);
    auto num_local_elems_threshold = params_.Get("Local elements threshold", 80);
    auto num_global_elems_threshold = params_.Get("Global elements threshold", 10);
    auto upscale_order = params_.Get("Upscaling order", 0);
    auto SVD_tol = params_.Get("SVD tolerance", 1e-6);
    auto multi_dist = params_.Get("Distribute multiple copies", false);
    
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
    seq_[0][0]->FemSequence()->SetUpscalingTargets(dim, upscale_order, start_form);

    // first num_elements.size()-1 levels of topo_ are constructed geometrically
    GeometricCoarsenings(num_elements, dim);

    int num_nonempty_procs;
    MPI_Comm_size(comm_, &num_nonempty_procs);
    int myid;
    MPI_Comm_rank(comm_, &myid);
    int num_redist_procs = num_nonempty_procs;
    int num_global_groups = 1;

    level_redist_procs.SetSize(num_levels);
    level_redist_procs[0]=num_redist_procs;
    for (int l = 0; l < num_elements.Size()-1; l++)
        level_redist_procs[l+1]=num_redist_procs;
    StopWatch chrono;
    MetisGraphPartitioner partitioner;
    partitioner.setParELAGDefaultMetisOptions();

    for (int l = num_elements.Size()-1; l < num_levels-1; ++l)
    {
        int k_l = redistribution_index[l];
        chrono.Clear();
        chrono.Start();

        int num_global_elems = topo_[k_l][l]->GetNumberGlobalTrueEntities(elem_t_);
        if (num_global_elems < num_global_elems_threshold)
        {
            if (verbose_)
            {
                std::cout << "SequenceHierarchy: global number of elements ("
                          << num_global_elems << ") on level " << l << " is "
                          << "below threshold (" << num_global_elems_threshold
                          << "), terminating the coarsening process earlier.\n";
            }
            for (int k = 0; k <= k_l; k++)
            {
                topo_[k].resize(l+1);
                seq_[k].resize(l+1);
            }
            redistribution_index.resize(l+1);
            redistributors_.resize(l+1);
            mycopy_.resize(l+1);
            level_is_redistributed_.resize(l+1);
            break;
        }
        if (&other_sequence_hierarchy != this)
        {
            if (l == other_sequence_hierarchy.topo_[0].size() - 1)
            {
                if (verbose_)
                {
                    std::cout << "SequenceHierarchy: maximal hierarchy depth "
                              << "reached on level " << l << ", terminating the"
                              << " coarsening process earlier.\n";
                }
                for (int k = 0; k <= k_l; k++)
                {
                    topo_[k].resize(l+1);
                    seq_[k].resize(l+1);
                }
                redistribution_index.resize(l+1);
                redistributors_.resize(l+1);
                mycopy_.resize(l+1);
                level_is_redistributed_.resize(l+1);
                break;
            }
        }

        seq_[k_l][l]->SetSVDTol(SVD_tol);

        const int num_local_elems = topo_[k_l][l]->GetNumberLocalEntities(elem_t_);
        const int min_num_local_elems =
                MinNonzeroNumLocalElements(l, num_local_elems_threshold);

        bool is_forced = false;
        if (&other_sequence_hierarchy != this)
        {
            is_forced = other_sequence_hierarchy.level_is_redistributed_[l];
        }
        
        if ((min_num_local_elems < num_local_elems_threshold) || is_forced)
        {
            num_redist_procs = max(num_nonempty_procs/proc_coarsening_factor, 1);
        }

        if (num_redist_procs < num_nonempty_procs)
        {
            level_is_redistributed_[l] = true;
            is_redistributed_ = true;
            if (multi_dist)
            {
                topo_.push_back(vector<shared_ptr<AgglomeratedTopology>>(num_levels));
                seq_.push_back(vector<shared_ptr<DeRhamSequence>>(num_levels));
                {
                    Timer redistribute = TimeManager::AddTimer(
                        std::string("Redistribution: Redistribute Topology and DeRhamSequence -- Level ")
                            .append(std::to_string(l)));
                    if (is_forced)
                        redistributors_[l] = make_unique<MultiRedistributor>(*topo_[k_l][l], num_nonempty_procs, num_redist_procs, *(other_sequence_hierarchy.redistributors_[l]));
                    else
                        redistributors_[l] = make_unique<MultiRedistributor>(*topo_[k_l][l], num_nonempty_procs, num_redist_procs);
                }
                auto &multi_redistributor = *redistributors_[l];
                num_copies_[l] = (num_nonempty_procs / num_redist_procs);
                num_global_groups *= num_copies_[l];
                num_global_copies_.push_back(num_global_groups);
                if (verbose_)
                {
                    if (is_forced)
                        std::cout << "SequenceHierarchy: redistributing"
                            << " level " << l << " to " << num_global_groups << " groups of " 
                            << num_redist_procs << " processors each\n";
                    else
                        std::cout << "SequenceHierarchy: minimal nonzero number of "
                            << "local elements (" << min_num_local_elems << ") on"
                            << " level " << l << " is below threshold ("
                            << num_local_elems_threshold << "), redistributing the"
                            << " level to " << num_global_groups << " groups of " 
                            << num_redist_procs << " processors each\n";
                }
                int mycopy = multi_redistributor.GetMyCopy();
                mycopy_[l+1] = mycopy;
                {
                    Timer redistribute = TimeManager::AddTimer(
                        std::string("Redistribution: Redistribute Topology and DeRhamSequence -- Level ")
                            .append(std::to_string(l)));
                    auto redist_parent_topos = multi_redistributor.GetRedistributedTopologies();
                    auto redist_parent_seqs = multi_redistributor.Redistribute(seq_[k_l][l]);
                    MPI_Comm child_comm = multi_redistributor.GetChildComm();
                    comms_.push_back(child_comm);
                    topo_[k_l + 1][l] = redist_parent_topos[mycopy]->RebuildOnDifferentComm(child_comm);
                    seq_[k_l + 1][l] = redist_parent_seqs[mycopy]->RebuildOnDifferentComm(topo_[k_l + 1][l]);
                }

                //NOTE (aschaf 09/14/22): copied from AgglomeratedTopology::Coarsen(Redistributor, ...)
                int tmp_num_local_elems = topo_[k_l+1][l]->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT);
                Array<int> partition(tmp_num_local_elems);
                if (tmp_num_local_elems > 0)
                {
                    int num_aggs = max((tmp_num_local_elems) / elem_coarsening_factor, 1);
                    auto elem_elem = topo_[k_l+1][l]->LocalElementElementTable();
                    partitioner.setParELAGDefaultFlags(num_aggs);
                    partitioner.doPartition(*elem_elem, num_aggs, partition);
                }

                topo_[k_l+1][l+1] = topo_[k_l+1][l]->CoarsenLocalPartitioning(partition, 0, 0, 2);
                seq_[k_l+1][l+1] = seq_[k_l+1][l]->Coarsen();
                redistribution_index[l+1] = k_l + 1;
            }
            else
            {
                Redistributor redistributor(*topo_[k_l][l], num_redist_procs);

                if (verbose_)
                {
                    std::cout << "SequenceHierarchy: minimal nonzero number of "
                            << "local elements (" << min_num_local_elems << ") on"
                            << " level " << l << " is below threshold ("
                            << num_local_elems_threshold << "), redistributing the"
                            << " level to " << num_redist_procs << " processors\n";
                }

                topo_[k_l][l+1] = topo_[k_l][l]->Coarsen(redistributor, partitioner,
                                            elem_coarsening_factor, 0, 0);
                seq_[k_l][l+1] = seq_[k_l][l]->Coarsen(redistributor);
                redistribution_index[l+1] = k_l;
            }
            num_nonempty_procs = num_redist_procs;
        }
        else
        {
            Array<int> partition(num_local_elems);
            if (num_local_elems > 0)
            {
                int num_aggs = ceil(((double)num_local_elems) / elem_coarsening_factor);
                auto elem_elem = topo_[k_l][l]->LocalElementElementTable();

                PARELAG_ASSERT_DEBUG(IsConnected(*elem_elem));

                partitioner.setParELAGDefaultFlags(num_aggs);
                partitioner.doPartition(*elem_elem, num_aggs, partition);
            }

            topo_[k_l][l+1] = topo_[k_l][l]->CoarsenLocalPartitioning(partition, 0, 0, 2);
            seq_[k_l][l+1] = seq_[k_l][l]->Coarsen();
            redistribution_index[l+1] = k_l;
        }

        if (verbose_) { PrintCoarseningTime(l, chrono.RealTime(), METIS); }
        level_redist_procs[l+1]=num_redist_procs;
    }

    if (verbose_)
    {
        std::cout << "SequenceHierarchy:\n";
        for (int i = 0; i < topo_[0].size(); ++i)
        {
            std::cout << "\tNumber of global elements on level " << i << " is "
                      << topo_[redistribution_index[i]][i]->GetNumberGlobalTrueEntities(elem_t_);
                          
            if (num_global_copies_[redistribution_index[i]] == 1)
                std::cout <<".\n";
            else
                std::cout << " (x" << num_global_copies_[redistribution_index[i]] <<").\n";
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
    DeRhamSequenceFE* seq = seq_[0][0]->FemSequence();
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

        Array<int> partition(topo_[0][l]->GetNumberLocalEntities(elem_t_));
        partitioner.Partition(num_elems[l], num_elems[l+1], partition);
        topo_[0][l+1] = topo_[0][l]->CoarsenLocalPartitioning(partition, 0, 0, dim == 3 ? 2 : 0);
        seq_[0][l+1] = seq_[0][l]->Coarsen();

        if (verbose_) { PrintCoarseningTime(l, chrono.RealTime(), MFEMRefined); }
    }
}

int SequenceHierarchy::MinNonzeroNumLocalElements(int level, int zero_replace)
{
    int num_local_elems = topo_[redistribution_index[level]][level]->GetNumberLocalEntities(elem_t_);
    num_local_elems = num_local_elems == 0 ? zero_replace : num_local_elems;
    int out;
    MPI_Allreduce(&num_local_elems, &out, 1, MPI_INT, MPI_MIN, comm_);
    return out;
}

void SequenceHierarchy::ApplyTruePTranspose(int l, int jform, const Vector &x, Vector &y)
{
    int k_l = redistribution_index[l];
    if (redistributors_[l])
    {
        Vector z(seq_[redistribution_index[l+1]][l]->GetNumberOfTrueDofs(jform));

        RedistributeVector(l, jform, x, z);

        seq_[redistribution_index[l+1]][l]->GetTrueP(jform).MultTranspose(z, y);
    }
    else
    {
        seq_[k_l][l]->GetTrueP(jform).MultTranspose(x, y);
    }
}

void SequenceHierarchy::ApplyTruePi(int l, int jform, const Vector &x, Vector &y)
{
    int k_l = redistribution_index[l];
    if (redistributors_[l])
    {
        Vector z(seq_[redistribution_index[l+1]][l]->GetNumberOfTrueDofs(jform));

        RedistributeVector(l, jform, x, z);

        seq_[redistribution_index[l+1]][l]->GetTruePi(jform).Mult(z, y);
    }
    else
    {
        seq_[k_l][l]->GetTruePi(jform).Mult(x, y);
    }
}

bool SequenceHierarchy::IsVectorRedistributed(int level, int jform, const Vector &x)
{
    if (IsRedistributed(level))
    {
        if (x.Size() == seq_[1][level]->GetNumberOfDofs(jform))
            return true;
    }

    return false;
}

const vector<shared_ptr<DeRhamSequence>>& SequenceHierarchy::GetDeRhamSequences(int k) const
{
    return seq_[k];
}

unique_ptr<ParallelCSRMatrix> SequenceHierarchy::RedistributeParMatrix(int level, int jform, const ParallelCSRMatrix *mat)
{
    PARELAG_NOT_IMPLEMENTED();

    int copies = num_copies_[level];
    vector<unique_ptr<ParallelCSRMatrix>> tmp(copies);
    for (int i(0); i < copies; i++)
    {
        // auto* td_rTD = sequence[ilevel]->RedistributedSequence(i)->ViewTrueDofRedTrueDof(sform);
        // auto* orig_td_rTD = orig_sequence[ilevel]->RedistributedSequence(i)->ViewTrueDofRedTrueDof(sform);
        // red_Gt[ilevel][i].reset(
        // mfem::RAP(orig_td_rTD, Gt[ilevel].get(), td_rTD)
        // );
    }
    return {};
}

unique_ptr<ParallelCSRMatrix> SequenceHierarchy::RedistributeParMatrix(int level, int jform, const ParallelCSRMatrix *mat, const SequenceHierarchy &range_hierarchy)
{
    Timer redistribute = TimeManager::AddTimer(
        std::string("Redistribution: Redistribute ParallelCSRMatrix -- Level ")
        .append(std::to_string(level)));
    unique_ptr<ParallelCSRMatrix> out;
    int copies = num_copies_[level];
    vector<unique_ptr<ParallelCSRMatrix>> tmp(copies);
    for (int i(0); i < copies; i++)
    {
        auto* td_rTD = redistributors_[level]->TrueDofRedistributedTrueDofPtr(i, jform);
        auto* orig_td_rTD = range_hierarchy.redistributors_[level]->TrueDofRedistributedTrueDofPtr(i, jform);
        tmp[i].reset(
            mfem::RAP(orig_td_rTD, mat, td_rTD)
        );
        tmp[i]->SetOwnerFlags(0, 0, 0);
    }
    int mycopy = redistributors_[level]->GetMyCopy();
    auto red_comm = redistributors_[level]->GetChildComm();
    // we do not need to delete these two, this is done by the constructor
    mfem::SparseMatrix* diag = new mfem::SparseMatrix;
    mfem::SparseMatrix* offd = new mfem::SparseMatrix;
    HYPRE_Int* cmap;
    tmp[mycopy]->GetDiag(*diag);
    tmp[mycopy]->GetOffd(*offd, cmap);
    out.reset(
        new mfem::HypreParMatrix(red_comm, tmp[mycopy]->GetGlobalNumRows(), tmp[mycopy]->GetGlobalNumCols(),
        tmp[mycopy]->GetRowStarts(), tmp[mycopy]->GetColStarts(), diag, offd, cmap, true)
        );
    // tmp[mycopy]->SetOwnerFlags(0, 0, 0);

    return out;
}

void SequenceHierarchy::RedistributeVector(int level, int jform, const Vector &x, Vector &redist_x)
{
    Timer redistribute = TimeManager::AddTimer(
        std::string("Redistribution: Redistribute Vector -- Level ")
        .append(std::to_string(level)));
    int k_lp1 = redistribution_index[level+1];
    int copies = num_copies_[level];
    vector<Vector> tmp(copies);
    int mycopy = redistributors_[level]->GetMyCopy();
    redist_x.SetSize(seq_[k_lp1][level]->GetNumberOfTrueDofs(jform));
    tmp[mycopy].SetDataAndSize(redist_x.GetData(), redist_x.Size());
    for (int i(0); i < copies; i++)
    {
        auto rTD_tD = redistributors_[level]->GetRedistributor(i)->TrueDofRedistribution(jform);
        rTD_tD.Mult(x, tmp[i]);
    }
}

void SequenceHierarchy::ShowTrueData(int level, int k, int groupid, int jform, MultiVector &true_v)
{
    // TODO (aschaf 09/23/22) Asserts
    int ilevel = level;
    int ik = k;
    int igid = groupid;
    int groupsize = 2;
    MultiVector v0(true_v.GetData(), true_v.NumberOfVectors(), true_v.Size());

    for ( ; ik >= 1; ik--)
    {
        MultiVector vk(v0.StealData(), v0.NumberOfVectors(), v0.Size());
        // first traverse communicator upwards
        for ( ; seq_[ik][ilevel]->ViewFinerSequence(); ilevel--)
        {
            MultiVector vi(vk.StealData(), vk.NumberOfVectors(), vk.Size());
            auto finer_sequence = seq_[ik][ilevel]->FinerSequence();
            vk.SetSizeAndNumberOfVectors(finer_sequence->GetNumTrueDofs(jform), true_v.NumberOfVectors());
            Mult(finer_sequence->GetTrueP(jform), vi, vk);
        }

        // copy to other processor
        auto * tD_rTD = redistributors_[ilevel]->TrueDofRedistributedTrueDofPtr(igid % groupsize, jform);
        vk.SetSizeAndNumberOfVectors(tD_rTD->Width(),  true_v.NumberOfVectors());
        v0.SetSizeAndNumberOfVectors(tD_rTD->Height(), true_v.NumberOfVectors());
        Mult(*tD_rTD, vk, v0);
        igid /= num_global_copies_[ik-1];
    }

    seq_[ik][ilevel]->ShowTrueData(jform, v0);
}

}
