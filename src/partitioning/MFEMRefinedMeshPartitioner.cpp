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

#include "MFEMRefinedMeshPartitioner.hpp"

#include "utilities/elagError.hpp"
#include "utilities/ParELAG_TimeManager.hpp"

namespace parelag
{
using namespace mfem;

MFEMRefinedMeshPartitioner::MFEMRefinedMeshPartitioner(int nDimensions):
    nDim(nDimensions)
{
}

void copyXtimes(int x, const Array<int> & orig, Array<int> & xcopy)
{
    int osize = orig.Size();
#ifdef PARELAG_ASSERTING
    int xsize = xcopy.Size();
#endif
    elag_assert(osize*x == xsize);

    const int * odata = orig.GetData();
    int * xdata = xcopy.GetData();

    int d;
    for(int i = 0; i < osize; ++i)
    {
        d = odata[i];
        for(int j = 0; j < x; ++j, ++xdata)
            *xdata = d;
    }
}


void MFEMRefinedMeshPartitioner::Partition(int nElements, int nParts, Array<int> & partitioning)
{
    int nsplits = 1<<nDim;// nsplit = 2^nDim.
    if (nElements == 0 && nParts == 0)
        return;

    elag_assert(nElements > nParts);
    elag_assert(partitioning.Size() == nElements);
    elag_assert(nElements % nParts == 0);

    int coarseningFactor = nElements / nParts;
    elag_assert(coarseningFactor % nsplits == 0);

    int * p = partitioning.GetData();

    // MFEM 4.1 changed the numbering in refinement.
#if (MFEM_VERSION_MAJOR >= 4 && MFEM_VERSION_MINOR >= 1)
    int j = 0;
    for(int i = 0; i < nParts; ++i)
        for(int k = 0; k < coarseningFactor; ++k)
            p[j++] = i;
    elag_assert(j == nElements);
#else
    int npass = 0;
    int tmp = coarseningFactor;

    while(tmp > 1)
    {
        tmp /= nsplits;
        ++npass;
    }

    for(int i = 0; i < nParts; ++i)
        p[i] = i;

    int len = nParts;
    for(int i = 0; i < npass; ++i)
    {
        Array<int> orig(p, len);
        Array<int> xcopy(p+len, len * (nsplits-1) );
        copyXtimes(nsplits-1, orig, xcopy);
        len *= nsplits;
    }
#endif
}

MFEMRefinedMeshPartitioner::~MFEMRefinedMeshPartitioner()
{
}

std::shared_ptr<mfem::ParMesh> BuildParallelMesh(MPI_Comm &comm, mfem::Mesh &mesh, std::vector<SerialRefinementInfo> &serial_refinements, ParameterList &parameter_list)
{
    auto timer = TimeManager::AddTimer("Mesh : Build Parallel Mesh");
    int nDimensions = mesh.Dimension();

    int num_ranks, myid;
    int num_ser_ref;
    MPI_Comm_size(comm, &num_ranks);
    MPI_Comm_rank(comm, &myid);
    mfem::Array<int> par_partitioning;
    const std::string part_type = parameter_list.Get("Parallel partitioning", "cartesian");

    if (part_type.compare("cartesian") == 0)
    {
        int cartesian[3] = {1, 1, 1};
        if (num_ranks > 1)
        {
            int num = num_ranks;
            cartesian[2] = int(ceil(pow(num, 1. / 3)));
            while( num_ranks % cartesian[2] != 0 && cartesian[2] < num_ranks)
                cartesian[2]++;
            num /= cartesian[2];
            cartesian[1] = int(ceil(sqrt(num)));
            while( num_ranks % cartesian[1] != 0 && cartesian[1] < num)
                cartesian[1]++;
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
    else if (part_type.compare("metis") == 0)
    {
        par_partitioning.MakeRef(mesh.GeneratePartitioning(num_ranks), mesh.GetNE());
        // post-process such that the first serial elements are on the first processor
        Array<int> perm(num_ranks);
        std::set<int> skip;
        int proc = 0;
        for (auto &&part : par_partitioning)
        {
            if (skip.find(part) == skip.end())
            {
                skip.insert(part);
                perm[part] = proc++;
                part = perm[part];
            }
            else
            {
                part = perm[part];
            }
        }
    }
    else if ( part_type.compare("geometric") == 0 && (num_ser_ref = serial_refinements.size()))
    {
        // Array<int> renumber;
        // serial_refinements[0].partition.Copy(renumber);
        MFEMRefinedMeshPartitioner partitioner(nDimensions);
        int num_elems = mesh.GetNE();
        int coarsening_factor = pow(2, nDimensions);
        int num_partitioning_levels = num_ser_ref;
        mfem::Array<int> partition0(num_elems);
        std::generate(partition0.begin(), partition0.end(), [n = 0] () mutable { return n++; }); // fill with [0,num_elems)
        for (int l = 0; l < num_ser_ref; l++)
        {
            if (num_elems / coarsening_factor < num_ranks)
            {
                num_partitioning_levels = l;
                break;
            }
            serial_refinements[l].partition.SetSize(num_elems);
            partitioner.Partition(num_elems, num_elems / coarsening_factor, serial_refinements[l].partition);
            num_elems /= coarsening_factor;
        }
        // serial_refinements.resize(num_ser_ref);
        for (int l = 0; l < num_partitioning_levels; l++)
        {
            for (auto &&a : partition0)
                a = serial_refinements[l].partition[a];
            serial_refinements[l].partition.SetSize(0);
        }
        partition0.Copy(par_partitioning);
        serial_refinements[0].partition.SetSize(0);
        int nParts = par_partitioning.Max() + 1;
        if (nParts > num_ranks)
        {
            int nLocalParts = nParts / num_ranks;
            int remainder = nParts - num_ranks * nLocalParts;
            Array<int> renumber(nParts);
            int o = 0;
            for (int p = 0; p < num_ranks; p++)
            {
                int loc = nLocalParts;
                if (remainder > 0)
                {
                    loc++;
                    remainder--;
                }
                std::fill_n(renumber.GetData() + o, loc, p);
                o += loc;
            }
            for (auto &&a : par_partitioning)
                a = renumber[a];
        }
    }
    else
        PARELAG_NOT_IMPLEMENTED();

    int proc_coarsen = parameter_list.Get("Processor coarsening factor", 2);
    int procNParts = num_ranks;
    for (int l = 0; l < serial_refinements.size(); l++)
    {
        procNParts = std::max(1, procNParts / proc_coarsen);
        serial_refinements[l].num_redist_proc = procNParts;
    }

    if (0 && serial_refinements.size())
    {
        auto timer = TimeManager::AddTimer("Mesh : build permutation map");
        mfem::Array<int> partitioning_permutation(mesh.GetNE());
        mfem::Array<mfem::Array<int>*> tmp1(num_ranks);
        for (auto && arr : tmp1)
        {
            arr = new mfem::Array<int>(0);
            arr->Reserve(par_partitioning.Size() / num_ranks * 15 / 10);
        }

        for (int i = 0; i < par_partitioning.Size(); i++)
            tmp1[par_partitioning[i]]->Append(i);
        
        mfem::Array<int> tmp2, elem_offsets(num_ranks + 1);
        elem_offsets[0] = 0;
        tmp2.Reserve(par_partitioning.Size());
        int o = 1;
        for (auto && arr : tmp1)
        {
            tmp2.Append(*arr);
            elem_offsets[o++] = arr->Size();
        }
        elem_offsets.PartialSum();

        // assert partitioning_permutation.Size() == tmp2.Size()

        for (int i = 0; i < partitioning_permutation.Size(); i++)
            partitioning_permutation[tmp2[i]] = i;
        tmp1.DeleteAll();
        int proc_coarsen = parameter_list.Get("Processor coarsening factor", 2);
        int coarseningFactor = 8;

        SerialRefinementInfo &ser_ref_info = serial_refinements[0];
        {
            mfem::Array<int> tmp3(par_partitioning.Size());
            auto* p = tmp3.GetData();
            auto* perm = partitioning_permutation.GetData();
            int nParts = par_partitioning.Size() / coarseningFactor; // 3D
            int j = 0;
            for(int i = 0; i < nParts; ++i)
                for(int k = 0; k < coarseningFactor; ++k)
                    p[perm[j++]] = i;

            int procNParts = std::max(1, num_ranks / proc_coarsen);
            ser_ref_info.num_redist_proc = procNParts;
            int num2 = nParts / procNParts;
            mfem::Array<int> tmp4(tmp3);
            for (auto && a : tmp4)
            {
                a = (a / num2);
            }

            {
                tmp1.SetSize(num_ranks);
                auto* data = tmp4.GetData();
                for (int o = 0; o < num_ranks; o++)
                {
                    tmp1[o] = new Array<int>;
                    int size = elem_offsets[o+1] - elem_offsets[o];
                    tmp1[o]->MakeRef(data + elem_offsets[o], size);
                }
            }
            ser_ref_info.elem_redist_procs.assign(tmp4.GetData() + elem_offsets[myid], tmp4.GetData() + elem_offsets[myid + 1]);

            // if (myid == 0)
            // {
            //     for (int p = 0; p < num_ranks; p++)
            //         tmp1[p]->Print(mfem::out, tmp1[p]->Size());
            // }

            int chunksize = par_partitioning.Size() / procNParts;
            ser_ref_info.partition.Reserve(chunksize);
            if (0 && myid < procNParts)
            {
                for (int iid = 0; iid < num_ranks; iid++)
                {
                    for (int i = 0; i < tmp1[iid]->Size(); i++)
                    {
                        if ((*tmp1[iid])[i] == myid)
                            ser_ref_info.partition.Append(tmp3[elem_offsets[iid] + i]);
                    }
                }
            }
        }
        int level_size = serial_refinements[0].partition.Size();
        for (int i = 1; i < serial_refinements.size(); i++)
        {
            int procNParts = std::max(1, serial_refinements[i-1].num_redist_proc / proc_coarsen);
            serial_refinements[i].num_redist_proc = procNParts;
            level_size = level_size / 8;
            serial_refinements[i].elem_redist_procs.resize(level_size, myid / proc_coarsen);
            if (myid >= procNParts)
                level_size = 0;
            else
                level_size *= proc_coarsen;
        }

    }

    return std::make_shared<mfem::ParMesh>(comm, mesh, par_partitioning);
}


}//namespace parelag
