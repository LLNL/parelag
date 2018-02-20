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

/*
 * coarseningMetis.cpp
 *
 *  Created on: May 16, 2014
 *      Author: barker29
 */

#include <fstream>
#include <sstream>
#include "mfem.hpp"
#include "elag.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

int main (int argc, char *argv[])
{
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    if (argc == 1)
    {
        if (myid == 0)
            std::cerr << "\nUsage: mpirun -np <np> topology <mesh_file>\n\n";
        return EXIT_FAILURE;
    }

    shared_ptr<ParMesh> pmesh;
    {
        std::ifstream imesh(argv[1]);
        if (!imesh)
        {
            if (myid == 0)
                std::cerr << "\nCan not open mesh file: " << argv[1] << "\n\n";
            return EXIT_FAILURE;
        }
        auto mesh = make_unique<Mesh>(imesh, 1, 1);
        imesh.close();

        {
            const int ref_levels = 2;
            //(int)floor(log(10000./mesh->GetNE())/log(2.)/mesh->Dimension());
            for (int l = 0; l < ref_levels; l++)
                mesh->UniformRefinement();
        }

        pmesh = make_shared<ParMesh>(comm, *mesh);

        const int par_ref_levels = 0;
        for (int l = 0; l < par_ref_levels; l++)
            pmesh->UniformRefinement();
    }

    std::vector<shared_ptr<AgglomeratedTopology>> topo(2);
    topo[0] = make_shared<AgglomeratedTopology>(pmesh, pmesh->Dimension());

    const int elem_per_agg = 64;
    if (myid == 0)
        std::cout << "Partitioning with metis with " << elem_per_agg
                  << " elements per aggregate." << std::endl;

    int nparts = pmesh->GetNE() / elem_per_agg;
    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    MetisGraphPartitioner partitioner;
    Array<int> partitioning(topo[0]->GetNumberLocalEntities(at_elem));
    partitioner.setFlags(MetisGraphPartitioner::KWAY);// BISECTION
    partitioner.setOption(METIS_OPTION_SEED, 0);// Fix the seed
    partitioner.setOption(METIS_OPTION_CONTIG,1);// contiguous partitions
    partitioner.setOption(METIS_OPTION_MINCONN,1);
    partitioner.setUnbalanceToll(1.05);
    partitioner.doPartition(
        *(topo[0]->LocalElementElementTable()), nparts, partitioning);

    topo[1] = topo[0]->CoarsenLocalPartitioning(partitioning, 1, 0);

    /*
      MFEMRefinedMeshPartitioner partitioner(pmesh->Dimension());
      Array<int> partitioning(topo[0]->GetNumberLocalEntities(at_elem));
      partitioner.Partition(topo[0]->GetNumberLocalEntities(at_elem), nparts, partitioning);
      topo[1] = topo[0]->CoarsenLocalPartitioning(partitioning, 0, 0);
    */

    for(unsigned long ilevel = 0; ilevel < topo.size(); ++ilevel)
    {
        for(int i = 0; i < pmesh->Dimension(); ++i)
        {
            topo[ilevel]->GetB(i);
            topo[ilevel]->Weight(i);
            auto tmp = topo[ilevel]->TrueWeight(i);
        }
    }

    std::stringstream msg;
    for(unsigned long ilevel = 0; ilevel < topo.size(); ++ilevel)
    {
        msg << "Level " << ilevel << "\n";
        topo[ilevel]->ShowMe(msg);
    }

    SerializedOutput( comm, std::cout, msg.str() );

    for(unsigned long ilevel = 0; ilevel < topo.size(); ++ilevel)
    {
        for(int i = 0; i < pmesh->Dimension()-1; ++i)
        {
            auto BB = ToUnique(
                Mult(topo[ilevel]->GetB(i), topo[ilevel]->GetB(i+1)));
            elag_assert(BB->MaxNorm() < 1e-12);
            BB.reset();

            AgglomeratedTopology::par_table_t & Bi  = topo[ilevel]->TrueB(i);
            AgglomeratedTopology::par_table_t & Bii = topo[ilevel]->TrueB(i+1);

            elag_assert( hypre_ParCSRMatrixMaxNorm(Bi) > 1 - 1e-12 );
            elag_assert( hypre_ParCSRMatrixMaxNorm(Bii) > 1 - 1e-12 );

            auto pBB = ToUnique(ParMult(&Bi, &Bii));
            elag_assert(hypre_ParCSRMatrixMaxNorm(*pBB) < 1e-12);
            elag_assert(hypre_ParCSRMatrixFrobeniusNorm(*pBB) < 1e-12);
            elag_assert(hypre_ParCSRMatrixNorml1(*pBB) < 1e-12);
            elag_assert(hypre_ParCSRMatrixNormlinf(*pBB) < 1e-12);
        }
    }

    ShowTopologyAgglomeratedElements(topo[1].get(), pmesh.get());
    ShowTopologyAgglomeratedFacets(topo[1].get(), pmesh.get());
    ShowTopologyBdrFacets(topo[1].get(), pmesh.get());

    return EXIT_SUCCESS;
}
