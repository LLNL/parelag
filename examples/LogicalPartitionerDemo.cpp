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

// Examples for calling the code
// mpirun -np 4 ./LogicalPartitionerDemo.exe
// mpirun -np 4 ./LogicalPartitionerDemo.exe --metis_domain_splitting --metis_agglomeration
// mpirun -np 4 ./LogicalPartitionerDemo.exe --metis_domain_splitting
// mpirun -np 4 ./LogicalPartitionerDemo.exe --metis_agglomeration

#include <fstream>
#include <sstream>

#include <mpi.h>

#include "elag.hpp"
#include "hypreExtension/hypreExtension.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

enum {TOPOLOGY=0, SPACES, ASSEMBLY, PRECONDITIONER, SOLVER};

const int NSTAGES = 5;
const char * stage_names[] = {"TOPOLOGY","SPACES","ASSEMBLY","PRECONDITIONER","SOLVER"};

int main (int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // Structured grid sizes in 3D
    Array<int> N(3);

    // Processors per dimension (only relevant for cartesian domain splitting)
    Array<int> Nproc(3);

    // Coarsening ratio in each dimension (only relevant for cartesian
    // agglomeration)
    Array<int> coarseningratio(3);

    // Overload options from command line
    OptionsParser args(argc, argv);
    int feorder = 0;
    args.AddOption(&feorder, "-feo", "--feorder",
                   "Polynomial order of fine finite element space.");
    int upscalingOrder = 0;
    args.AddOption(&upscalingOrder, "-uo", "--upscalingorder",
                   "Target polynomial order of coarse space.");
    N[0] = 32;
    args.AddOption(&N[0], "--Nx", "--Nx", "Grid size in x direction.");
    N[1] = 32;
    args.AddOption(&N[1], "--Ny", "--Ny", "Grid size in y direction.");
    N[2] = 32;
    args.AddOption(&N[2], "--Nz", "--Nz", "Grid size in z direction.");
    int nLevels = 4;
    args.AddOption(&nLevels, "--num-levels", "--num-levels",
                   "Number of levels for unstructured coarsening.");
    bool metis_domain_splitting = false;
    args.AddOption(&metis_domain_splitting, "--metis_domain_splitting", "--metis_domain_splitting",
                   "--no_metis_domain_splitting", "--no_metis_domain_splitting",
                   "Whether to use Metis for domain splitting.");
    Nproc[0] = 2;
    args.AddOption(&Nproc[0], "--Npx", "--Npx", "Number of processors in x direction.");
    Nproc[1] = 2;
    args.AddOption(&Nproc[1], "--Npy", "--Npy", "Number of processors in y direction.");
    Nproc[2] = 1;
    args.AddOption(&Nproc[2], "--Npz", "--Npz", "Number of processors in z direction.");
    if (num_procs == 1)
    {
        Nproc = 1;
    }
    if (num_procs != Nproc[0]*Nproc[1]*Nproc[2])
        mfem_error("Number of processors do not match input: Npx, Npy, Npz");
    bool metis_agglomeration = false;
    args.AddOption(&metis_agglomeration, "--metis_agglomeration", "--metis_agglomeration",
                   "--no_metis_agglomeration", "--no_metis_agglomeration",
                   "Whether to use Metis for mesh agglomeration.");
    coarseningratio[0] = 2;
    args.AddOption(&coarseningratio[0], "--cx", "--cx", "Coarsening ratio in x direction.");
    coarseningratio[1] = 2;
    args.AddOption(&coarseningratio[1], "--cy", "--cy", "Coarsening ratio in y direction.");
    coarseningratio[2] = 2;
    args.AddOption(&coarseningratio[2], "--cz", "--cz", "Coarsening ratio in z direction.");
    int metis_coarsening_factor = 16;
    args.AddOption(&metis_coarsening_factor, "--metis_coarsening_factor", "--metis_coarsening_factor",
                   "Coarsening factor for Metis partitioner.");
    bool do_visualize = true;
    args.AddOption(&do_visualize, "-v", "--do-visualize", "-nv", "--no-visualize",
                   "Do interactive GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(std::cout);
        MPI_Finalize();
        return 1;
    }
    PARELAG_ASSERT(args.Good());

    // Default linear solver options
    constexpr int print_iter = 0;
    constexpr int max_num_iter = 500;
    constexpr double rtol = 1e-6;
    constexpr double atol = 1e-12;

    if (myid == 0)
    {
        std::cout << "Finite Element Order " << feorder << "\n";
        std::cout << "Upscaling Order " << upscalingOrder << "\n";
        std::cout << "Nx " << N[0] << "\n";
        std::cout << "Ny " << N[1] << "\n";
        std::cout << "Nz " << N[2] << "\n";
        if(metis_domain_splitting)
            std::cout << "METIS is used to generate parallel domain splits\n";
        else
            std::cout << "Parallel domain splits are cartesian\n";
        if(metis_agglomeration)
            std::cout << "METIS is used to generate agglomerates\n";
        else
            std::cout << "Cartesian structure used to generate agglomerates\n";
    }

    shared_ptr<ParMesh> pmesh;
    Array<int> ess_attr;
    std::vector<Array<LogicalCartesianMaterialId>> parallel_ijk(nLevels);
    {
        constexpr auto el_type = Element::HEXAHEDRON;
#if (MFEM_VERSION_MAJOR < 4)
        auto mesh = make_unique<Mesh>(N[0],N[1],N[2],el_type,1,1.0,1.0,1.0);
#else
        auto mesh = make_unique<Mesh>(N[0],N[1],N[2],el_type,1,1.0,1.0,1.0,false);
#endif

        // Ensure all element attributes defaults to 1
        for(int i(0); i < mesh->GetNE(); ++i)
            mesh->GetElement(i)->SetAttribute(1);

        // Give elements we wish to keep unagglomerated unique attributes
        int attr = 2;
        for(int k(0); k < N[2]; ++k)
        {
            mesh->GetElement(k*N[0]*N[1])->SetAttribute(attr); attr++;
            mesh->GetElement(k*N[0]*N[1]+N[0]-1)->SetAttribute(attr); attr++;
            mesh->GetElement(k*N[0]*N[1]+N[0]*N[1]-1)->SetAttribute(attr); attr++;
            mesh->GetElement(k*N[0]*N[1]+N[0]*N[1]-N[0])->SetAttribute(attr); attr++;
            mesh->GetElement(k*N[0]*N[1]+0.5*N[0]*N[1]-0.5*N[0])->SetAttribute(attr); attr++;
        }

        // Setup ijk indices on the fine grid
        Array<LogicalCartesianMaterialId> serial_ijk(mesh->GetNE());
        CartesianIJK::SetupCartesianIJKMaterialId(
            *mesh,N,CartesianIJK::XYZ,serial_ijk);

        // Create parallel domain splits. Choice between METIS or Cartesian
        unique_ptr<int[]> domain_partitioning_data;
        if(metis_domain_splitting)
            domain_partitioning_data.reset(
                mesh->GeneratePartitioning(num_procs,1));
        else
            domain_partitioning_data.reset(
                mesh->CartesianPartitioning(Nproc.GetData()));
        Array<int> domain_partitioning(domain_partitioning_data.get(),
                                       mesh->GetNE());

        // FIXME: Does this need to be "mesh" or could it be "pmesh"?
        ess_attr.SetSize(mesh->bdr_attributes.Max());
        ess_attr = 1;

        pmesh = make_shared<ParMesh>(comm,*mesh,domain_partitioning);

        // Distribute the serial ijk indices to parallel
        parallel_ijk[0].SetSize(pmesh->GetNE());
        Distribute(comm,serial_ijk,domain_partitioning,parallel_ijk[0]);

    }

    const int nDimensions = pmesh->Dimension();

    if(nDimensions == 3)
        pmesh->ReorientTetMesh();

    ConstantCoefficient coeffH1(1.);
    ConstantCoefficient coeffDer(1.);
    ConstantCoefficient ubdr(0);
    ConstantCoefficient f(1);

    DenseMatrix timings(nLevels, NSTAGES);
    timings = 0.0;

    StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    StopWatch chronoInterior;

    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);
    chronoInterior.Clear();
    chronoInterior.Start();
    // Create topology on fine grid first
    topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
    chronoInterior.Stop();
    timings(0,TOPOLOGY) = chronoInterior.RealTime();

    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    if(!metis_agglomeration) // Use cartesian agglomeration to build topology
    {
        // Instantiate the LogicalPartitioner and the coarsening operator
        LogicalPartitioner partitioner;
        CoarsenLogicalCartesianOperatorMaterialId coarseningOp(coarseningratio);
        for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
        {
            chronoInterior.Clear();
            chronoInterior.Start();
            // Prepare storage for partitioning
            Array<int> partitioning(
                topology[ilevel]->GetNumberLocalEntities(at_elem));

            // Generate partitioning
            partitioner.Partition<LogicalCartesianMaterialId,CoarsenLogicalCartesianOperatorMaterialId>(
                *(topology[ilevel]->LocalElementElementTable()),
                parallel_ijk[ilevel],coarseningOp,partitioning);

            // Construct coarse agglomerated topology based on partitioning
            topology[ilevel+1] =
                topology[ilevel]->CoarsenLocalPartitioning(partitioning, 1, 1);

            // Setup ijk logical indices for use on the next level
            partitioner.ComputeCoarseLogical<LogicalCartesianMaterialId,CoarsenLogicalCartesianOperatorMaterialId>(
                coarseningOp,topology[ilevel]->AEntityEntity(at_elem),
                parallel_ijk[ilevel],parallel_ijk[ilevel+1]);
            chronoInterior.Stop();
            timings(ilevel+1,TOPOLOGY) = chronoInterior.RealTime();
        }
    }
    else // or construct agglomerated topology based on METIS
    {
        std::vector<Array<MetisMaterialId>> info(nLevels);
        info[0].SetSize(pmesh->GetNE());

        MetisGraphPartitioner mpartitioner;
        mpartitioner.setFlags(MetisGraphPartitioner::KWAY ); // BISECTION
        mpartitioner.setOption(METIS_OPTION_SEED, 0); // Fix the seed
        mpartitioner.setOption(METIS_OPTION_CONTIG,1); // Ask metis to provide contiguous partitions
        mpartitioner.setOption(METIS_OPTION_MINCONN,1);
        mpartitioner.setUnbalanceToll(2);

        LogicalPartitioner partitioner;
        std::vector<unique_ptr<CoarsenMetisMaterialId>>
            coarseningOp(nLevels-1);

        for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
        {
            Array<int> partitioning(
                topology[ilevel]->GetNumberLocalEntities(at_elem));

            chronoInterior.Clear();
            chronoInterior.Start();

            int num_partitions = partitioning.Size()/metis_coarsening_factor;
            if(num_partitions == 0) num_partitions = 1;

            coarseningOp[ilevel] = make_unique<CoarsenMetisMaterialId>(
                mpartitioner,*(topology[ilevel]),num_partitions,info[ilevel]);

            // Setup logical info on fine grid
            if(ilevel == 0)
                coarseningOp[0]->FillFinestMetisMaterialId(*pmesh,info[0]);
            // Generate the metis partitioning
            partitioner.Partition<MetisMaterialId,CoarsenMetisMaterialId>(
                *(topology[ilevel]->LocalElementElementTable()),
                info[ilevel],*(coarseningOp[ilevel]),partitioning);
            // Build coarser topology based on partitioning
            topology[ilevel+1] =
                topology[ilevel]->CoarsenLocalPartitioning(partitioning, 1, 1);

            // Setup logical info for next level
            partitioner.ComputeCoarseLogical<MetisMaterialId,CoarsenMetisMaterialId>(
                *(coarseningOp[ilevel]),topology[ilevel]->AEntityEntity(at_elem),
                info[ilevel],info[ilevel+1]);
            chronoInterior.Stop();
            timings(ilevel+1,TOPOLOGY) = chronoInterior.RealTime();
        }
    }
    chrono.Stop();
    if(myid == 0)
        std::cout<<"Timing ELEM_AGG: Mesh Agglomeration done in "
                 << chrono.RealTime() << " seconds.\n";

    for (int ilevel = 0; ilevel < nLevels; ++ilevel)
        if (do_visualize)
            ShowTopologyAgglomeratedElements(topology[ilevel].get(),pmesh.get());

    chronoInterior.Clear();
    chronoInterior.Start();
    constexpr double tolSVD = 1e-9;
    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());
    if(nDimensions == 3)
        sequence[0] = make_shared<DeRhamSequence3D_FE>(
            topology[0],pmesh.get(),feorder);
    else
        sequence[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(
            topology[0],pmesh.get(),feorder);

    DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();

    int jFormStart = 0;
    sequence[0]->SetjformStart(jFormStart);

    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 0, make_unique<MassIntegrator>(coeffH1), false);
    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 1, make_unique<VectorFEMassIntegrator>(coeffDer), true);

    // set up coefficients / targets
    sequence[0]->FemSequence()->SetUpscalingTargets(nDimensions, upscalingOrder);
    chronoInterior.Stop();
    timings(0,SPACES) = chronoInterior.RealTime();

    chrono.Clear();
    chrono.Start();
    for(int i(0); i < nLevels-1; ++i)
    {
        sequence[i]->SetSVDTol( tolSVD );
        StopWatch chronoInterior;
        chronoInterior.Clear();
        chronoInterior.Start();
        sequence[i+1] = sequence[i]->Coarsen();
        chronoInterior.Stop();
        timings(i+1, SPACES) = chronoInterior.RealTime();
        if(myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL" << i
                      << ": Coarsening done in " << chronoInterior.RealTime()
                      << " seconds.\n";
    }
    chrono.Stop();

    if(myid == 0)
        std::cout<<"Timing ELEM_AGG: Coarsening done in " << chrono.RealTime()
                 << " seconds.\n";

    const int form = 0;
    using ind_t = decltype(topology.size());
    for(ind_t ilevel = 0; ilevel < topology.size(); ++ilevel)
    {
        for(int i = 0; i < nDimensions-1; ++i)
        {
            {
                auto BB = ToUnique(
                    Mult(topology[ilevel]->GetB(i), topology[ilevel]->GetB(i+1)));
                elag_assert(BB->MaxNorm() < 1e-12);
            }

            auto& Bi  = topology[ilevel]->TrueB(i);
            auto& Bii = topology[ilevel]->TrueB(i+1);

            elag_assert(hypre_ParCSRMatrixMaxNorm(Bi) > 1 - 1e-12);
            elag_assert(hypre_ParCSRMatrixMaxNorm(Bii) > 1 - 1e-12);

            {
                auto pBB = ToUnique(ParMult(&Bi, &Bii));
                elag_assert(hypre_ParCSRMatrixMaxNorm(*pBB) < 1e-12);
                elag_assert(hypre_ParCSRMatrixFrobeniusNorm(*pBB) < 1e-12);
                elag_assert(hypre_ParCSRMatrixNorml1(*pBB) < 1e-12);
                elag_assert(hypre_ParCSRMatrixNormlinf(*pBB) < 1e-12);
            }
        }
    }

    // testUpscalingHdiv(sequence);
    FiniteElementSpace * fespace = sequence[0]->FemSequence()->GetFeSpace(form);
    auto b = make_unique<LinearForm>(fespace);
    b->AddDomainIntegrator(new DomainLFIntegrator(f));
    b->Assemble();

    auto lift = make_unique<GridFunction>(fespace);
    lift->ProjectBdrCoefficient(ubdr, ess_attr);

    DenseMatrix errors_L2_2(nLevels, nLevels);
    errors_L2_2 = 0.0;
    Vector norm_L2_2(nLevels);
    norm_L2_2 = 0.;

    DenseMatrix errors_div_2(nLevels, nLevels);
    errors_div_2 = 0.0;
    Vector norm_div_2(nLevels);
    norm_div_2 = 0.;

    Array<int> iter(nLevels);
    iter = 0;
    Array<int> ndofs(nLevels);
    ndofs = 0;
    Array<int> nnz(nLevels);
    nnz = 0;

    double tdiff;

    Array<SparseMatrix *> allP(nLevels-1);
    Array<SparseMatrix *> allD(nLevels);

    for(int i = 0; i < nLevels - 1; ++i)
        allP[i] = sequence[i]->GetP(form);

    for(int i = 0; i < nLevels; ++i)
        allD[i] = sequence[i]->GetDerivativeOperator(form);

    std::vector<unique_ptr<SparseMatrix>> Ml(nLevels);
    std::vector<unique_ptr<SparseMatrix>> Wl(nLevels);

    for(int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        Ml[k] = sequence[k]->ComputeMassOperator(form);
        Wl[k] = sequence[k]->ComputeMassOperator(form+1);
        chrono.Stop();
        tdiff = chrono.RealTime();
        if(myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in "
                      << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;
    }

    std::vector<unique_ptr<Vector>> rhs(nLevels);
    std::vector<unique_ptr<Vector>> ess_data(nLevels);
    rhs[0] = std::move(b);
    ess_data[0] = std::move(lift);
    for(int i = 0; i < nLevels-1; ++i)
    {
        rhs[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(form));
        ess_data[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(form));
        sequence[i]->GetP(form)->MultTranspose(*(rhs[i]), *(rhs[i+1]));
        sequence[i]->GetPi(form)->ComputeProjector();
        sequence[i]->GetPi(form)->GetProjectorMatrix().Mult(
            *(ess_data[i]), *(ess_data[i+1]));
    }

    std::vector<unique_ptr<Vector>> sol(nLevels);
    std::vector<unique_ptr<Vector>> help(nLevels);

    for(int k(0); k < nLevels; ++k)
    {
        sol[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
        *(sol[k]) = 0.;
        help[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
        *(help[k]) = 0.;
    }

    for(int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        SparseMatrix * M = Ml[k].get();
        SparseMatrix * W = Wl[k].get();
        SparseMatrix * D = allD[k];
        auto DtWD = ExampleRAP(*D, *W, *D);
        auto A = ToUnique(Add(*M, *DtWD));

        const int nlocdofs = A->Height();
        Array<int> marker(nlocdofs);
        marker = 0;
        sequence[k]->GetDofHandler(form)->MarkDofsOnSelectedBndr(ess_attr,marker);

        for(int mm = 0; mm < nlocdofs; ++mm)
            if(marker[mm])
                A->EliminateRowCol(mm, ess_data[k]->operator ()(mm), *(rhs[k]));

        const SharingMap & h1_dofTrueDof(
            sequence[k]->GetDofHandler(0)->GetDofTrueDof());

        Vector prhs(h1_dofTrueDof.GetTrueLocalSize());
        h1_dofTrueDof.Assemble(*(rhs[k]), prhs);
        unique_ptr<HypreParMatrix> pA =
            Assemble(h1_dofTrueDof, *A, h1_dofTrueDof);

        elag_assert(prhs.Size() == pA->Height());

        chrono.Stop();
        tdiff = chrono.RealTime();
        if(myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in "
                      << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;

        DtWD.reset();
        A.reset();

        ndofs[k] = pA->GetGlobalNumRows();
        nnz[k] = pA->NNZ();

        iter[k] = UpscalingHypreSolver(
            form, pA.get(), prhs,
            sequence[k].get(),
            k, PRECONDITIONER, SOLVER,
            print_iter, max_num_iter, rtol, atol,
            timings, h1_dofTrueDof, *(sol[k]));

        //ERROR NORMS
        {
            *(help[k]) = *(sol[k]);
            for(int j = k; j > 0; --j)
                allP[j-1]->Mult(*(help[j]), *(help[j-1]));

            norm_L2_2(k) = Ml[k]->InnerProduct(*(sol[k]), *(sol[k]));
            Vector dsol( allD[k]->Height() );
            allD[k]->Mult(*(sol[k]), dsol );
            norm_div_2(k) = Wl[k]->InnerProduct(dsol, dsol );

            for(int j(0); j < k; ++j)
            {
                if(help[j]->Size() != sol[j]->Size() || sol[j]->Size() != allD[j]->Width() )
                    mfem_error("size don't match \n");

                const int size  = sol[j]->Size();
                const int dsize = allD[j]->Height();
                Vector u_H( help[j]->GetData(), size );
                Vector u_h( sol[j]->GetData(), size  );
                Vector u_diff( size ), du_diff( dsize );
                u_diff = 0.; du_diff = 0.;

                subtract(u_H, u_h, u_diff);
                allD[j]->Mult(u_diff, du_diff);

                errors_L2_2(k,j) =  Ml[j]->InnerProduct(u_diff, u_diff);
                errors_div_2(k,j) =  Wl[j]->InnerProduct(du_diff, du_diff);
            }
        }

        //VISUALIZE SOLUTION
        if (do_visualize)
        {
            MultiVector tmp(sol[k]->GetData(), 1, sol[k]->Size() );
            sequence[k]->show(form, tmp);
        }

    }

    OutputUpscalingTimings(ndofs, nnz, iter, timings,
                           stage_names);

    ReduceAndOutputUpscalingErrors(errors_L2_2, norm_L2_2,
                                   errors_div_2, norm_div_2);


    return EXIT_SUCCESS;
}
