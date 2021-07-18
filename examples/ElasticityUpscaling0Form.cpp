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

#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

#include <mpi.h>

#include "elag.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

enum {TOPOLOGY=0, SPACES, ASSEMBLY, PRECONDITIONER, SOLVER, NSTAGES};

int main (int argc, char *argv[])
{
    //XXX: It would perform worse, as far as AMG is concerned, for MFEM versions
    //     < 4.2!
    //XXX: This is to be run only in serial currently!

    // 1. Initialize MPI
    mpi_session sess(argc,argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    const char* meshfile_c = "../meshes/boxcyl.mesh3d";
    bool gen_mesh = false;
    int ser_ref_levels = 0;
    int par_ref_levels = 2;
    int feorder = 0;
    int upscalingOrder = 0;
    bool do_visualize = true;
    bool hex = false;
    OptionsParser args(argc, argv);
    args.AddOption(&meshfile_c, "-m", "--mesh",
                   "MFEM mesh file to load.");
    args.AddOption(&ser_ref_levels, "-sr", "--nref_serial",
                   "Number of times to refine serial mesh.");
    args.AddOption(&par_ref_levels, "-pr", "--nref_parallel",
                   "Number of times to refine parallel mesh.");
    args.AddOption(&feorder, "-feo", "--feorder",
                   "Polynomial order of fine finite element space.");
    args.AddOption(&upscalingOrder, "-uo", "--upscalingorder",
                   "Target polynomial order of coarse space.");
    args.AddOption(&hex, "-hex", "--hexahedra", "-tet", "--tetrahedra",
                   "Use hexahedra instead of tetrahedra, if mesh is generated.");
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
    std::string meshfile(meshfile_c);

    // default linear solver options
    constexpr int print_iter = 1;
    constexpr int max_num_iter = 500;
    constexpr double rtol = 1e-8;
    constexpr double atol = 1e-24;

    if (myid == 0)
    {
        std::cout << "Read mesh " << meshfile << std::endl;
        std::cout << "Finite Element Order " << feorder << std::endl;
        std::cout << "Upscaling Order " << upscalingOrder << std::endl;
        std::cout << "Refine mesh in serial "
                  << ser_ref_levels << " times.\n";
        std::cout << "Refine mesh in parallel "
                  << par_ref_levels << " times.\n";
    }

    shared_ptr<ParMesh> pmesh;
    Array<int> ess_attr;
    {
        // 2. Read the (serial) mesh from the given mesh file and
        //    uniformly refine it.
        std::ifstream imesh(meshfile.c_str());
        if (!imesh)
        {
            if (myid == 0)
            {
                std::cerr << std::endl << "Can not open mesh file: "
                          << meshfile << std::endl << std::endl;
                std::cout << "Constructing default 3D mesh..." << std::endl;
            }
            gen_mesh = true;
        }
        // This guy just gets used as a stepping-stone to the ParMesh;
        // once we get there, it is destroyed.
        auto mesh = gen_mesh? (hex ? make_unique<Mesh>(8, 2, 2, Element::HEXAHEDRON,
                                                       true, 4.0, 1.0, 1.0) :
                                     make_unique<Mesh>(8, 2, 2, Element::TETRAHEDRON,
                                                       true, 4.0, 1.0, 1.0)) :
                              make_unique<Mesh>(imesh, 1, 1);
        imesh.close();

        ess_attr.SetSize(mesh->bdr_attributes.Max());
        ess_attr = 0;
        if (gen_mesh)
            ess_attr[4] = 1;
        else
            ess_attr[0] = 1;

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        pmesh = make_shared<ParMesh>(comm,*mesh);
    }

    const int nDimensions = pmesh->Dimension();
    const int nLevels = par_ref_levels+1;
    std::vector<int> level_nElements(nLevels);
    if (myid == 0)
        std::cout << "NE: " << pmesh->GetNE() << " NF: " << pmesh->GetNumFaces() << " (BDR: " <<  pmesh->GetNBE() << ")" << std::endl;
    for (int l = 0; l < par_ref_levels; l++)
    {
        level_nElements[par_ref_levels-l] = pmesh->GetNE();// / (1 << (par_ref_levels-l));
        pmesh->UniformRefinement();
        if (myid == 0)
            std::cout << "NE: " << pmesh->GetNE() << " NF: " << pmesh->GetNumFaces() << " (BDR: " <<  pmesh->GetNBE() << ")" << std::endl;
    }
    level_nElements[0] = pmesh->GetNE();

    if (nDimensions == 3)
        pmesh->ReorientTetMesh();

    ConstantCoefficient coeffH1(1.);
    ConstantCoefficient coeffDer(1.);

    ConstantCoefficient lambda(1.);
    ConstantCoefficient mu(1.);
    VectorArrayCoefficient ubdrv(nDimensions);
    for (int i = 0; i < nDimensions; ++i)
    {
        ubdrv.Set(i, new ConstantCoefficient(0.0));
    }
    VectorArrayCoefficient f(nDimensions);
    for (int i = 0; i < nDimensions; ++i)
    {
        f.Set(i, new ConstantCoefficient(0.0));
    }
    VectorArrayCoefficient g(nDimensions);
    for (int i = 0; i < nDimensions-1; ++i)
    {
        g.Set(i, new ConstantCoefficient(0.0));
    }
    {
        Vector pull_force(pmesh->bdr_attributes.Max());
        pull_force = 0.0;
        if (gen_mesh)
            pull_force(2) = -1.0e-2;
        else
            pull_force(1) = -1.0e-2;
        g.Set(nDimensions-1, new PWConstCoefficient(pull_force));
    }

    DenseMatrix timings(nLevels, NSTAGES);
    timings = 0.0;

    StopWatch chrono;

    MFEMRefinedMeshPartitioner partitioner(nDimensions);

//    MetisGraphPartitioner partitioner;
//    partitioner.setFlags(MetisGraphPartitioner::KWAY);
////    partitioner.setFlags(MetisGraphPartitioner::RECURSIVE);
//    partitioner.setOption(METIS_OPTION_SEED, 0);
//    partitioner.setOption(METIS_OPTION_CONTIG, 1);

//    GeometricBoxPartitioner partitioner;

    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    StopWatch chronoInterior;
    chrono.Clear();
    chrono.Start();
    chronoInterior.Clear();
    chronoInterior.Start();
    topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
    chronoInterior.Stop();
    timings(0,TOPOLOGY) = chronoInterior.RealTime();

    constexpr auto AT_elem = AgglomeratedTopology::ELEMENT;
    for (int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        if (myid == 0)
            std::cout << "Elements level " << ilevel << ": "
                      << topology[ilevel]->GetNumberLocalEntities(AT_elem) << std::endl;
        Array<int> partitioning(
            topology[ilevel]->GetNumberLocalEntities(AT_elem));
        chronoInterior.Clear();
        chronoInterior.Start();

        partitioner.Partition(
            topology[ilevel]->GetNumberLocalEntities(AT_elem),
            level_nElements[ilevel+1],
            partitioning);

//        partitioner.doPartition(
//            *(topology[ilevel]->LocalElementElementTable()),
//            level_nElements[ilevel+1],
//            partitioning);

//        partitioner.doPartition(
//            *pmesh,
//            level_nElements[ilevel+1],
//            partitioning);

        topology[ilevel+1] =
            topology[ilevel]->CoarsenLocalPartitioning(partitioning,false,false);
        chronoInterior.Stop();
        timings(ilevel+1,TOPOLOGY) = chronoInterior.RealTime();
    }
    chrono.Stop();
    if (myid == 0)
    {
        std::cout << "Elements level " << nLevels-1 << ": "
                  << topology[nLevels-1]->GetNumberLocalEntities(AT_elem) << std::endl;
        std::cout<<"Timing ELEM_AGG: Mesh Agglomeration done in "
                 << chrono.RealTime() << " seconds.\n";
    }

    if (do_visualize && nDimensions <= 3)
        for (int ilevel = 1; ilevel < nLevels; ++ilevel)
            ShowTopologyAgglomeratedElements(topology[ilevel].get(), pmesh.get());

    chronoInterior.Clear();
    chronoInterior.Start();
    constexpr double tolSVD = 1e-9;
    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());

    if (nDimensions == 3)
        sequence[0] = make_shared<DeRhamSequence3D_FE>(
            topology[0],pmesh.get(),feorder);
    else
        sequence[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(
            topology[0],pmesh.get(),feorder);

    DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();
    PARELAG_ASSERT(DRSequence_FE);

    constexpr int jFormStart = 0;
    sequence[0]->SetjformStart(jFormStart);

    DRSequence_FE->ReplaceMassIntegrator(
        AT_elem,
        0,
        make_unique<MassIntegrator>(coeffH1),
        false);
    DRSequence_FE->ReplaceMassIntegrator(
        AT_elem,
        1,
        make_unique<VectorFEMassIntegrator>(coeffDer),
        true);

    Array<Coefficient *> L2coeff;
    Array<VectorCoefficient *> Hdivcoeff;
    Array<VectorCoefficient *> Hcurlcoeff;
    Array<Coefficient *> H1coeff;
    fillVectorCoefficientArray(nDimensions, upscalingOrder, Hcurlcoeff);
    fillVectorCoefficientArray(nDimensions, upscalingOrder, Hdivcoeff);
    fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);
    fillCoefficientArray(nDimensions, upscalingOrder+1, H1coeff);

    std::vector<unique_ptr<MultiVector>>
        targets(sequence[0]->GetNumberOfForms());

    int jform(0);

    targets[jform] = DRSequence_FE->InterpolateScalarTargets(jform, H1coeff);
    ++jform;

    if (nDimensions == 3)
    {
        targets[jform] =
            DRSequence_FE->InterpolateVectorTargets(jform, Hcurlcoeff);
        ++jform;
    }

    targets[jform] = DRSequence_FE->InterpolateVectorTargets(jform, Hdivcoeff);
    ++jform;

    targets[jform] = DRSequence_FE->InterpolateScalarTargets(jform, L2coeff);
    ++jform;

    freeCoeffArray(L2coeff);
    freeCoeffArray(Hdivcoeff);
    freeCoeffArray(Hcurlcoeff);
    freeCoeffArray(H1coeff);

    Array<MultiVector*> targets_in(targets.size());
    for (int ii = 0; ii < targets_in.Size(); ++ii)
        targets_in[ii] = targets[ii].get();

    sequence[0]->SetTargets(targets_in);
    chronoInterior.Stop();
    timings(0,SPACES) = chronoInterior.RealTime();

    chrono.Clear();
    chrono.Start();
    for (int i(0); i < nLevels-1; ++i)
    {
        if (myid == 0)
            std::cout << "Coarsening level " << i << "...\n";
        sequence[i]->SetSVDTol(tolSVD);
        StopWatch chronoInterior;
        chronoInterior.Clear();
        chronoInterior.Start();
        sequence[i+1] = sequence[i]->Coarsen();
        chronoInterior.Stop();
        timings(i+1, SPACES) = chronoInterior.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL" << i << ": "
                      << "Coarsening done in " << chronoInterior.RealTime()
                      << " seconds.\n";
    }
    chrono.Stop();

    if (myid == 0)
        std::cout << "Timing ELEM_AGG: Coarsening done in "
                  << chrono.RealTime() << " seconds.\n";

    // build elasticity problem
    chrono.Clear();
    chrono.Start();

    H1_FECollection fec(feorder+1, nDimensions);
    FiniteElementSpace fes(pmesh.get(), &fec, nDimensions, Ordering::byNODES);

    Array<int> ess_tdof_list;
//    fes.GetEssentialTrueDofs(ess_attr, ess_tdof_list);

    LinearForm b(&fes);
    b.AddDomainIntegrator(new VectorDomainLFIntegrator(f));
    b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(g));
    b.Assemble();

    GridFunction x(&fes);
#if MFEM_VERSION >= 40000
    x.ProjectBdrCoefficient(ubdrv, ess_attr);
#else
    x.ProjectBdrCoefficientTangent(ubdrv, ess_attr);
    #warning "Using ProjectBdrCoefficientTangent() in place of ProjectBdrCoefficient() since MFEM version < 4.0!"
#endif

    BilinearForm a(&fes);
    a.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
    a.Assemble();

    auto A = make_unique<SparseMatrix>();
    auto B = make_unique<Vector>();
    Vector X;
    a.FormLinearSystem(ess_tdof_list, x, b, *A, X, *B);

    chrono.Stop();
    if (myid == 0)
        std::cout << "Timing ELEM_AGG_LEVEL " << 0 << ": "
                  << "Assembly done in " << chrono.RealTime()
                  << " seconds.\n";
    timings(0, ASSEMBLY) = chrono.RealTime();

    // solve fine problem
/*    Array<int> rc_starts(3);
    rc_starts[0] = 0;
    rc_starts[2] = rc_starts[1] = A->Height();
    HypreParMatrix pA(comm, A->Height(), A->Width(),
                      rc_starts.GetData(), rc_starts.GetData(), A.get());
    HypreBoomerAMG amg(pA);
    amg.SetSystemsOptions(nDimensions, true);
    HyprePCG hpcg(pA);
    hpcg.SetTol(rtol);
    hpcg.SetMaxIter(max_num_iter);
    hpcg.SetPrintLevel(2);
    hpcg.SetPreconditioner(amg);
    hpcg.Mult(*B, X);

    a.RecoverFEMSolution(X, b, x);

    if (do_visualize)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock << "parallel " << num_procs << " " << myid << "\n";
        sol_sock.precision(8);
        sol_sock << "solution\n" << *pmesh << x << std::flush;
    }
*/
    // pretest constructs
    jform = 0;

    auto lift = make_unique<GridFunction>(&fes);
#if MFEM_VERSION >= 40000
    lift->ProjectBdrCoefficient(ubdrv, ess_attr);
#else
    lift->ProjectBdrCoefficientTangent(ubdrv, ess_attr);
    #warning "Using ProjectBdrCoefficientTangent() in place of ProjectBdrCoefficient() since MFEM version < 4.0!"
#endif

    DenseMatrix errors_L2_2(nLevels, nLevels);
    errors_L2_2 = 0.0;
    Vector norm_L2_2(nLevels);
    norm_L2_2 = 0.;

    DenseMatrix errors_energy_2(nLevels, nLevels);
    errors_energy_2 = 0.0;
    Vector norm_energy_2(nLevels);
    norm_energy_2 = 0.;

    Array<int> iter(nLevels);
    iter = 0;
    Array<int> ndofs(nLevels);
    ndofs = 0;
    Array<int> nnz(nLevels);
    nnz = 0;

    double tdiff;

    Array<SparseMatrix *> allP(nLevels-1);
    std::vector<unique_ptr<SparseMatrix>> allPb(nLevels-1);
    Array<SparseMatrix *> allD(nLevels);

    for (int i = 0; i < nLevels - 1; ++i)
    {
        allP[i] = sequence[i]->GetP(jform);
        Array<int> ro(nDimensions+1), co(nDimensions+1);
        ro[0] = 0;
        co[0] = 0;
        for (int k(1); k <= nDimensions; ++k)
        {
            ro[k] = ro[k-1] + allP[i]->Height();
            co[k] = co[k-1] + allP[i]->Width(); 
        }
        BlockMatrix PB(ro, co);
        for (int k(0); k < nDimensions; ++k)
            PB.SetBlock(k, k, allP[i]);
        allPb[i].reset(PB.CreateMonolithic());
    }

    for (int i = 0; i < nLevels; ++i)
        allD[i] = sequence[i]->GetDerivativeOperator(jform);

    std::vector<unique_ptr<SparseMatrix>> Ml(nLevels);
    std::vector<unique_ptr<SparseMatrix>> Mlb(nLevels);
    std::vector<unique_ptr<SparseMatrix>> Alb(nLevels);

    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        Array<int> ro(nDimensions+1);
        Ml[k] = sequence[k]->ComputeMassOperator(jform);
        ro[0] = 0;
        for (int i(1); i <= nDimensions; ++i)
            ro[i] = ro[i-1] + Ml[k]->Height();
        BlockMatrix MB(ro);
        for (int i(0); i < nDimensions; ++i)
            MB.SetBlock(i, i, Ml[k].get());
        Mlb[k].reset(MB.CreateMonolithic());
        if (0 == k)
            Alb[k] = std::move(A);
        else
            Alb[k].reset(RAP(*(allPb[k-1]), *(Alb[k-1]), *(allPb[k-1])));
        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k << ": "
                      << "Assembly done in " << tdiff
                      << " seconds.\n";
        timings(k, ASSEMBLY) += tdiff;
    }

    std::vector<unique_ptr<Vector>> rhs(nLevels);
    std::vector<unique_ptr<Vector>> ess_data(nLevels);
    rhs[0] = std::move(B);
    ess_data[0]= std::move(lift);
    for (int i = 0; i < nLevels-1; ++i)
    {
        rhs[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(jform)*nDimensions);
        ess_data[i+1] =
            make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(jform)*nDimensions);
        allPb[i]->MultTranspose(*(rhs[i]), *(rhs[i+1]));
        sequence[i]->GetPi(jform)->ComputeProjector();
        Array<int> ro(nDimensions+1), co(nDimensions+1);
        ro[0] = 0;
        co[0] = 0;
        for (int k(1); k <= nDimensions; ++k)
        {
            ro[k] = ro[k-1] + sequence[i]->GetPi(jform)->GetProjectorMatrix().Height();
            co[k] = co[k-1] + sequence[i]->GetPi(jform)->GetProjectorMatrix().Width();
        }
        BlockMatrix PiB(ro, co);
        for (int k(0); k < nDimensions; ++k)
            PiB.SetBlock(k, k, &(sequence[i]->GetPi(jform)->GetProjectorMatrix()));
        SparseMatrix *Pib = PiB.CreateMonolithic();
        Pib->Mult(*(ess_data[i]), *(ess_data[i+1]));
        delete Pib;
    }

    std::vector<unique_ptr<Vector>> sol(nLevels);
    std::vector<unique_ptr<Vector>> help(nLevels);

    for (int k(0); k < nLevels; ++k)
    {
        sol[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(jform)*nDimensions);
        *(sol[k]) = 0.;
        help[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(jform)*nDimensions);
        *(help[k]) = 0.;
    }

    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();

        unique_ptr<HypreParMatrix> pA;
        Array<int> rc_starts(3);
        rc_starts[0] = 0;
        {
            const int nlocdofs = Alb[k]->Height();
            const int nlocvardofs = sequence[k]->GetNumberOfDofs(jform);
            elag_assert(nlocvardofs*nDimensions == nlocdofs);
            Array<int> marker(nlocvardofs);
            marker = 0;
            sequence[k]->GetDofHandler(jform)->MarkDofsOnSelectedBndr(
                ess_attr, marker);

            for (int i(0); i < nDimensions; ++i)
                for (int mm = 0; mm < nlocvardofs; ++mm)
                    if (marker[mm])
                        Alb[k]->EliminateRowCol(
                            mm + i*nlocvardofs,
                            ess_data[k]->operator()(mm + i*nlocvardofs),
                            *(rhs[k]));
            rc_starts[2] = rc_starts[1] = nlocdofs;
            pA = make_unique<HypreParMatrix>(comm, Alb[k]->Height(), Alb[k]->Width(),
                     rc_starts.GetData(), rc_starts.GetData(), Alb[k].get());
        }

        elag_assert(rhs[k]->Size() == pA->Height());
        elag_assert(sol[k]->Size() == pA->Height());

        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k
                      << ": Assembly done in " << tdiff << " seconds."
                      << std::endl;
        timings(k, ASSEMBLY) += tdiff;

        ndofs[k] = pA->GetGlobalNumRows();
        nnz[k] = pA->NNZ();

        chrono.Clear();
        chrono.Start();
        HypreBoomerAMG prec(*pA);
#if MFEM_VERSION >= 40200
        prec.SetSystemsOptions(nDimensions, true);
#else
        #warning "Not using a systems AMG since MFEM version < 4.2!"
#endif
        Vector tmp1(pA->Height()), tmp2(pA->Height());
        tmp1 = 1.; tmp2 = 2.;
        prec.Mult(tmp1, tmp2);
        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing LEVEL " << k << ": "
                      << "Preconditioner done in " << tdiff
                      << " seconds.\n";
        timings(k, PRECONDITIONER) = tdiff;

        CGSolver pcg(comm);
        pcg.SetPrintLevel(print_iter);
        pcg.SetMaxIter(max_num_iter);
        pcg.SetRelTol(rtol);
        pcg.SetAbsTol(atol);
        pcg.SetOperator(*pA);
        pcg.SetPreconditioner(prec);
        chrono.Clear();
        chrono.Start();
        pcg.Mult(*(rhs[k]),*(sol[k]));
        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing LEVEL " << k << ": " << "Solver done in "
                      << tdiff << " seconds.\n";
        timings(k,SOLVER) = tdiff;

        if (myid == 0)
        {
            if (pcg.GetConverged())
                std::cout << "PCG converged in " << pcg.GetNumIterations()
                          << "iterations with a final residual norm "
                          << pcg.GetFinalNorm() << std::endl;
            else
                std::cout << "PCG did not converge in "
                          << pcg.GetNumIterations() << " iterations. "
                          << "Final residual norm is " << pcg.GetFinalNorm()
                          << std::endl;
        }

        iter[k] = pcg.GetNumIterations();

        //ERROR NORMS
        {
            *(help[k]) = *(sol[k]);
            for (int j = k; j > 0; --j)
                allPb[j-1]->Mult(*(help[j]), *(help[j-1]));

            norm_L2_2(k) = Mlb[k]->InnerProduct(*(sol[k]), *(sol[k]));
            norm_energy_2(k) = Alb[k]->InnerProduct(*(sol[k]), *(sol[k]));

            for (int j(0); j < k; ++j)
            {
                PARELAG_TEST_FOR_EXCEPTION(
                    help[j]->Size() != sol[j]->Size(),
                    std::runtime_error,
                    "main(): Sizes don't match.");

                const int size = sol[j]->Size();
                Vector u_H(help[j]->GetData(),size);
                Vector u_h(sol[j]->GetData(),size);
                Vector u_diff(size);
                u_diff = 0.;

                subtract(u_H, u_h, u_diff);

                if (do_visualize && 0 == j)
                {
                    GridFunction x(&fes);
                    x = u_diff;
                    char vishost[] = "localhost";
                    int  visport   = 19916;
                    socketstream sol_sock(vishost, visport);
                    sol_sock << "parallel " << num_procs << " " << myid << "\n";
                    sol_sock.precision(8);
                    sol_sock << "solution\n" << *pmesh << x << std::flush;
                    sol_sock << "plot_caption " << "'Error " << j << "-" << k
                             << "'" << std::flush;
                }

                errors_L2_2(k,j) = Mlb[j]->InnerProduct(u_diff, u_diff);
                errors_energy_2(k,j) = Alb[j]->InnerProduct(u_diff, u_diff);
            }
        }

        if (do_visualize)
        {
            GridFunction x(&fes);
            x = *(help[0]);
            char vishost[] = "localhost";
            int  visport   = 19916;
            socketstream sol_sock(vishost, visport);
            sol_sock << "parallel " << num_procs << " " << myid << "\n";
            sol_sock.precision(8);
            sol_sock << "solution\n" << *pmesh << x << std::flush;
            sol_sock << "plot_caption " << "'solution " << k << "'" << std::flush;
        }
    }

    DenseMatrix errors_L2(nLevels, nLevels);
    errors_L2 = 0.;
    Vector norm_L2(nLevels);
    norm_L2 = 0.;
    DenseMatrix errors_energy(nLevels, nLevels);
    errors_energy = 0.;
    Vector norm_energy(nLevels);
    norm_energy = 0.;

    MPI_Reduce(errors_L2_2.Data(),errors_L2.Data(),
               errors_L2.Height()*errors_L2.Width(),
               MPI_DOUBLE,MPI_SUM,0,comm);
    MPI_Reduce(norm_L2_2.GetData(),norm_L2.GetData(),norm_L2.Size(),
               MPI_DOUBLE,MPI_SUM,0,comm);
    MPI_Reduce(errors_energy_2.Data(),errors_energy.Data(),
               errors_energy.Height()*errors_energy.Width(),
               MPI_DOUBLE,MPI_SUM,0,comm);
    MPI_Reduce(norm_energy_2.GetData(),norm_energy.GetData(),norm_energy.Size(),
               MPI_DOUBLE,MPI_SUM,0,comm);

    std::transform(errors_L2.Data(),
                   errors_L2.Data()+errors_L2.Height()*errors_L2.Width(),
                   errors_L2.Data(),(double(*)(double)) sqrt);
    std::transform(norm_L2.GetData(),norm_L2.GetData()+norm_L2.Size(),
                   norm_L2.GetData(),(double(*)(double)) sqrt);
    std::transform(errors_energy.Data(),
                   errors_energy.Data()+errors_energy.Height()*errors_energy.Width(),
                   errors_energy.Data(),(double(*)(double)) sqrt);
    std::transform(norm_energy.GetData(),norm_energy.GetData()+norm_energy.Size(),
                   norm_energy.GetData(),(double(*)(double)) sqrt);

    if (myid == 0)
    {
        std::cout << std::endl << "{\n";
        constexpr int w = 14;
        std::cout << "%level" << std::setw(w) << "size" << std::setw(w)
                  << "nnz" << std::setw(w) << "nit" << std::setw(w)
                  << "Topology" << std::setw(w) << "TSpaces" << std::setw(w)
                  << "Assembly" << std::setw(w)
                  << "Preconditioner" << std::setw(w) << "Solver\n";
        for (int i(0); i < nLevels; ++i)
            std::cout << i << std::setw(w) << ndofs[i] << std::setw(w)
                      << nnz[i] << std::setw(w) << iter[i] << std::setw(w)
                      << timings(i,TOPOLOGY) << std::setw(w)
                      << timings(i,SPACES) << std::setw(w)
                      << timings(i,ASSEMBLY) << std::setw(w)
                      << timings(i,PRECONDITIONER) << std::setw(w)
                      << timings(i,SOLVER) << std::endl << "}\n";

        std::cout << std::endl << "{" << std::endl
                  << "% || uh - uH ||\n";
        errors_L2.PrintMatlab(std::cout);
        std::cout << "% || uh ||\n";
        norm_L2.Print(std::cout, nLevels);
        std::cout << "% || uh - uH ||_A\n";
        errors_energy.PrintMatlab(std::cout);
        std::cout << "% || uh ||_A\n";
        norm_energy.Print(std::cout, nLevels);
        std::cout << "}\n";
    }

    return EXIT_SUCCESS;
}
