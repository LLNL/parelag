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

//                       Elag Upscaling - Parallel Version
//
// solves a mixed Darcy problem with unstructured coarsening, using
// an H1-L2 solver.

#include <fstream>
#include <sstream>

#include <mpi.h>

#include "elag.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

enum {ASSEMBLY = 0, PRECONDITIONER, SOLVER};

const int NSTAGES = 3;
const char * stage_names[] = {"ASSEMBLY", "PRECONDITIONER", "SOLVER"};

int main (int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session sess(argc,argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    if (myid == 0)
    {
        std::cout << "------" << std::endl;
        for (int i=0; i<argc; ++i)
            std::cout << argv[i] << " ";;
        std::cout << std::endl << "------" << std::endl;
    }

#ifndef ParELAG_ENABLE_PETSC
    if (!myid) std::cerr << "This test requires PETSc" << std::endl;
    return 1;
#else

    // Program options
    const char* meshfile_c = "mesh.mesh3d";
    const char* petsc_opts_c = "";
    int ser_ref_levels = 0;
    int par_ref_levels = 2;
    int coarseningFactor = 8;
    int feorder = 0;
    int upscalingOrder = 0;
    int aggressiveLevels = 1;
    bool do_visualize = true;
    bool unassembled = false;
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
    args.AddOption(&coarseningFactor, "--coarsening-factor", "--coarsening-factor",
                   "Factor by which to coarsen mesh at each level.");
    args.AddOption(&aggressiveLevels, "--aggressive-levels", "--aggressive-levels",
                   "Number of levels of aggressive coarsening.");
    args.AddOption(&do_visualize, "-v", "--do-visualize", "-nv", "--no-visualize",
                   "Do interactive GLVis visualization.");
    args.AddOption(&petsc_opts_c, "--petsc_opts", "--petsc_opts",
                   "Options file for PETSc solvers etc.");
    args.AddOption(&unassembled, "--unassembled", "--unassembled",
                   "--assembled", "--assembled",
                   "Whether to use PETSc unassembled or fully assembled matrix format.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(std::cout);
        MPI_Finalize();
        return 1;
    }
    PARELAG_ASSERT(args.Good());
    std::string meshfile(meshfile_c);
    std::string petsc_opts(petsc_opts_c);

    Operator::Type petsc_tid;
    if (unassembled)
        petsc_tid = Operator::PETSC_MATIS;
    else
        petsc_tid = Operator::PETSC_MATAIJ;
    if (!petsc_opts.size())
    {
        PetscInitialize(NULL, NULL, NULL, NULL);
    }
    else
    {
        PetscInitialize(NULL, NULL, petsc_opts.c_str(), NULL);
    }

    // default linear solver options
    constexpr int max_num_iter = 5000;
    constexpr double rtol = 1e-6;
    constexpr double atol = 1e-12;

    if (myid == 0)
    {
        std::cout << "Read mesh " << meshfile << "\n";
        std::cout << "Refine mesh in serial " << ser_ref_levels << " times.\n";
        std::cout << "Refine mesh in parallel " << par_ref_levels << " times.\n";
        std::cout << "FE order " << feorder << "\n";
        std::cout << "Upscaling order " << upscalingOrder << "\n";
        std::cout << "Coarsening Factor " << coarseningFactor << "\n";
        std::cout << "Number of Aggressive Levels " << aggressiveLevels << "\n";
    }

    StopWatch chrono;
    std::shared_ptr<ParMesh> pmesh;
    {
        // 2. Read the (serial) mesh from the given mesh file and
        // uniformly refine it.
        std::unique_ptr<Mesh> mesh;
        std::ifstream imesh(meshfile.c_str());
        if (imesh)
        {
            mesh = make_unique<Mesh>(imesh, 1, 1);
            imesh.close();
        }
        else
        {
            if (myid == 0)
            {
                std::cerr << "\nCannot open mesh file " << meshfile
                          << ", falling back to default behavior." << std::endl;
                std::cout << "Generating cube mesh with 8 hexahedral elements.\n";
            }
            mesh = make_unique<Mesh>(2, 2, 2, Element::HEXAHEDRON, 1);
        }

        for (int l=0; l<ser_ref_levels; l++)
            mesh->UniformRefinement();

        // I want to get rid of this because the mesh knows how to
        // partition itself. This does something marginally different
        // than when mfem::Mesh calls metis, so the numbers output at
        // the end will not be what they were, but they are still ok.
        MetisGraphPartitioner partitioner;
        Array<int> partitioning(mesh->GetNE());
        partitioner.setFlags(MetisGraphPartitioner::KWAY);// BISECTION
        partitioner.setOption(METIS_OPTION_SEED, 0);// Fix the seed
        partitioner.setOption(METIS_OPTION_CONTIG,1);// Contiguous partitions
        partitioner.doPartition(mesh->ElementToElementTable(),
                                num_procs, partitioning);
        pmesh = make_shared<mfem::ParMesh>(comm, *mesh, partitioning);

        // The above should just be:
        // pmesh = make_shared<mfem::ParMesh>(comm, *mesh);
        mesh.reset();

        for (int i = 0; i < par_ref_levels; ++i)
            pmesh->UniformRefinement();
    }
    const int nDimensions = pmesh->Dimension();

    ConstantCoefficient coeffL2(1.);
    ConstantCoefficient coeffHdiv(1.);

    Array<int> level_NE(1);
    int ne = pmesh->GetNE();
    level_NE[0] = ne;
    for (int i=0; i<aggressiveLevels; ++i)
    {
        ne /= (coarseningFactor)*(coarseningFactor);
        level_NE.Append( fmax(ne,1) );
        if (ne < coarseningFactor)
            break;
    }
    while (ne > coarseningFactor)
    {
        ne /= coarseningFactor;
        level_NE.Append( fmax(ne,1) );
    }
    std::cout << "level_NE:" << std::endl;
    level_NE.Print(std::cout);

    int nLevels=level_NE.Size();
    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    chrono.Clear();
    chrono.Start();
    topology[0] = make_shared<AgglomeratedTopology>(pmesh, 1);

    MetisGraphPartitioner partitioner;
    Array<int> partitioning;
    partitioner.setFlags(MetisGraphPartitioner::KWAY );// BISECTION
    partitioner.setOption(METIS_OPTION_SEED, 0);// Fix the seed
    partitioner.setOption(METIS_OPTION_CONTIG,1);// contiguous partitions
        partitioner.setOption(METIS_OPTION_MINCONN,1);
    partitioner.setUnbalanceToll(1.05);

    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        partitioning.SetSize( level_NE[ilevel] );
        partitioner.doPartition(
            *(topology[ilevel]->LocalElementElementTable()),
            topology[ilevel]->Weight(at_elem),level_NE[ilevel+1], partitioning);
        topology[ilevel+1] =
            topology[ilevel]->CoarsenLocalPartitioning(partitioning,false,false);
    }

    chrono.Stop();
    if (myid == 0)
        std::cout << "Timing ELEM_AGG: Mesh Agglomeration done in "
                  << chrono.RealTime() << " seconds \n";

    //-----------------------------------------------------//

    std::vector<shared_ptr<DeRhamSequence> > sequence(topology.size());
    sequence[0] = make_shared<DeRhamSequence3D_FE>(
        topology[0], pmesh.get(), feorder);

    DeRhamSequenceFE* DRSequence_FE = sequence[0]->FemSequence();

    sequence[0]->FemSequence()->ReplaceMassIntegrator(
        at_elem, 3, make_unique<MassIntegrator>(coeffL2), false);
    sequence[0]->FemSequence()->ReplaceMassIntegrator(
        at_elem, 2, make_unique<VectorFEMassIntegrator>(coeffHdiv), true);

    // okay, people, we have coeffL2 and L2coeff ???
    Array<Coefficient *> L2coeff;
    Array<VectorCoefficient *> Hdivcoeff;
    fillVectorCoefficientArray(nDimensions, upscalingOrder, Hdivcoeff);
    fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);

    std::vector<std::unique_ptr<MultiVector>>
        targets(sequence[0]->GetNumberOfForms());

    int jform(0);
    ++jform;
    ++jform;
    targets[jform] =
        sequence[0]->FemSequence()->InterpolateVectorTargets(jform, Hdivcoeff);
    ++jform;
    targets[jform] =
        sequence[0]->FemSequence()->InterpolateScalarTargets(jform, L2coeff);
    ++jform;

    freeCoeffArray(L2coeff);
    freeCoeffArray(Hdivcoeff);

    Array<MultiVector *> targets_in(targets.size());
    for (int ii = 0; ii < targets_in.Size(); ++ii)
        targets_in[ii] = targets[ii].get();

    const int jFormStart = nDimensions-1;
    sequence[0]->SetjformStart(jFormStart);
    sequence[0]->SetTargets(targets_in);

    chrono.Clear();
    chrono.Start();
    constexpr double tolSVD = 1e-9;
    for (int i(0); i < nLevels-1; ++i)
    {
        sequence[i]->SetSVDTol(tolSVD);
        StopWatch chronoInterior;
        chronoInterior.Clear();
        chronoInterior.Start();
        sequence[i+1] = sequence[i]->Coarsen();
        chronoInterior.Stop();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL" << i << ": Coarsening done in "
                      << chronoInterior.RealTime() << " seconds \n";
    }
    chrono.Stop();

    if (myid == 0)
        std::cout << "Timing ELEM_AGG: Coarsening done in "
                  << chrono.RealTime() << " seconds \n";

    const int uform = pmesh->Dimension() - 1;
    const int pform = pmesh->Dimension();

    // testUpscalingHdiv(sequence);
    FiniteElementSpace * ufespace = DRSequence_FE->GetFeSpace(uform);
    FiniteElementSpace * pfespace = DRSequence_FE->GetFeSpace(pform);

    LinearForm b(ufespace);
    ConstantCoefficient fbdr(0.);
    b.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fbdr));
    b.Assemble();

    LinearForm q(pfespace);
    ConstantCoefficient source(1.);
    q.AddDomainIntegrator(new DomainLFIntegrator(source));
    q.Assemble();

    DenseMatrix u_errors_L2_2(nLevels, nLevels);
    u_errors_L2_2 = 0.0;
    Vector u_norm_L2_2(nLevels);
    u_norm_L2_2 = 0.;
    DenseMatrix p_errors_L2_2(nLevels, nLevels);
    p_errors_L2_2 = 0.0;
    Vector p_norm_L2_2(nLevels);
    p_norm_L2_2 = 0.;

    DenseMatrix errors_div_2(nLevels, nLevels);
    errors_div_2 = 0.0;
    Vector norm_div_2(nLevels);
    norm_div_2 = 0.;

    DenseMatrix timings(nLevels, NSTAGES);
    timings = 0.0;

    Array<int> iter(nLevels);
    iter = 0;
    Array<int> ndofs(nLevels);
    ndofs = 0;
    Array<int> nnzs(nLevels);
    nnzs = 0;

    double tdiff;

    Array<SparseMatrix *> allPu(nLevels-1);
    Array<SparseMatrix *> allPp(nLevels-1);
    Array<SparseMatrix *> allD(nLevels);

    for (int i = 0; i < nLevels - 1; ++i)
    {
        allPu[i] = sequence[i]->GetP(uform);
        allPp[i] = sequence[i]->GetP(pform);
    }

    for (int i = 0; i < nLevels; ++i)
        allD[i] = sequence[i]->GetDerivativeOperator(uform);

    std::vector<unique_ptr<mfem::SparseMatrix>> Ml(nLevels);
    std::vector<unique_ptr<mfem::SparseMatrix>> Wl(nLevels);

    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        Ml[k] = sequence[k]->ComputeMassOperator(uform);
        Wl[k] = sequence[k]->ComputeMassOperator(pform);
        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k
                      << ": Assembly mass matrices done in " << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;
    }

    std::vector<Array<int>> blockOffsets(nLevels);
    for (int k(0); k < nLevels; ++k)
    {
        blockOffsets[k].SetSize(3);
        int * p = blockOffsets[k].GetData();
        p[0] = 0;
        p[1] = sequence[k]->GetNumberOfDofs(uform);
        p[2] = p[1] + sequence[k]->GetNumberOfDofs(pform);
    }

    std::vector<std::unique_ptr<BlockVector>> rhs(nLevels);
    rhs[0] = make_unique<BlockVector>(blockOffsets[0]);
    rhs[0]->GetBlock(0) = b;
    rhs[0]->GetBlock(1) = q;

    for (int i = 0; i < nLevels-1; ++i)
    {
        rhs[i+1] = make_unique<BlockVector>(blockOffsets[i+1]);
        allPu[i]->MultTranspose(rhs[i]->GetBlock(0),rhs[i+1]->GetBlock(0));
        allPp[i]->MultTranspose(rhs[i]->GetBlock(1),rhs[i+1]->GetBlock(1));
    }

    std::vector<unique_ptr<BlockVector>> sol(nLevels);
    std::vector<unique_ptr<BlockVector>> help(nLevels);

    for (int k(0); k<nLevels; ++k)
    {
        sol[k] = make_unique<BlockVector>(blockOffsets[k]);
        help[k] = make_unique<BlockVector>(blockOffsets[k]);
        *(help[k]) = 0.;
    }

    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        SparseMatrix * M = Ml[k].get();
        SparseMatrix * W = Wl[k].get();
        SparseMatrix * D = allD[k];
        auto B = ToUnique(Mult(*W, *D));
        auto Bt = ToUnique(Transpose(*B));

        const SharingMap & l2_dofTrueDof(
            sequence[k]->GetDofHandler(pform)->GetDofTrueDof());
        const SharingMap & hdiv_dofTrueDof(
            sequence[k]->GetDofHandler(uform)->GetDofTrueDof());

        Array<int> trueBlockOffsets(3);
        trueBlockOffsets[0] = 0;
        trueBlockOffsets[1] = hdiv_dofTrueDof.GetTrueLocalSize();
        trueBlockOffsets[2] =
            trueBlockOffsets[1] + l2_dofTrueDof.GetTrueLocalSize();

        BlockVector prhs(trueBlockOffsets);
        hdiv_dofTrueDof.Assemble(rhs[k]->GetBlock(0), prhs.GetBlock(0));
        l2_dofTrueDof.Assemble(rhs[k]->GetBlock(1), prhs.GetBlock(1));

        auto pM = AssemblePetsc(hdiv_dofTrueDof, *M, hdiv_dofTrueDof,petsc_tid);
        auto pB = AssemblePetsc(l2_dofTrueDof, *B, hdiv_dofTrueDof,petsc_tid);
        auto pBt = AssemblePetsc(hdiv_dofTrueDof, *Bt, l2_dofTrueDof,petsc_tid);

        std::unique_ptr<PetscParMatrix> op;
        {
            // We construct the block operator as in Darcy case
            BlockOperator bop(trueBlockOffsets);
            bop.owns_blocks = 0;
            bop.SetBlock(0,0, pM.get());
            bop.SetBlock(0,1, pBt.get());
            bop.SetBlock(1,0, pB.get());

            // And convert the block operator to a PetscParMatrix here
            if (unassembled)
            {
                op = make_unique<PetscParMatrix>(pM->GetComm(), &bop,
                                                 Operator::PETSC_MATIS);
            }
            else
            {
                op = make_unique<PetscParMatrix>(pM->GetComm(), &bop,
                                                 Operator::PETSC_MATAIJ);
            }
        }

        std::unique_ptr<PetscPreconditioner> prec;

        // different prefix for each level
        //std::string prefix = "parelag_level" + std::to_string(k) + "_";
        // same prefix for all levels
        std::string prefix = "parelag_levels_";
        if (unassembled)
        {
            // For saddle point problems, we need to provide BDDC the list of
            // essential and boundary dofs.
            // Since the velocity_space is the only space that may have boundary
            // dofs and it is ordered first then the pressure space, we don't
            // need any offset when specifying the dofs.
            Array<int> bdr_dof;
            bool local = true;
            if (pmesh->bdr_attributes.Size())
            {
               Array<int> bdr(pmesh->bdr_attributes.Max());
               bdr = 1;
               bdr_dof.SetSize(hdiv_dofTrueDof.GetLocalSize(),0);
               sequence[k]->GetDofHandler(uform)->MarkDofsOnSelectedBndr(bdr, bdr_dof);
               bdr_dof.SetSize(hdiv_dofTrueDof.GetLocalSize() + l2_dofTrueDof.GetLocalSize(),0);
#if 0
               Array<int> bdr_tdof(hdiv_dofTrueDof.GetTrueLocalSize());
               hdiv_dofTrueDof.IgnoreNonLocal(bdr_dof,bdr_tdof);
               bdr_dof.SetSize(bdr_tdof.Size());
               int cum = 0;
               for (int i = 0; i<bdr_tdof.Size(); i++)
               {
                  if (bdr_tdof[i]) bdr_dof[cum++] = i;
               }
               bdr_dof.SetSize(cum);
               local = false;
#endif
            }
            else
            {
               MFEM_ABORT("Need to know the boundary dofs");
            }
  
            PetscBDDCSolverParams opts;
            // boundary dofs are specified in local vdofs ordering
            opts.SetNatBdrDofs(&bdr_dof,local);
            // See also command line options .petsc_rc_darcy_bddc
            prec = make_unique<PetscBDDCSolver>(pM->GetComm(),*op,opts,prefix.c_str());
        }
        else
        {
            // With PETSc, we can construct the (same) block-diagonal solver with
            // command line options (see .petsc_rc_darcy_fieldsplit)
            prec = make_unique<PetscFieldSplitSolver>(pM->GetComm(),*op,prefix.c_str());
        }

        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k
                      << ": Assembly Darcy Operator and Preconditioner done in "
                      << tdiff << "s. \n";
        timings(k, ASSEMBLY) += tdiff;

        ndofs[k] = pM->GetGlobalNumRows() + pB->GetGlobalNumRows();
        nnzs[k] = pM->NNZ() + pB->NNZ();
        // Solver
        {
            chrono.Clear();
            chrono.Start();

            {
                BlockVector x( trueBlockOffsets );
                BlockVector y( trueBlockOffsets );
                x = 1.0; y = 0.;
                prec->Mult(x,y);
            }

            chrono.Stop();
            tdiff = chrono.RealTime();
            if (myid == 0)
                std::cout << "Timing PRECONDITIONER_LEVEL " << k
                          << ": Preconditioner Computed "
                          << tdiff << "s. \n";
            timings(k,PRECONDITIONER) = tdiff;

            BlockVector psol( trueBlockOffsets );
            psol = 0.;
            std::string solvertype;
            std::unique_ptr<PetscLinearSolver> solver;
            if (unassembled)
            {
               // We can use conjugate gradients to solve the problem
               solver = make_unique<PetscPCGSolver>(MPI_COMM_WORLD);
               solvertype = "PCG";
            }
            else
            {
               solver = make_unique<PetscLinearSolver>(MPI_COMM_WORLD);
               solvertype = "MINRES";
            }
            solver->SetOperator(*op);
            solver->SetPreconditioner(*prec);
            solver->SetAbsTol(atol);
            solver->SetRelTol(rtol);
            solver->SetMaxIter(max_num_iter);
            chrono.Clear();
            chrono.Start();
            solver->Mult(prhs, psol );
            chrono.Stop();
            tdiff = chrono.RealTime();
            if (myid == 0)
                std::cout << "Timing " << solvertype << "_LEVEL " << k
                          << ": Solver done in " << tdiff << "s. \n";
            timings(k,SOLVER) = tdiff;

            if (myid == 0)
            {
                if (solver->GetConverged())
                    std::cout << solvertype << " converged in "
                              << solver->GetNumIterations()
                              << " with a final residual norm "
                              << solver->GetFinalNorm() << "\n";
                else
                    std::cout << solvertype << " did not converge in "
                              << solver->GetNumIterations()
                              << ". Final residual norm is "
                              << solver->GetFinalNorm() << "\n";
            }
            hdiv_dofTrueDof.Distribute(psol.GetBlock(0), sol[k]->GetBlock(0));
            l2_dofTrueDof.Distribute(psol.GetBlock(1), sol[k]->GetBlock(1));
            iter[k] = solver->GetNumIterations();
        }

        // ERROR NORMS
        {
            *(help[k]) = *(sol[k]);
            for (int j=k; j>0; --j)
            {
                allPu[j-1]->Mult(help[j]->GetBlock(0), help[j-1]->GetBlock(0));
                allPp[j-1]->Mult(help[j]->GetBlock(1), help[j-1]->GetBlock(1));
            }

            u_norm_L2_2(k) =
                Ml[k]->InnerProduct(sol[k]->GetBlock(0),sol[k]->GetBlock(0));
            p_norm_L2_2(k) =
                Wl[k]->InnerProduct(sol[k]->GetBlock(1), sol[k]->GetBlock(1));
            Vector dsol(allD[k]->Size());
            allD[k]->Mult(sol[k]->GetBlock(0), dsol);
            norm_div_2(k) = Wl[k]->InnerProduct(dsol, dsol);

            for (int j(0); j < k; ++j)
            {
                if (help[j]->Size() != sol[j]->Size() ||
                    sol[j]->GetBlock(0).Size() != allD[j]->Width() )
                    mfem_error("size don't match \n");

                const int usize = sol[j]->GetBlock(0).Size();
                const int psize = sol[j]->GetBlock(1).Size();
                const int dsize = allD[j]->Size();
                Vector u_H(help[j]->GetData(), usize);
                Vector u_h(sol[j]->GetData(), usize);
                Vector p_H(help[j]->GetData(), psize);
                Vector p_h(sol[j]->GetData(), psize);
                Vector u_diff(usize), du_diff(dsize), p_diff(psize);
                u_diff = 0.; du_diff = 0.; p_diff = 0.;

                subtract(u_H, u_h, u_diff);
                allD[j]->Mult(u_diff, du_diff);
                subtract(p_H, p_h, p_diff);

                u_errors_L2_2(k,j) = Ml[j]->InnerProduct(u_diff, u_diff);
                errors_div_2(k,j) = Wl[j]->InnerProduct(du_diff, du_diff);
                p_errors_L2_2(k,j) = Wl[j]->InnerProduct(p_diff, p_diff);
            }
        }

        // visualize solution
        if (1)
        {
            MultiVector u(sol[k]->GetData(), 1, sol[k]->GetBlock(0).Size());
            sequence[k]->show(uform, u);
            MultiVector p(sol[k]->GetBlock(1).GetData(), 1,
                          sol[k]->GetBlock(1).Size());
            MPI_Barrier(comm);
            sequence[k]->show(pform, p);
        }
    }

    OutputUpscalingTimings(ndofs, nnzs, iter, timings, stage_names);

    ReduceAndOutputUpscalingErrors(u_errors_L2_2, u_norm_L2_2,
                                   p_errors_L2_2, p_norm_L2_2,
                                   errors_div_2, norm_div_2);

    PetscFinalize();
    return EXIT_SUCCESS;
#endif
}
