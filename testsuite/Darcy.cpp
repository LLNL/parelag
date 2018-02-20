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
//
// Sample runs:  mpirun -np 4 ./Upscaling1Form --mesh meshfile

#include <fstream>
#include <sstream>

#include <mpi.h>

#include "elag.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

enum {ASSEMBLY = 0, PRECONDITIONER, SOLVER, NSTAGES};

void deformation3D(const Vector & in, Vector & out)
{
    out(1)= in(1) + .5*exp( in(2) );
    out(0) = in(0) + sin( out(1) );
}

void deformation2D(const Vector & in, Vector & out)
{
    out(0) = in(0) + sin( in(1) );
}

int main (int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    StopWatch chrono;

    // Program options
    const char* meshfile_c = "mesh.mesh3d";
    int ser_ref_levels = 0;
    int par_ref_levels = 2;
    int feorder = 0;
    int upscalingOrder = 0;
    bool deformation = false;
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
    args.AddOption(&deformation, "--deformation", "--deformation",
                   "--no-deformation", "--no-deformation",
                   "Apply artificial (curved) deformation to mesh.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(std::cout);
        MPI_Finalize();
        return 1;
    }
    PARELAG_ASSERT(args.Good());
    std::string meshfile(meshfile_c);

    // Default linear solver options
    constexpr int print_iter = 0;
    constexpr int max_num_iter = 500;
    constexpr double rtol = 1e-6;
    constexpr double atol = 1e-12;

    if (myid == 0)
    {
        std::cout << "Read mesh " << meshfile << "\n";
        std::cout << "Refine mesh in serial " << ser_ref_levels << " times.\n";
        std::cout << "Refine mesh in parallel " << par_ref_levels << " times.\n";
    }

    shared_ptr<ParMesh> pmesh;
    {
        // 2. Read the (serial) mesh from the given mesh file and
        // uniformely refine it.
        std::ifstream imesh(meshfile.c_str());
        if (!imesh)
        {
            std::cerr << "\nCan not open mesh file: " << meshfile << "\n\n";
            return EXIT_FAILURE;
        }
        auto mesh = make_unique<Mesh>(imesh, 1, 1);
        imesh.close();

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        pmesh = make_shared<ParMesh>(comm, *mesh);
    }

    const int nDimensions = pmesh->Dimension();

    const int nLevels = par_ref_levels+1;
    Array<int> level_nElements(nLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    level_nElements[0] = pmesh->GetNE();

    if (nDimensions == 3)
        pmesh->ReorientTetMesh();

    if (deformation)
    {
        if (nDimensions == 2)
            pmesh->Transform( deformation2D );
        else
            pmesh->Transform( deformation3D );
    }


    ConstantCoefficient coeffL2(1.);
    ConstantCoefficient coeffHdiv(1.);

    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    chrono.Clear();
    chrono.Start();
    topology[0] = make_shared<AgglomeratedTopology>(pmesh,1);

    MFEMRefinedMeshPartitioner partitioner(nDimensions);
    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    for (int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        Array<int> partitioning(
            topology[ilevel]->GetNumberLocalEntities(at_elem));
        partitioner.Partition(
            topology[ilevel]->GetNumberLocalEntities(at_elem),
            level_nElements[ilevel+1], partitioning);
        topology[ilevel+1] =
            topology[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);
    }
    chrono.Stop();
    if (myid == 0)
        std::cout <<"Timing ELEM_AGG: Mesh Agglomeration done in "
                  << chrono.RealTime() << " seconds.\n";

    //-----------------------------------------------------//

    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());
    sequence[0] = make_shared<DeRhamSequence3D_FE>(
        topology[0], pmesh.get(), feorder);

    DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();

    const int jFormStart = nDimensions-1;
    sequence[0]->SetjformStart(jFormStart);

    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 3, make_unique<MassIntegrator>(coeffL2), false);
    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 2, make_unique<VectorFEMassIntegrator>(coeffHdiv), true);

    // set up coefficients / targets
    sequence[0]->FemSequence()->SetUpscalingTargets(nDimensions, upscalingOrder);

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
            std::cout <<"Timing ELEM_AGG_LEVEL" << i << ": Coarsening done in "
                      << chronoInterior.RealTime() << " seconds.\n";
    }
    chrono.Stop();

    if (myid == 0)
        std::cout << "Timing ELEM_AGG: Coarsening done in " << chrono.RealTime()
                  << " seconds.\n";

    const int uform = nDimensions - 1;
    const int pform = nDimensions;

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

    std::vector<unique_ptr<SparseMatrix>> Ml(nLevels);
    std::vector<unique_ptr<SparseMatrix>> Wl(nLevels);

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
                      << ": Assembly done in " << tdiff << "s.\n";
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

    std::vector<unique_ptr<BlockVector>> rhs(nLevels);
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

    for (int k(0); k < nLevels; ++k)
    {
        sol[k] = make_unique<BlockVector>(blockOffsets[k]);
        help[k] = make_unique<BlockVector>(blockOffsets[k]);
        *(help[k]) = 0.;
    }

    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();

        // The blocks, managed here
        unique_ptr<HypreParMatrix> pM;
        unique_ptr<HypreParMatrix> pB;
        unique_ptr<HypreParMatrix> pBt;
        unique_ptr<HypreParMatrix> S;

        // The sharing maps and block offset computations
        const SharingMap & l2_dofTrueDof(
            sequence[k]->GetDofHandler(pform)->GetDofTrueDof());
        const SharingMap & hdiv_dofTrueDof(
            sequence[k]->GetDofHandler(uform)->GetDofTrueDof());

        Array<int> trueBlockOffsets(3);
        trueBlockOffsets[0] = 0;
        trueBlockOffsets[1] = hdiv_dofTrueDof.GetTrueLocalSize();
        trueBlockOffsets[2] =
            trueBlockOffsets[1] + l2_dofTrueDof.GetTrueLocalSize();
        {
            SparseMatrix * M = Ml[k].get();
            SparseMatrix * W = Wl[k].get();
            SparseMatrix * D = allD[k];
            auto B = ToUnique(Mult(*W, *D));
            auto Bt = ToUnique(Transpose(*B));

            pM = Assemble(hdiv_dofTrueDof, *M, hdiv_dofTrueDof);
            pB = Assemble(l2_dofTrueDof, *B, hdiv_dofTrueDof);
            pBt = Assemble(hdiv_dofTrueDof, *Bt, l2_dofTrueDof);

            // Setup Schur complement
            auto tmp = Assemble(hdiv_dofTrueDof, *Bt, l2_dofTrueDof);
            Vector diag(pM->Height());
            pM->GetDiag(diag);

            for (int i =0; i < diag.Size(); ++i)
                diag(i) = 1./diag(i);

            tmp->ScaleRows(diag);
            S = ToUnique(ParMult(pB.get(), tmp.get()));
        }

        BlockOperator op(trueBlockOffsets);
        op.owns_blocks = 0;
        op.SetBlock(0,0,pM.get());
        op.SetBlock(0,1,pBt.get());
        op.SetBlock(1,0,pB.get());

        // Setup right-hand side
        BlockVector prhs(trueBlockOffsets);
        hdiv_dofTrueDof.Assemble(rhs[k]->GetBlock(0), prhs.GetBlock(0));
        l2_dofTrueDof.Assemble(rhs[k]->GetBlock(1), prhs.GetBlock(1));

        // Setup solver and prec
        auto Mprec = make_unique<HypreDiagScale>(*pM);
        auto Sprec = make_unique<HypreBoomerAMG>(*S);

        BlockDiagonalPreconditioner prec(trueBlockOffsets);
        prec.owns_blocks = 0;
        prec.SetDiagonalBlock(0,Mprec.get());
        prec.SetDiagonalBlock(1,Sprec.get());

        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k
                      << ": Assembly done in " << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;

        ndofs[k] = pM->GetGlobalNumRows() + pB->GetGlobalNumRows();

        //Solver
        {
            chrono.Clear();
            chrono.Start();

            {
                BlockVector x(trueBlockOffsets);
                BlockVector y(trueBlockOffsets);
                x = 1.0; y = 0.;
                prec.Mult(x,y);
            }

            chrono.Stop();
            tdiff = chrono.RealTime();

            if (myid == 0)
                std::cout << "Timing PRECONDITIONER_LEVEL " << k
                          << ": Preconditioner Computed " << tdiff << "s.\n";
            timings(k,PRECONDITIONER) = tdiff;

            BlockVector psol(trueBlockOffsets);
            psol = 0.;
            MINRESSolver minres(comm);
            minres.SetPrintLevel(print_iter);
            minres.SetMaxIter(max_num_iter);
            minres.SetRelTol(rtol);
            minres.SetAbsTol(atol);
            minres.SetOperator(op );
            minres.SetPreconditioner( prec );
            chrono.Clear();
            chrono.Start();
            minres.Mult(prhs, psol );
            chrono.Stop();
            tdiff = chrono.RealTime();
            if (myid == 0)
                std::cout << "Timing MINRES_LEVEL " << k
                          << ": Solver done in " << tdiff << "s. \n";
            timings(k,SOLVER) = tdiff;

            if (myid == 0)
            {
                if (minres.GetConverged())
                    std::cout << "Minres converged in "
                              << minres.GetNumIterations()
                              << " with a final residual norm "
                              << minres.GetFinalNorm() << ".\n";
                else
                    std::cout << "Minres did not converge in "
                              << minres.GetNumIterations()
                              << ". Final residual norm is "
                              << minres.GetFinalNorm() << ".\n";
            }

            hdiv_dofTrueDof.Distribute(psol.GetBlock(0), sol[k]->GetBlock(0));
            l2_dofTrueDof.Distribute(psol.GetBlock(1), sol[k]->GetBlock(1));
            iter[k] = minres.GetNumIterations();
        }

        //ERROR NORMS
        {
            *(help[k]) = *(sol[k]);
            for (int j = k; j > 0; --j)
            {
                allPu[j-1]->Mult(help[j]->GetBlock(0),help[j-1]->GetBlock(0));
                allPp[j-1]->Mult(help[j]->GetBlock(1),help[j-1]->GetBlock(1));
            }

            u_norm_L2_2(k) = Ml[k]->InnerProduct(
                sol[k]->GetBlock(0), sol[k]->GetBlock(0));
            p_norm_L2_2(k) = Wl[k]->InnerProduct(
                sol[k]->GetBlock(1), sol[k]->GetBlock(1));
            Vector dsol( allD[k]->Size() );
            allD[k]->Mult(sol[k]->GetBlock(0), dsol );
            norm_div_2(k) = Wl[k]->InnerProduct(dsol, dsol);

            for (int j(0); j < k; ++j)
            {
                if (help[j]->Size() != sol[j]->Size() ||
                   sol[j]->GetBlock(0).Size() != allD[j]->Width() )
                    mfem_error("size don't match \n");

                const int usize  = sol[j ]->GetBlock(0).Size();
                const int psize  = sol[j]->GetBlock(1).Size();
                const int dsize = allD[j]->Size();
                Vector u_H( help[j]->GetData(), usize );
                Vector u_h( sol[j]->GetData(), usize  );
                Vector p_H( help[j]->GetData(), psize );
                Vector p_h( sol[j]->GetData(), psize  );
                Vector u_diff( usize ), du_diff( dsize ), p_diff( psize );
                u_diff = 0.; du_diff = 0.; p_diff = 0.;

                subtract(u_H, u_h, u_diff);
                allD[j]->Mult(u_diff, du_diff);
                subtract(p_H, p_h, p_diff);

                u_errors_L2_2(k,j) =  Ml[j]->InnerProduct(u_diff, u_diff);
                errors_div_2(k,j) =  Wl[j]->InnerProduct(du_diff, du_diff);
                p_errors_L2_2(k,j) =  Wl[j]->InnerProduct(p_diff, p_diff);
            }
        }

        //VISUALIZE SOLUTION
        if (1)
        {
            MultiVector u(sol[k]->GetData(), 1, sol[k]->GetBlock(0).Size() );
            sequence[k]->show(uform, u);
            MultiVector p(sol[k]->GetBlock(1).GetData(), 1, sol[k]->GetBlock(1).Size() );
            sequence[k]->show(pform, p);
        }
    }

    DenseMatrix u_errors_L2(nLevels, nLevels);
    Vector u_norm_L2(nLevels);
    DenseMatrix p_errors_L2(nLevels, nLevels);
    Vector p_norm_L2(nLevels);
    DenseMatrix errors_div(nLevels, nLevels);
    Vector norm_div(nLevels);
    // DenseMatrix timings(nLevels-1, NSTAGES);

    MPI_Reduce(u_errors_L2_2.Data(), u_errors_L2.Data(),
               u_errors_L2.Height()*u_errors_L2.Width(),
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(u_norm_L2_2.GetData(), u_norm_L2.GetData(), u_norm_L2.Size(),
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(p_errors_L2_2.Data(), p_errors_L2.Data(),
               p_errors_L2.Height()*p_errors_L2.Width(),
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(p_norm_L2_2.GetData(), p_norm_L2.GetData(), p_norm_L2.Size(),
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(errors_div_2.Data(), errors_div.Data(),
               errors_div.Height()*errors_div.Width(),
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(norm_div_2.GetData(), norm_div.GetData(), norm_div.Size(),
               MPI_DOUBLE, MPI_SUM, 0, comm);

    if (myid == 0)
    {
        std::transform(u_errors_L2.Data(),
                       u_errors_L2.Data() +
                       u_errors_L2.Height()*u_errors_L2.Width(),
                       u_errors_L2.Data(), (double(*)(double)) sqrt);
        std::transform(u_norm_L2.GetData(), u_norm_L2.GetData()+u_norm_L2.Size(),
                       u_norm_L2.GetData(), (double(*)(double)) sqrt);
        std::transform(p_errors_L2.Data(),
                       p_errors_L2.Data() +
                       p_errors_L2.Height()*p_errors_L2.Width(),
                       p_errors_L2.Data(), (double(*)(double)) sqrt);
        std::transform(p_norm_L2.GetData(),
                       p_norm_L2.GetData()+p_norm_L2.Size(),
                       p_norm_L2.GetData(), (double(*)(double)) sqrt);
        std::transform(errors_div.Data(),
                       errors_div.Data()+errors_div.Height()*errors_div.Width(),
                       errors_div.Data(), (double(*)(double)) sqrt);
        std::transform(norm_div.GetData(), norm_div.GetData()+norm_div.Size(),
                       norm_div.GetData(), (double(*)(double)) sqrt);
    }

    if (myid == 0)
    {
    std::cout << "\n{\n";
    constexpr int w = 14;
    std::cout << "%level" << std::setw(w) << "size" << std::setw(w)
              << "nit" << std::setw(w) << "Assembly" << std::setw(w)
              << "Prec " << std::setw(w) << "Solver\n";
    for (int i(0); i < nLevels; ++i)
        std::cout << i << std::setw(w) << ndofs[i] << std::setw(w)
                  << iter[i] << std::setw(w)
                  << timings(i,ASSEMBLY) << std::setw(w)
                  << timings(i,PRECONDITIONER) << std::setw(w)
                  << timings(i,SOLVER) << "\n";
    std::cout << "}\n";

    std::cout << "\n{\n";
    std::cout << "% || uh - uH ||\n";
    u_errors_L2.PrintMatlab(std::cout);
    std::cout << "% || uh ||\n";
    u_norm_L2.Print(std::cout, nLevels);
    std::cout << "% || ph - pH ||\n";
    p_errors_L2.PrintMatlab(std::cout);
    std::cout << "% || ph ||\n";
    p_norm_L2.Print(std::cout, nLevels);
    std::cout << "% || div ( uh - uH ) ||\n";
    errors_div.PrintMatlab(std::cout);
    std::cout << "% || div uh ||\n";
    norm_div.Print(std::cout, nLevels);
    std::cout << "}\n";
    }

    return EXIT_SUCCESS;
}
