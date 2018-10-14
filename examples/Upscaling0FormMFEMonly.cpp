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
    // 1. Initialize MPI
    mpi_session sess(argc,argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    const char* meshfile_c = "../meshes/boxcyl.mesh3d";
    int ser_ref_levels = 0;
    int feorder = 0;
    bool do_visualize = true;
    OptionsParser args(argc, argv);
    args.AddOption(&meshfile_c, "-m", "--mesh",
                   "MFEM mesh file to load.");
    args.AddOption(&ser_ref_levels, "-sr", "--nref_serial",
                   "Number of times to refine serial mesh.");
    args.AddOption(&feorder, "-feo", "--feorder",
                   "Polynomial order of fine finite element space.");
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
    constexpr int print_iter = 0;
    constexpr int max_num_iter = 500;
    constexpr double rtol = 1e-6;
    constexpr double atol = 1e-12;

    if (myid == 0)
    {
        std::cout << "Read mesh " << meshfile << std::endl;
        std::cout << "Finite Element Order " << feorder << std::endl;
        std::cout << "Refine mesh in serial "
                  << ser_ref_levels << " times.\n";
    }

    shared_ptr<ParMesh> pmesh;
    shared_ptr<ParMesh> pmesh1;
    Array<int> ess_attr;
    {
        // 2. Read the (serial) mesh from the given mesh file and
        //    uniformly refine it.
        std::ifstream imesh(meshfile.c_str());
        if (!imesh)
        {
            if (myid == 0)
                std::cerr << std::endl << "Can not open mesh file: "
                          << meshfile << std::endl << std::endl;
            return EXIT_FAILURE;
        }
        // This guy just gets used as a stepping-stone to the ParMesh;
        // once we get there, it is destroyed.
        auto mesh = make_unique<Mesh>(imesh, 1, 1);
        imesh.close();

        ess_attr.SetSize(mesh->bdr_attributes.Max());
        ess_attr = 1;

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        pmesh = make_shared<ParMesh>(comm,*mesh);
        pmesh1 = make_shared<ParMesh>(comm,*mesh);
    }
//    pmesh->ReorientTetMesh();
//    pmesh1->ReorientTetMesh();

    if (myid == 0)
        std::cout << "NE: " << pmesh->GetNE() << " NF: " << pmesh->GetNumFaces() << " (BDR: " <<  pmesh->GetNBE() << ") Edges: " << pmesh->GetNEdges() << std::endl;

    const int nDimensions = pmesh->Dimension();
    const int nLevels = 2;
    DenseMatrix timings(nLevels, NSTAGES);
    timings = 0.0;

    StopWatch chrono;

    constexpr auto AT_elem = AgglomeratedTopology::ELEMENT;
    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    StopWatch chronoInterior;
    chrono.Clear();
    chrono.Start();
    chronoInterior.Clear();
    chronoInterior.Start();
    topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
    topology[1] = make_shared<AgglomeratedTopology>(pmesh1, nDimensions);
    chronoInterior.Stop();
    timings(0,TOPOLOGY) = chronoInterior.RealTime();

    ConstantCoefficient coeffH1(1.);
    ConstantCoefficient coeffDer(1.);
    ConstantCoefficient ubdr(0);
    ConstantCoefficient f(1);

    chronoInterior.Clear();
    chronoInterior.Start();
    std::vector<shared_ptr<DeRhamSequenceFE>> sequence(topology.size());

    if (nDimensions == 3)
        sequence[1] = make_shared<DeRhamSequence3D_FE>(
            topology[1],pmesh1.get(),feorder);
    else
        sequence[1] = make_shared<DeRhamSequence2D_Hdiv_FE>(
            topology[1],pmesh1.get(),feorder);

    DeRhamSequenceFE * DRSequence_FE = sequence[1]->FemSequence();
    PARELAG_ASSERT(DRSequence_FE);

    constexpr int jFormStart = 0;
    sequence[1]->SetjformStart(jFormStart);

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

    if (nDimensions == 3)
        sequence[0] = make_shared<DeRhamSequence3D_FE>(
            topology[0],pmesh.get(),feorder,false);
    else
        sequence[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(
            topology[0],pmesh.get(),feorder,false);

    DRSequence_FE = sequence[0]->FemSequence();
    PARELAG_ASSERT(DRSequence_FE);

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
        false);

    pmesh->UniformRefinement();
//    pmesh->ReorientTetMesh();
    if (myid == 0)
        std::cout << "NE: " << pmesh->GetNE() << " NF: " << pmesh->GetNumFaces() << " (BDR: " <<  pmesh->GetNBE() << " attr: " << pmesh->bdr_attributes.Max() << ") Edges: " << pmesh->GetNEdges() << std::endl;

    DRSequence_FE->Update();

    chronoInterior.Stop();
    timings(0,SPACES) = chronoInterior.RealTime();

    const int jform = 0;
    FiniteElementSpace * fespace = DRSequence_FE->GetFeSpace(jform);
    SparseMatrix *Prol = (SparseMatrix *)fespace->GetUpdateOperator();
    PARELAG_ASSERT(Prol);
    unique_ptr<SparseMatrix> P = make_unique<SparseMatrix>(*Prol);
    SparseMatrix *P1 = (SparseMatrix *)DRSequence_FE->GetFeSpace(jform+1)->GetUpdateOperator();
    PARELAG_ASSERT(P1);

    // using unique_ptr here because I want to pass these to the rhs
    // and ess_data data structures below.
    auto b = make_unique<LinearForm>(fespace);
    b->AddDomainIntegrator(new DomainLFIntegrator(f));
    b->Assemble();

    auto lift = make_unique<GridFunction>(fespace);
    lift->ProjectBdrCoefficient(ubdr, ess_attr);
    auto lift1 = make_unique<GridFunction>(sequence[0]->GetFeSpace(jform));
    lift1->ProjectBdrCoefficient(ubdr, ess_attr);

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

    std::vector<unique_ptr<SparseMatrix>> Ml(nLevels);
    std::vector<unique_ptr<SparseMatrix>> Wl(nLevels);
    Ml[0] = sequence[0]->ComputeMassOperator(jform);
    Wl[0] = sequence[0]->ComputeMassOperator(jform+1);
    Ml[1] = unique_ptr<SparseMatrix>(RAP(*P, *Ml[0], *P));
    Wl[1] = unique_ptr<SparseMatrix>(RAP(*P1, *Wl[0], *P1));
//    Ml[1] = sequence[1]->ComputeMassOperator(jform);
//    Wl[1] = sequence[1]->ComputeMassOperator(jform+1);
    dropSmallEntry(*Ml[1], 0.0);
    dropSmallEntry(*Wl[1], 0.0);

    Array<SparseMatrix *> allP(nLevels-1);
    Array<SparseMatrix *> allD(nLevels);

    sequence[0]->SetP(jform, move(P));

    for (int i = 0; i < nLevels - 1; ++i)
        allP[i] = sequence[i]->GetP(jform);

    for (int i = 0; i < nLevels; ++i)
        allD[i] = sequence[i]->GetDerivativeOperator(jform);

    std::vector<unique_ptr<Vector>> rhs(nLevels);
    std::vector<unique_ptr<Vector>> ess_data(nLevels);
    rhs[0] = std::move(b);
    ess_data[0]= std::move(lift);
    ess_data[1]= std::move(lift1);
    for (int i = 0; i < nLevels-1; ++i)
    {
        rhs[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(jform));
        sequence[i]->GetP(jform)->MultTranspose(*(rhs[i]), *(rhs[i+1]));
    }

    std::vector<unique_ptr<Vector>> sol(nLevels);
    std::vector<unique_ptr<Vector>> help(nLevels);

    for (int k(0); k < nLevels; ++k)
    {
        sol[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(jform));
        *(sol[k]) = 0.;
        help[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(jform));
        *(help[k]) = 0.;
    }

    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();

        unique_ptr<HypreParMatrix> pA;
        const SharingMap & h1_dofTrueDof =
            sequence[k]->GetDofHandler(0)->GetDofTrueDof();
        {
            SparseMatrix * M = Ml[k].get();
            SparseMatrix * W = Wl[k].get();
            SparseMatrix * D = allD[k];
            auto A = ToUnique(Add(*M, *ExampleRAP(*D,*W,*D)));

            const int nlocdofs = A->Height();
            Array<int> marker(nlocdofs);
            marker = 0;
            sequence[k]->GetDofHandler(jform)->MarkDofsOnSelectedBndr(
                ess_attr,marker);

            for (int mm = 0; mm < nlocdofs; ++mm)
                if (marker[mm])
                    A->EliminateRowCol(
                        mm, ess_data[k]->operator ()(mm), *(rhs[k]));

            pA = Assemble(h1_dofTrueDof,*A,h1_dofTrueDof);
        }

        Vector prhs{h1_dofTrueDof.GetTrueLocalSize()};
        h1_dofTrueDof.Assemble(*(rhs[k]), prhs);

        elag_assert(prhs.Size() == pA->Height());

        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k
                      << ": Assembly done in " << tdiff << " seconds."
                      << std::endl;
        timings(k, ASSEMBLY) += tdiff;

        ndofs[k] = pA->GetGlobalNumRows();
        nnz[k] = pA->NNZ();

        HypreExtension::HypreBoomerAMGData data;

        chrono.Clear();
        chrono.Start();
        HypreExtension::HypreBoomerAMG prec(*pA, data);
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

        Vector psol(pA->Height());
        psol = 0.;
        CGSolver pcg(comm);
        pcg.SetPrintLevel(print_iter);
        pcg.SetMaxIter(max_num_iter);
        pcg.SetRelTol(rtol);
        pcg.SetAbsTol(atol);
        pcg.SetOperator(*pA);
        pcg.SetPreconditioner(prec);
        chrono.Clear();
        chrono.Start();
        pcg.Mult(prhs,psol);
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

        h1_dofTrueDof.Distribute(psol, *(sol[k]));
        iter[k] = pcg.GetNumIterations();

        //ERROR NORMS
        {
            *(help[k]) = *(sol[k]);
            for (int j = k; j > 0; --j)
                allP[j-1]->Mult(*(help[j]), *(help[j-1]));

            norm_L2_2(k) = Ml[k]->InnerProduct(*(sol[k]), *(sol[k]));
            Vector dsol(allD[k]->Height());
            allD[k]->Mult(*(sol[k]), dsol);
            norm_div_2(k) = Wl[k]->InnerProduct(dsol, dsol);

            for (int j(0); j < k; ++j)
            {
                PARELAG_TEST_FOR_EXCEPTION(
                    help[j]->Size() != sol[j]->Size() ||
                    sol[j]->Size() != allD[j]->Width(),
                    std::runtime_error,
                    "main(): Sizes don't match.");

                const int size = sol[j]->Size();
                const int dsize = allD[j]->Height();
                Vector u_H(help[j]->GetData(),size);
                Vector u_h(sol[j]->GetData(),size);
                Vector u_diff(size), du_diff(dsize);
                u_diff = 0.; du_diff = 0.;

                subtract(u_H, u_h, u_diff);
                allD[j]->Mult(u_diff, du_diff);

                errors_L2_2(k,j) = Ml[j]->InnerProduct(u_diff, u_diff);
                errors_div_2(k,j) = Wl[j]->InnerProduct(du_diff, du_diff);
            }
        }

        if (do_visualize)
        {
            MultiVector tmp(sol[k]->GetData(), 1, sol[k]->Size());
            sequence[k]->show(jform, tmp);
        }
    }

    DenseMatrix errors_L2(nLevels, nLevels);
    errors_L2 = 0.;
    Vector norm_L2(nLevels);
    norm_L2 = 0.;
    DenseMatrix errors_div(nLevels, nLevels);
    errors_div = 0.;
    Vector norm_div(nLevels);
    norm_div = 0.;

    MPI_Reduce(errors_L2_2.Data(),errors_L2.Data(),
               errors_L2.Height()*errors_L2.Width(),
               MPI_DOUBLE,MPI_SUM,0,comm);
    MPI_Reduce(norm_L2_2.GetData(),norm_L2.GetData(),norm_L2.Size(),
               MPI_DOUBLE,MPI_SUM,0,comm);
    MPI_Reduce(errors_div_2.Data(),errors_div.Data(),
               errors_div.Height()*errors_div.Width(),
               MPI_DOUBLE,MPI_SUM,0,comm);
    MPI_Reduce(norm_div_2.GetData(),norm_div.GetData(),norm_div.Size(),
               MPI_DOUBLE,MPI_SUM,0,comm);

    std::transform(errors_L2.Data(),
                   errors_L2.Data()+errors_L2.Height()*errors_L2.Width(),
                   errors_L2.Data(),(double(*)(double)) sqrt);
    std::transform(norm_L2.GetData(),norm_L2.GetData()+norm_L2.Size(),
                   norm_L2.GetData(),(double(*)(double)) sqrt);
    std::transform(errors_div.Data(),
                   errors_div.Data()+errors_div.Height()*errors_div.Width(),
                   errors_div.Data(),(double(*)(double)) sqrt);
    std::transform(norm_div.GetData(),norm_div.GetData()+norm_div.Size(),
                   norm_div.GetData(),(double(*)(double)) sqrt);

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
        std::cout << "% || grad ( uh - uH ) ||\n";
        errors_div.PrintMatlab(std::cout);
        std::cout << "% || grad uh ||\n";
        norm_div.Print(std::cout, nLevels);
        std::cout << "}\n";
    }

    return EXIT_SUCCESS;
}
