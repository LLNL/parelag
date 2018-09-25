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
#include "utilities/mars_utils.hpp"

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
    int par_ref_levels = 1;
    int feorder = 0;
    int upscalingOrder = 0;
    bool do_visualize = true;
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
        std::cout << "Upscaling Order " << upscalingOrder << std::endl;
        std::cout << "Refine mesh in serial "
                  << ser_ref_levels << " times.\n";
        std::cout << "Refine mesh in parallel "
                  << par_ref_levels << " times.\n";
    }

    shared_ptr<Mesh> mesh;
    shared_ptr<Mesh> tmesh;
    Array<int> ess_attr;
    {
        std::ifstream imesh(meshfile.c_str());
        if (!imesh)
        {
            if (myid == 0)
                std::cerr << std::endl << "Can not open mesh file: "
                          << meshfile << " Going default..." << std::endl << std::endl;

             mesh = make_unique<Mesh>(4, 4, 4, Element::TETRAHEDRON, 1);
        } else
        {
            mesh = make_shared<Mesh>(imesh, 1, 1);
            imesh.close();
        }

        ess_attr.SetSize(mesh->bdr_attributes.Max());
        ess_attr = 1;

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();
    }

    const int nDimensions = mesh->Dimension();
    const int geom = mesh->GetElement(0)->GetType();
    const int bdr_geom = (mesh->GetNBE() == 0)? 0 : mesh->GetBdrElement(0)->GetType();
    const int nLevels = par_ref_levels+1;
    Array<int> *partitions[nLevels - 1];
    mars::Mesh2 mesh2;
    mars::Mesh3 mesh3;
    mars::Mesh4 mesh4;
    switch (nDimensions)
    {
        case 2:
            convert_mfem_to_mars_mesh(*mesh, mesh2);
            break;
        case 3:
            convert_mfem_to_mars_mesh(*mesh, mesh3);
            break;
        case 4:
            convert_mfem_to_mars_mesh(*mesh, mesh4);
            break;
        default:
            PARELAG_ASSERT(false);
    }
    for (int l = 0; l < par_ref_levels; l++)
    {
        int *partitioning;
        int nels;
        switch (nDimensions)
        {
            case 2:
                partitioning = uniform_refine(mesh2);
                PARELAG_ASSERT(mesh2.n_active_elements() == mesh2.n_elements());
                nels = mesh2.n_active_elements();
                tmesh = convert_mars_to_mfem_mesh(mesh2, geom, bdr_geom);
                tmesh->CheckPartitioning(partitioning);
                break;
            case 3:
                partitioning = uniform_refine(mesh3);
                PARELAG_ASSERT(mesh3.n_active_elements() == mesh3.n_elements());
                nels = mesh3.n_active_elements();
                tmesh = convert_mars_to_mfem_mesh(mesh3, geom, bdr_geom);
                tmesh->CheckPartitioning(partitioning);
                break;
//            case 4:
//                partitioning = uniform_refine(mesh4);
//                PARELAG_ASSERT(mesh4.n_active_elements() == mesh4.n_elements());
//                nels = mesh4.n_active_elements();
//                tmesh = convert_mars_to_mfem_mesh(mesh4, geom, bdr_geom);
//                tmesh->CheckPartitioning(partitioning);
//                break;
            default:
                PARELAG_ASSERT(false);
        }
        partitions[nLevels - l - 2] = new Array<int>(partitioning, nels);
        partitions[nLevels - l - 2]->MakeDataOwner();
    }
    switch (nDimensions)
    {
        case 2:
            mesh = convert_mars_to_mfem_mesh(mesh2, geom, bdr_geom);
            break;
        case 3:
            mesh = convert_mars_to_mfem_mesh(mesh3, geom, bdr_geom);
            break;
//        case 4:
//            mesh = convert_mars_to_mfem_mesh(mesh4, geom, bdr_geom);
//            break;
        default:
            PARELAG_ASSERT(false);
    }

    if (nDimensions == 3)
        mesh->ReorientTetMesh();

    shared_ptr<ParMesh> pmesh = make_shared<ParMesh>(comm, *mesh);

    ConstantCoefficient coeffH1(1.);
    ConstantCoefficient coeffDer(1.);
    ConstantCoefficient ubdr(0);
    ConstantCoefficient f(1);

    DenseMatrix timings(nLevels, NSTAGES);
    timings = 0.0;

    StopWatch chrono;

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
        PARELAG_ASSERT(partitions[ilevel]->Size() == topology[ilevel]->GetNumberLocalEntities(AT_elem));
        chronoInterior.Clear();
        chronoInterior.Start();
        topology[ilevel+1] = topology[ilevel]->CoarsenLocalPartitioning(*partitions[ilevel],false,false);
        delete partitions[ilevel];
        chronoInterior.Stop();
        timings(ilevel+1,TOPOLOGY) = chronoInterior.RealTime();
    }
    chrono.Stop();
    if (myid == 0)
        std::cout<<"Timing ELEM_AGG: Mesh Agglomeration done in "
                 << chrono.RealTime() << " seconds.\n";

    if (do_visualize && nDimensions <= 3)
        for (int ilevel = 0; ilevel < nLevels-1; ++ilevel)
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

    // testUpscalingHdiv(sequence);
    jform = 0;
    FiniteElementSpace * fespace = DRSequence_FE->GetFeSpace(jform);

    // using unique_ptr here because I want to pass these to the rhs
    // and ess_data data structures below.
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

    for (int i = 0; i < nLevels - 1; ++i)
        allP[i] = sequence[i]->GetP(jform);

    for (int i = 0; i < nLevels; ++i)
        allD[i] = sequence[i]->GetDerivativeOperator(jform);

    std::vector<unique_ptr<SparseMatrix>> Ml(nLevels);
    std::vector<unique_ptr<SparseMatrix>> Wl(nLevels);

    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        Ml[k] = sequence[k]->ComputeMassOperator(jform);
        Wl[k] = sequence[k]->ComputeMassOperator(jform+1);
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
    rhs[0] = std::move(b);
    ess_data[0]= std::move(lift);
    for (int i = 0; i < nLevels-1; ++i)
    {
        rhs[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(jform));
        ess_data[i+1] =
            make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(jform));
        sequence[i]->GetP(jform)->MultTranspose(*(rhs[i]), *(rhs[i+1]));
        sequence[i]->GetPi(jform)->ComputeProjector();
        sequence[i]->GetPi(jform)->GetProjectorMatrix().Mult(
            *(ess_data[i]),*(ess_data[i+1]));
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
