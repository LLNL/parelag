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
// Sample runs: ./3DHcurlWeakScaling --nref_parallel 2

#include <fstream>
#include <sstream>

#include <mpi.h>

#include "elag.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

enum {AMGe = 0, AMS};
enum {ASSEMBLY = 0,PRECONDITIONER_AMGe,PRECONDITIONER_AMS,SOLVER_AMGe,SOLVER_AMS};

const int NSOLVERS = 2;
const char * solver_names[] = {"AMGe","AMS"};

const int NSTAGES = 5;
const char * stage_names[] = {"ASSEMBLY", "prec AMGe", "prec AMS",
                              "SOLVER_AMGe", "SOLVER_AMS"};

int main (int argc, char *argv[])
{
    // 1. Initialize MPI
    parelag::mpi_session sess(argc,argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    StopWatch chrono;

    const int nbdr = 6;
    Array<int> ess_one(nbdr);
    ess_one = 0;
    Array<int> ess_zeros(nbdr);
    ess_zeros = 1;
    ess_zeros[0] = 0;
    ess_zeros[nbdr-1] = 0;
    Array<int> nat_one(nbdr);
    nat_one = 0;
    nat_one[0] = 1;
    Array<int> nat_zeros(nbdr);
    nat_zeros = 0;
    nat_zeros[nbdr-1] = 1;

    // Overload options from command line
    int par_ref_levels = 4;
    int feorder = 0;
    int upscalingOrder = 0;
    bool do_visualize = true;
    bool reportTiming = true;
    OptionsParser args(argc, argv);
    args.AddOption(&par_ref_levels, "-pr", "--nref_parallel",
                   "Number of times to refine parallel mesh.");
    args.AddOption(&feorder, "-feo", "--feorder",
                   "Polynomial order of fine finite element space.");
    args.AddOption(&upscalingOrder, "-uo", "--upscalingorder",
                   "Target polynomial order of coarse space.");
    args.AddOption(&do_visualize, "-v", "--do-visualize", "-nv", "--no-visualize",
                   "Do interactive GLVis visualization.");
    args.AddOption(&reportTiming, "--report_timing", "--report_timing",
                   "--no_report_timing", "--no_report_timing",
                   "Output timings to stdout.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(std::cout);
        MPI_Finalize();
        return 1;
    }
    PARELAG_ASSERT(args.Good());

    Array<int> ess_attr(nbdr);
    for (int i(0); i < nbdr; ++i)
        ess_attr[i] = ess_one[i] + ess_zeros[i];

    Array<int> nat_attr(nbdr);
    for (int i(0); i < nbdr; ++i)
        nat_attr[i] = nat_one[i] + nat_zeros[i];

    // default linear solver options
    constexpr int print_iter = 0;
    constexpr int max_num_iter = 500;
    constexpr double rtol = 1e-6;
    constexpr double atol = 1e-12;

    if (myid == 0)
        std::cout << "Refine mesh in parallel " << par_ref_levels << " times.\n";

    int n = 1;
    while(n*n*n < num_procs)
        ++n;

    elag_assert(n*n*n == num_procs);

//      Element::Type type = Element::TETRAHEDRON;
    Element::Type type = Element::HEXAHEDRON;

    auto mesh = make_unique<Mesh>(n, n, n, type, true);

    const int nDimensions = mesh->Dimension();

    const int n_el_c = mesh->GetNE();
    Array<int> partitioning(n_el_c);
    for (int i = 0; i < n_el_c; ++i)
    {
        if (type == Element::TETRAHEDRON)
            partitioning[i] = i/6;
        else
            partitioning[i] = i;
    }

    auto pmesh = make_shared<ParMesh>(comm, *mesh, partitioning);
    mesh.reset();

    const int nLevels = par_ref_levels+1;
    Array<int> level_nElements(nLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }

    {
        Vector ver_coord;
        pmesh->GetVertices(ver_coord);
        const int nv = pmesh->GetNV();
        //for (int i = 0; i < nv; i++)
        //      ver_coord(i) = exp(ver_coord(i));
        for (int i = 0; i < nv; i++)
            ver_coord(nv+i) += .5*exp(ver_coord(2*nv+i));
        for (int i = 0; i < nv; i++)
            ver_coord(i) += sin(ver_coord(nv+i));
        pmesh->SetVertices(ver_coord);
    }

    level_nElements[0] = pmesh->GetNE();

    if (nDimensions == 3)
        pmesh->ReorientTetMesh();

    ConstantCoefficient coeff2Form(1.);
    ConstantCoefficient coeff1Form(1.);

    Vector ess_bc(nbdr), nat_bc(nbdr);
    ess_bc = 0.; nat_bc = 0.;

    for (int i(0); i < nbdr; ++i)
        if (ess_one[i] == 1)
            ess_bc(i) = 1.;

    for (int i(0); i < nbdr; ++i)
        if (nat_one[i] == 1)
            nat_bc(i) = -1.;

    PWConstCoefficient ubdr(ess_bc);
    PWConstCoefficient fbdr(nat_bc);

    Vector allones(nDimensions);
    allones = 1.;
    VectorConstantCoefficient tmp(allones);
    VectorRestrictedCoefficient fbdr3d(tmp, nat_one);

    MFEMRefinedMeshPartitioner partitioner(nDimensions);
    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    chrono.Clear();
    chrono.Start();
    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
    for (int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        Array<int> partitioning(topology[ilevel]->GetNumberLocalEntities(at_elem));
        partitioner.Partition(topology[ilevel]->GetNumberLocalEntities(at_elem),
                              level_nElements[ilevel+1], partitioning);
        topology[ilevel+1] =
            topology[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);
    }
    chrono.Stop();
    if (myid == 0 && reportTiming)
        std::cout << "Timing ELEM_AGG: Mesh Agglomeration done in "
                  << chrono.RealTime() << " seconds.\n";

    //-----------------------------------------------------//

    constexpr double tolSVD = 1e-9;
    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size() );
    sequence[0] = make_shared<DeRhamSequence3D_FE>(
        topology[0], pmesh.get(), feorder);

    DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();
    PARELAG_ASSERT(DRSequence_FE);

    constexpr int jFormStart = 0;
    sequence[0]->SetjformStart(jFormStart);

    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 2, make_unique<VectorFEMassIntegrator>(coeff2Form), false);
    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 1, make_unique<VectorFEMassIntegrator>(coeff1Form), true);

    // set up coefficients / targets
    sequence[0]->FemSequence()->SetUpscalingTargets(nDimensions, upscalingOrder, 2);

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
        if (myid == 0 && reportTiming)
            std::cout << "Timing ELEM_AGG_LEVEL" << i << ": Coarsening done in "
                      << chronoInterior.RealTime() << " seconds.\n";
    }
    chrono.Stop();

    if (myid == 0 && reportTiming)
        std::cout << "Timing ELEM_AGG: Coarsening done in " << chrono.RealTime()
                  << " seconds.\n";

    int form = 1;

    //  testUpscalingHdiv(sequence);
    FiniteElementSpace * fespace = sequence[0]->FemSequence()->GetFeSpace(form);
    auto b = make_unique<LinearForm>(fespace);
    b->AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(fbdr3d));
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

    DenseMatrix timings(nLevels, NSTAGES);
    timings = 0.0;

    Array2D<int> iter(nLevels, NSOLVERS);
    iter = 0;
    Array<int> ndofs(nLevels);
    ndofs = 0;
    Array<int> nnz(nLevels);
    nnz = 0;

    double tdiff;

    Array<SparseMatrix *> allP(nLevels-1);
    Array<SparseMatrix *> allD(nLevels);
    std::vector<unique_ptr<HypreParMatrix>> allC(nLevels);

    for (int i = 0; i < nLevels - 1; ++i)
        allP[i] = sequence[i]->GetP(form);

    for (int i = 0; i < nLevels; ++i)
        allC[i] = sequence[i]->ComputeTrueDerivativeOperator(form-1);

    for (int i = 0; i < nLevels; ++i)
        allD[i] = sequence[i]->GetDerivativeOperator(form);

    std::vector<unique_ptr<HypreParMatrix>> allP_bc(nLevels-1);
    std::vector<unique_ptr<HypreParMatrix>> allC_bc(nLevels);

    for (int i = 0; i < nLevels - 1; ++i)
        allP_bc[i] = sequence[i]->ComputeTrueP(form, ess_attr);

    for (int i = 0; i < nLevels; ++i)
        allC_bc[i] = sequence[i]->ComputeTrueDerivativeOperator(form-1, ess_attr);

    std::vector<unique_ptr<SparseMatrix>> Ml(nLevels);
    std::vector<unique_ptr<SparseMatrix>> Wl(nLevels);

    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        Ml[k] = sequence[k]->ComputeMassOperator(form);
        Wl[k] = sequence[k]->ComputeMassOperator(form+1);
        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0 && reportTiming)
            std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in "
                      << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;
    }

    std::vector<unique_ptr<Vector>> rhs(nLevels);
    std::vector<unique_ptr<Vector>> ess_data(nLevels);
    rhs[0] = std::move(b);
    ess_data[0] = std::move(lift);
    for (int i = 0; i < nLevels-1; ++i)
    {
        rhs[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(form));
        ess_data[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(form));
        sequence[i]->GetP(form)->MultTranspose(*(rhs[i]), *(rhs[i+1]) );
        sequence[i]->GetPi(form)->ComputeProjector();
        sequence[i]->GetPi(form)->GetProjectorMatrix().Mult(
            *(ess_data[i]), *(ess_data[i+1]) );
    }

    std::vector<unique_ptr<Vector>> sol_AMGe(nLevels);
    std::vector<unique_ptr<Vector>> sol_AMS(nLevels);
    std::vector<unique_ptr<Vector>> help(nLevels);

    for (int k(0); k < nLevels; ++k)
    {
        sol_AMGe[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
        *(sol_AMGe[k]) = 0.;
        sol_AMS[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
        *(sol_AMS[k]) = 0.;
        help[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
        *(help[k]) = 0.;
    }

    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();

        unique_ptr<HypreParMatrix> pA;
        const SharingMap & form1_dofTrueDof(
            sequence[k]->GetDofHandler(1)->GetDofTrueDof());
        {
            auto M = Ml[k].get();
            auto W = Wl[k].get();
            auto D = allD[k];

            auto A = ToUnique(Add(*M, *ExampleRAP(*D, *W, *D)));

            const int nlocdofs = A->Height();
            Array<int> marker( nlocdofs );
            marker = 0;
            sequence[k]->GetDofHandler(form)->MarkDofsOnSelectedBndr(
                ess_attr, marker);

            for (int mm = 0; mm < nlocdofs; ++mm)
                if (marker[mm])
                    A->EliminateRowCol(
                        mm, ess_data[k]->operator ()(mm), *(rhs[k]) );

            pA = Assemble(form1_dofTrueDof, *A, form1_dofTrueDof);

        }
        Vector prhs( form1_dofTrueDof.GetTrueLocalSize() );
        form1_dofTrueDof.Assemble(*(rhs[k]), prhs);

        elag_assert(prhs.Size() == pA->Height() );

        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0 && reportTiming)
            std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in "
                      << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;

        ndofs[k] = pA->GetGlobalNumRows();
        nnz[k] = pA->NNZ();

        // AMGe SOLVER not implemented
        {
            (*sol_AMGe[k]) = 0.;
            timings(k, PRECONDITIONER_AMGe) = 0.;
            timings(k,SOLVER_AMGe) = 0.;
            iter(k,AMGe) = 0;
        }

        iter(k,AMS) = UpscalingHypreSolver(
            form, pA.get(), prhs, sequence[k].get(),
            k, PRECONDITIONER_AMS, SOLVER_AMS,
            print_iter, max_num_iter, rtol, atol,
            timings, form1_dofTrueDof, *(sol_AMS[k]), reportTiming);

        auto& sol = sol_AMS;
        //ERROR NORMS
        {
            *(help[k]) = *(sol[k]);
            for (int j = k; j > 0; --j)
                allP[j-1]->Mult( *(help[j]), *(help[j-1]) );

            norm_L2_2(k) = Ml[k]->InnerProduct(*(sol[k]), *(sol[k]) );
            Vector dsol( allD[k]->Height() );
            allD[k]->Mult(*(sol[k]), dsol );
            norm_div_2(k) = Wl[k]->InnerProduct(dsol, dsol );

            for (int j(0); j < k; ++j)
            {
                if (help[j]->Size() != sol[j]->Size() ||
                    sol[j]->Size() != allD[j]->Width() )
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

    if (reportTiming)
    {
        OutputUpscalingTimings(ndofs, nnz, iter, timings,
                               solver_names, stage_names);
    }

    ReduceAndOutputUpscalingErrors(errors_L2_2, norm_L2_2,
                                   errors_div_2, norm_div_2);

    return EXIT_SUCCESS;
}
