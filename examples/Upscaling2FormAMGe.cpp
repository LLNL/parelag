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
#include <sstream>

#include <mpi.h>

#include "elag.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

enum {TOPOLOGY=0, SPACES, ASSEMBLY, PRECONDITIONER, SOLVER};

const int NSTAGES = 5;
const char * stage_names[] = {"TOPOLOGY", "SPACES", "ASSEMBLY","PRECONDITIONER","SOLVER"};

int main (int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    elag_trace_init(comm);

    const char* meshfile_c = "../meshes/boxcyl.mesh3d";
    int ser_ref_levels = 0;
    int par_ref_levels = 2;
    int feorder = 0;
    int upscalingOrder = 0;
    bool do_visualize = true;
    OptionsParser args(argc, argv);
    args.AddOption(&meshfile_c, "-m", "--meshfile",
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
        std::cout << "Read mesh " << meshfile << "\n";
        std::cout << "Finite Element Order " << feorder << "\n";
        std::cout << "Upscaling Order " << upscalingOrder << "\n";
        std::cout << "Refine mesh in serial " << ser_ref_levels << " times.\n";
        std::cout << "Refine mesh in parallel " << par_ref_levels << " times.\n";
    }

    shared_ptr<ParMesh> pmesh;
    Array<int> ess_attr;
    {
        // 2. Read the (serial) mesh from the given mesh file and
        // uniformly refine it. If we can't find a file on the command line
        // we generate a boring structured mesh.
        std::ifstream imesh(meshfile.c_str());
        unique_ptr<Mesh> mesh;
        if (imesh)
        {
            mesh = make_unique<Mesh>(imesh, 1, 1);
            imesh.close();
        }
        else
        {
            if (myid == 0)
            {
                std::cout << "Could not find given mesh file: " << meshfile
                          << std::endl << "Generating structured mesh."
                          << std::endl;
            }
            mesh = make_unique<Mesh>(2, 2, 2, Element::HEXAHEDRON, true);
        }
        ess_attr.SetSize(mesh->bdr_attributes.Max());
        ess_attr = 1;

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

    if(nDimensions == 3)
        pmesh->ReorientTetMesh();

    ConstantCoefficient coeffL2(1.);
    ConstantCoefficient coeffHdiv(1.);
    Vector ubdr_vect(pmesh->Dimension());
    ubdr_vect = 0.0;
    VectorConstantCoefficient ubdr(ubdr_vect);
    Vector f_vect(pmesh->Dimension());
    f_vect = 0.0;
    f_vect[nDimensions-1] = 1.0;
    VectorConstantCoefficient f(f_vect);

    DenseMatrix timings(nLevels, NSTAGES);
    timings = 0.0;

    StopWatch chrono;

    MFEMRefinedMeshPartitioner partitioner(nDimensions);
    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    StopWatch chronoInterior;
    chrono.Clear();
    chrono.Start();
    chronoInterior.Clear();
    chronoInterior.Start();
    elag_trace("Generate Fine Grid Topology");
    topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
    elag_trace("Generate Fine Grid Topology Finished");
    chronoInterior.Stop();
    timings(0,TOPOLOGY) = chronoInterior.RealTime();

    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        Array<int> partitioning(topology[ilevel]->GetNumberLocalEntities(at_elem));
        chronoInterior.Clear();
        chronoInterior.Start();
        partitioner.Partition(topology[ilevel]->GetNumberLocalEntities(at_elem),
                              level_nElements[ilevel+1], partitioning);
        topology[ilevel+1] =
            topology[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);
        chronoInterior.Stop();
        timings(ilevel+1,TOPOLOGY) = chronoInterior.RealTime();
    }
    chrono.Stop();
    if(myid == 0)
        std::cout<< "Timing ELEM_AGG: Mesh Agglomeration done in "
                 << chrono.RealTime() << " seconds.\n";

    //-----------------------------------------------------//

    chronoInterior.Clear();
    chronoInterior.Start();
    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());
    sequence[0] = make_shared<DeRhamSequence3D_FE>(
        topology[0], pmesh.get(), feorder);

    DeRhamSequenceFE* DRSequence_FE = sequence[0]->FemSequence();

    constexpr int jFormStart = 0;
    sequence[0]->SetjformStart( jFormStart );

    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 3, make_unique<MassIntegrator>(coeffL2), false);
    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 2, make_unique<VectorFEMassIntegrator>(coeffHdiv), true);

    // set up coefficients / targets
    sequence[0]->FemSequence()->SetUpscalingTargets(nDimensions, upscalingOrder);

    chronoInterior.Stop();
    timings(0,SPACES) = chronoInterior.RealTime();
    chrono.Clear();
    chrono.Start();
    constexpr double tolSVD = 1e-9;
    for(int i(0); i < nLevels-1; ++i)
    {
        sequence[i]->SetSVDTol(tolSVD);
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
        std::cout << "Timing ELEM_AGG: Coarsening done in "
                  << chrono.RealTime() << " seconds.\n";

    const int form = 2;

    // testUpscalingHdiv(sequence);
    FiniteElementSpace * fespace = sequence[0]->FemSequence()->GetFeSpace(form);
    auto b = make_unique<LinearForm>(fespace);
    b->AddDomainIntegrator( new VectorFEDomainLFIntegrator(f));
    b->Assemble();

    auto lift = make_unique<GridFunction>(fespace);
    lift->ProjectBdrCoefficientNormal(ubdr, ess_attr);

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
            std::cout << "Timing ELEM_AGG_LEVEL " << k
                      << ": Assembly done in " << tdiff << "s.\n";
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

        unique_ptr<HypreParMatrix> pA;
        const SharingMap & hdiv_dofTrueDof(
            sequence[k]->GetDofHandler(nDimensions-1)->GetDofTrueDof() );
        // Form the parallel A matrix
        {
            SparseMatrix * M = Ml[k].get();
            SparseMatrix * W = Wl[k].get();
            SparseMatrix * D = allD[k];
            auto A = ToUnique(Add(*M, *ExampleRAP(*D, *W, *D)));

            const int nlocdofs = A->Height();
            Array<int> marker(nlocdofs);
            marker = 0;
            sequence[k]->GetDofHandler(form)->MarkDofsOnSelectedBndr(
                ess_attr, marker);

            for(int mm = 0; mm < nlocdofs; ++mm)
                if(marker[mm])
                    A->EliminateRowCol(
                        mm, ess_data[k]->operator ()(mm), *(rhs[k]) );

            pA = Assemble(hdiv_dofTrueDof, *A, hdiv_dofTrueDof);
        }

        Vector prhs(hdiv_dofTrueDof.GetTrueLocalSize());
        hdiv_dofTrueDof.Assemble(*(rhs[k]), prhs);

        elag_assert(prhs.Size() == pA->Height() );

        chrono.Stop();
        tdiff = chrono.RealTime();
        if(myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k
                      << ": Assembly done in " << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;

        ndofs[k] = pA->GetGlobalNumRows();
        nnz[k] = pA->NNZ();

        chrono.Clear();
        chrono.Start();
        // Same idea as targets; I own them, MLHiptmairSolver views
        // them. This will (probably) change as "shared_ptr" makes its
        // way in (in progress).
        Array<DeRhamSequence *> seqs(nLevels-k);
        for (int ii = 0; ii < nLevels-k; ++ii)
            seqs[ii] = sequence[ii+k].get();

        MLHiptmairSolver<HdivProblem3D> prec(seqs, ess_attr);
        prec.SetMatrix(pA.get());
        chrono.Stop();
        tdiff = chrono.RealTime();
        if(myid == 0)
            std::cout << "Timing LEVEL " << k
                      << ": Preconditioner done in " << tdiff << "s. \n";
        timings(k, PRECONDITIONER) = tdiff;

        Vector psol( pA->Height() );
        psol = 0.;
        CGSolver pcg(comm);
        pcg.SetPrintLevel(print_iter);
        pcg.SetMaxIter(max_num_iter);
        pcg.SetRelTol(rtol);
        pcg.SetAbsTol(atol);
        pcg.SetOperator(*pA );
        pcg.SetPreconditioner( prec );
        chrono.Clear();
        chrono.Start();
        pcg.Mult(prhs, psol );
        chrono.Stop();
        tdiff = chrono.RealTime();
        if(myid == 0)
            std::cout << "Timing LEVEL " << k << ": Solver done in "
                      << tdiff << "s. \n";
        timings(k,SOLVER) = tdiff;

        if(myid == 0)
        {
            if(pcg.GetConverged())
                std::cout << "PCG converged in " << pcg.GetNumIterations()
                          << " with a final residual norm " << pcg.GetFinalNorm()
                          << "\n";
            else
                std::cout << "PCG did not converge in "
                          << pcg.GetNumIterations()
                          << ". Final residual norm is " << pcg.GetFinalNorm()
                          << "\n";
        }

        hdiv_dofTrueDof.Distribute(psol, *(sol[k]));
        iter[k] = pcg.GetNumIterations();

        // ERROR NORMS
        {
            *(help[k]) = *(sol[k]);
            for(int j = k; j > 0; --j)
                allP[j-1]->Mult( *(help[j]), *(help[j-1]) );

            norm_L2_2(k) = Ml[k]->InnerProduct(*(sol[k]), *(sol[k]) );
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

        // VISUALIZE SOLUTION
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
