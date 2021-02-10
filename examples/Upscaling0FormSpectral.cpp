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
const char * stage_names[] = {"TOPOLOGY", "SPACES", "ASSEMBLY","PRECONDITIONER",
                              "SOLVER"};

double checkboard_coeff(const Vector& x)
{
    elag_assert(2 <= x.Size() && x.Size() <= 3);
    double d = 10.;

    if ((x.Size() == 2 &&
         ((int)ceil(x(0)*d) & 1) == ((int)ceil(x(1)*d) & 1)) ||
        (x.Size() == 3 &&
         ((((int)ceil(x(2)*d) & 1) &&
           ((int)ceil(x(0)*d) & 1) == ((int)ceil(x(1)*d) & 1)) ||
          (!((int)ceil(x(2)*d) & 1) &&
          ((int)ceil(x(0)*d) & 1) != ((int)ceil(x(1)*d) & 1)))))
    {
        return 1e6;
    }
    else
        return 1e0;
}

int main (int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    const char* meshfile_c = "../meshes/boxcyl.mesh3d";
    int ser_ref_levels = 0;
    int par_ref_levels = 4;
    int feorder = 0;
    int upscalingOrder = 0;
    int max_evects = 10;
    double spect_tol = 0.005;
    bool do_visualize = true;
    int coarsening_step = 1;
    OptionsParser args(argc, argv);
    args.AddOption(&meshfile_c, "-m", "--meshfile",
                   "MFEM mesh file to load.");
    args.AddOption(&ser_ref_levels, "-sr", "--ser_ref_levels",
                   "Number of times to refine serial mesh.");
    args.AddOption(&par_ref_levels, "-pr", "--par_ref_levels",
                   "Number of times to refine parallel mesh.");
    args.AddOption(&feorder, "-feo", "--feorder",
                   "Polynomial order of fine finite element space.");
    args.AddOption(&upscalingOrder, "-uo", "--upscalingorder",
                   "Target polynomial order of coarse space.");
    args.AddOption(&do_visualize, "-v", "--do-visualize", "-nv", "--no-visualize",
                   "Do interactive GLVis visualization.");
    args.AddOption(&max_evects, "-mev", "--max_evects",
                   "Maximum eigenvectors per agglomerate in spectral method.");
    args.AddOption(&spect_tol, "-st", "--spect_tol",
                   "Spectral tolerance for eigenvalues in spectral method.");
    args.AddOption(&coarsening_step, "-cs", "--coarsening_step",
                   "Number of refines to undo with a single coarsen.");
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
        std::cout << "Maximal number of eigenvectors " << max_evects << "\n";
        std::cout << "Spectral tolerance " << spect_tol << "\n";
        std::cout << "Coarsening step " << coarsening_step << "\n";
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
        ess_attr = 0;
        ess_attr[0] = 1;
        ess_attr[2] = 1;

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        pmesh = make_shared<ParMesh>(comm, *mesh);
    }
    const int nDimensions = pmesh->Dimension();

    const int n_refine_levels = par_ref_levels+1;
    Array<int> level_nElements(n_refine_levels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    level_nElements[0] = pmesh->GetNE();

    if(nDimensions == 3)
        pmesh->ReorientTetMesh();

    ConstantCoefficient coeffH1(1.);
    FunctionCoefficient coeffDer(checkboard_coeff);
    Vector bdr(ess_attr.Size());
    bdr = 0.;
    bdr(0) = 1.;
    PWConstCoefficient ubdr(bdr);
    ConstantCoefficient f(0);

    const int nLevels = (par_ref_levels / coarsening_step) + 1;
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
    topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
    chronoInterior.Stop();
    timings(0,TOPOLOGY) = chronoInterior.RealTime();

    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    for (int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        Array<int> partitioning(topology[ilevel]->GetNumberLocalEntities(at_elem));
        chronoInterior.Clear();
        chronoInterior.Start();
        partitioner.Partition(topology[ilevel]->GetNumberLocalEntities(at_elem),
                              level_nElements[(ilevel+1)*coarsening_step],
                              partitioning);
        topology[ilevel+1] =
            topology[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);
        chronoInterior.Stop();
        timings(ilevel+1,TOPOLOGY) = chronoInterior.RealTime();
    }
    chrono.Stop();
    if(myid == 0)
        std::cout<<"Timing ELEM_AGG: Mesh Agglomeration done in "
                 << chrono.RealTime() << " seconds.\n";

    //-----------------------------------------------------//

    chronoInterior.Clear();
    chronoInterior.Start();
    constexpr double tolSVD = 1e-9;
    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());
    if(nDimensions == 3)
        sequence[0] = make_shared<DeRhamSequence3D_FE>(
            topology[0], pmesh.get(), feorder);
    else
        sequence[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(
            topology[0], pmesh.get(), feorder);

    DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();

    constexpr int jFormStart = 0;
    sequence[0]->SetjformStart(jFormStart);

    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 0, make_unique<MassIntegrator>(coeffH1), false);
    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 1, make_unique<VectorFEMassIntegrator>(coeffDer), true);

    // set up coefficients / targets
    sequence[0]->FemSequence()->SetUpscalingTargets(nDimensions, upscalingOrder);
    chronoInterior.Stop();
    timings(0,SPACES) = chronoInterior.RealTime();

    const int form = 0;
    Array<SparseMatrix *> allP(nLevels-1);
    Array<SparseMatrix *> allD(nLevels);

    std::vector<unique_ptr<SparseMatrix>> Ml(nLevels);
    std::vector<unique_ptr<SparseMatrix>> Wl(nLevels);
    chrono.Clear();
    chrono.Start();
    for(int i(0); i < nLevels-1; ++i)
    {
        sequence[i]->SetSVDTol( tolSVD );
        StopWatch chronoInterior;
        chronoInterior.Clear();
        chronoInterior.Start();
        allD[i] = sequence[i]->GetDerivativeOperator(form);
        Ml[i] = sequence[i]->ComputeMassOperator(form);
        Wl[i] = sequence[i]->ComputeMassOperator(form+1);

        // XXX: The matrices in the eigenvalue problems need to be
        // consistent with the systems solved bellow.
        sequence[i]->AgglomerateDofs();
        {
            unique_ptr<SparseMatrix> DtWD_d;
            {
                auto W_d = AssembleAgglomerateMatrix(
                    at_elem, *(sequence[i]->GetM(at_elem,form+1)),
                    sequence[i]->GetDofAgg(form+1),
                    sequence[i]->GetDofAgg(form+1));
                auto D_d = DistributeAgglomerateMatrix(
                    at_elem, *(allD[i]), sequence[i]->GetDofAgg(form+1),
                    sequence[i]->GetDofAgg(form));
                DtWD_d = ExampleRAP(*D_d, *W_d, *D_d);
            }
            auto M_d = AssembleAgglomerateMatrix(
                at_elem,*(sequence[i]->GetM(at_elem,form)),
                sequence[i]->GetDofAgg(form),sequence[i]->GetDofAgg(form));

            elag_assert(DtWD_d->Height() == DtWD_d->Width());
            elag_assert(M_d->Height() == M_d->Width());
            elag_assert(DtWD_d->Height() == M_d->Height());

            auto A_d = ToUnique(Add(*M_d, *DtWD_d));

            if(myid == 0)
                std::cout << "Computing local spectral targets for level "
                          << i+1 << "...\n";

            auto localH1functions = ComputeLocalSpectralTargetsFromAEntity(
                at_elem,*A_d,sequence[i]->GetDofAgg(form),spect_tol,max_evects);
            if(myid == 0)
                std::cout << "Done computing local spectral targets for level "
                          << i+1 << ".\n";

            sequence[i]->OwnLocalTargets(at_elem, 0,std::move(localH1functions));
            sequence[i]->PopulateLocalTargetsFromForm(0);
        }

        if (myid == 0)
            std::cout << "Computing coarse basis for level " << i+1 << " ...\n";

        sequence[i+1] = sequence[i]->Coarsen();

        if (myid == 0)
            std::cout << "Done computing coarse basis for level " << i+1 << ".\n";
        allP[i] = sequence[i]->GetP(form);
        chronoInterior.Stop();
        timings(i+1, SPACES) = chronoInterior.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL" << i << ": Coarsening done in "
                      << chronoInterior.RealTime() << " seconds.\n";
    }

    allD[nLevels-1] = sequence[nLevels-1]->GetDerivativeOperator(form);
    Ml[nLevels-1] = sequence[nLevels-1]->ComputeMassOperator(form);
    Wl[nLevels-1] = sequence[nLevels-1]->ComputeMassOperator(form+1);
    chrono.Stop();

    if(myid == 0)
        std::cout << "Timing ELEM_AGG: Coarsening done in "
                  << chrono.RealTime() << " seconds.\n";

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

    std::vector<unique_ptr<Vector>> rhs(nLevels);
    std::vector<unique_ptr<Vector>> ess_data(nLevels);
    rhs[0] = std::move(b);
    ess_data[0] = std::move(lift);
    for(int i = 0; i < nLevels-1; ++i)
    {
        rhs[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(form));
        ess_data[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(form));
        sequence[i]->GetP(form)->MultTranspose(*(rhs[i]), *(rhs[i+1]) );
        sequence[i]->GetPi(form)->ComputeProjector();
        sequence[i]->GetPi(form)->GetProjectorMatrix().Mult(*(ess_data[i]), *(ess_data[i+1]) );
    }

    std::vector<unique_ptr<Vector>> sol(nLevels);
    std::vector<unique_ptr<Vector>> help(nLevels);

    for(int k(0); k < nLevels; ++k)
    {
        sol[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
        *(sol[k]) = 0.;
        help[k] = make_unique<Vector>( sequence[k]->GetNumberOfDofs(form) );
        *(help[k]) = 0.;
    }

    for(int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        // XXX: The system matrix needs to be consistent with the
        // hierarchy construction above.
        SparseMatrix * M = Ml[k].get();
        SparseMatrix * W = Wl[k].get();
        SparseMatrix * D = allD[k];

        const SharingMap & h1_dofTrueDof(
            sequence[k]->GetDofHandler(0)->GetDofTrueDof());

        unique_ptr<HypreParMatrix> pA;
        {
            auto A = ToUnique(Add(*M, *ExampleRAP(*D, *W, *D)));

            const int nlocdofs = A->Height();
            Array<int> marker(nlocdofs);
            marker = 0;
            sequence[k]->GetDofHandler(form)->MarkDofsOnSelectedBndr(
                ess_attr, marker);

            for(int mm = 0; mm < nlocdofs; ++mm)
                if(marker[mm])
                    A->EliminateRowCol(mm, ess_data[k]->operator ()(mm), *(rhs[k]));
            pA = Assemble(h1_dofTrueDof, *A, h1_dofTrueDof);
        }

        Vector prhs(h1_dofTrueDof.GetTrueLocalSize());
        h1_dofTrueDof.Assemble(*(rhs[k]), prhs);

        elag_assert(prhs.Size() == pA->Height());

        chrono.Stop();
        tdiff = chrono.RealTime();
        if(myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k
                      << ": Assembly done in " << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;

        ndofs[k] = pA->GetGlobalNumRows();
        nnz[k] = pA->NNZ();

        iter[k] = UpscalingHypreSolver(
            form, pA.get(), prhs, sequence[k].get(),
            k, PRECONDITIONER, SOLVER,
            print_iter, max_num_iter, rtol, atol,
            timings, h1_dofTrueDof, *(sol[k]));

        //ERROR NORMS
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
