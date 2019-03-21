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

/* Demonstrates using the LogicalPartitioner with either structured or
 * unstructured coarsening to generate a hierarchy that preserves the boundary
 * interface (defined using mesh attributes) of a mesh embedded within
 * a larger mesh e.g. sphere within a cube.
 */

// Examples for calling the code
// mpirun -np 4 ./EmbeddedMeshDemo.exe
// mpirun -np 4 ./EmbeddedMeshDemo.exe --unstructured --coarsening_factor 8

#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

#include <mpi.h>

#include "elag.hpp"
#include "hypreExtension/hypreExtension.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

enum {EMPTY = 0, HYPRE};
enum {ASSEMBLY = 0, PRECONDITIONER_EMPTY, PRECONDITIONER_HYPRE, SOLVER_EMPTY, SOLVER_HYPRE};

const int NSOLVERS = 2;
const char * solver_names[] = {"EMPTY","HYPRE"};

const int NSTAGES = 5;
const char * stage_names[] = {"ASSEMBLY", "prec EMPTY", "prec HYPRE",
                              "SOLVER_EMPTY", "SOLVER_HYPRE"};
int main (int argc, char *argv[])
{
    std::cout << "------" << std::endl;
    for (int i=0; i<argc; ++i)
        std::cout << argv[i] << " ";;
    std::cout << std::endl << "------" << std::endl;

    // Initialize MPI
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // Command line options
    const char* meshfile_c = "../meshes/sphere_in_sphere_25K.mesh3D";
    int form = 0;
    int feorder = 0;
    int upscalingOrder = 0;
    int ser_ref_levels = 0;
    int par_ref_levels = 1;
    int coarsening_factor = 8;
    int n_bdr_attributes = 8;
    int nLevels = 2;
    bool do_visualize = true;
    bool unstructured = false;
    Array<int> ess_attr;
    Array<int> nat_attr;
    OptionsParser args(argc, argv);
    args.AddOption(&meshfile_c, "-m", "--mesh",
                   "MFEM mesh file to load.");
    args.AddOption(&ser_ref_levels, "-sr", "--ser_ref_levels",
                   "Number of times to refine serial mesh.");
    args.AddOption(&par_ref_levels, "-pr", "--par_ref_levels",
                   "Number of times to refine parallel mesh.");
    args.AddOption(&coarsening_factor, "-cf", "--coarsening_factor",
                   "Geometric coarsening ratio.");
    args.AddOption(&n_bdr_attributes, "--number-boundary-attributes",
                   "--number-boundary-attributes",
                   "Number of boundary attributes in mesh.");
    args.AddOption(&feorder, "-feo", "--feorder",
                   "Polynomial order of fine finite element space.");
    args.AddOption(&upscalingOrder, "-uo", "--upscalingorder",
                   "Target polynomial order of coarse space.");
    args.AddOption(&nLevels, "--num-levels", "--num-levels",
                   "Number of levels for unstructured coarsening.");
    args.AddOption(&unstructured, "--unstructured", "--unstructured",
                   "--structured", "--structured",
                   "Whether to do unstructured coarsening");
    args.AddOption(&do_visualize, "-v", "--do-visualize", "-nv", "--no-visualize",
                   "Do interactive GLVis visualization.");
    args.AddOption(&ess_attr, "--essential-attributes", "--essential-attributes",
                   "Array of 0s, 1s for essential boundary attributes.");
    args.AddOption(&nat_attr, "--natural-attributes", "--natural-attributes",
                   "Array of 0s, 1s for essential boundary attributes.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(std::cout);
        MPI_Finalize();
        return 1;
    }
    PARELAG_ASSERT(args.Good());
    std::string meshfile(meshfile_c);

    if (ess_attr.Size() == 0 && nat_attr.Size() == 0)
    {
        ess_attr.SetSize(n_bdr_attributes);
        ess_attr = 1;
        nat_attr.SetSize(n_bdr_attributes);
        nat_attr = 0;
    }

    if (!unstructured)
        nLevels = par_ref_levels + 1;

    // default linear solver options
    constexpr int print_iter = 0;
    constexpr int max_num_iter = 500;
    constexpr double rtol = 1e-6;
    constexpr double atol = 1e-12;

    if(myid == 0)
    {
        std::cout << "Read mesh " << meshfile << "\n";
        std::cout << "Refine mesh in serial " << ser_ref_levels << " times.\n";
        std::cout << "Refine mesh in parallel " << par_ref_levels << " times.\n";
    }

    shared_ptr<ParMesh> pmesh;
    {
        std::ifstream imesh(meshfile.c_str());
        unique_ptr<Mesh> mesh;
        if (imesh)
        {
            mesh = make_unique<Mesh>(imesh, 1, 1);
            imesh.close();
        }
        else
        {
            if (!myid)
            {
                std::cerr << "\nCannot open mesh file " << meshfile
                          << ", falling back to default behavior." << std::endl;
                std::cout << "Generating cube mesh with 8 hexahedral elements.\n";
            }
            mesh = make_unique<Mesh>(2, 2, 2, Element::HEXAHEDRON, true);
            // Change mesh attribute of bottom half of cube
            for ( int i = 0; i < 4; i++)
                mesh->GetElement(i)->SetAttribute(2);
        }

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        pmesh = make_shared<mfem::ParMesh>(comm,*mesh);

    }

    const int nDimensions = pmesh->Dimension();
    Array<int> level_nElements(nLevels);

    for (int l=0; l<par_ref_levels; l++)
    {
        if (!unstructured)
            level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    pmesh->PrintCharacteristics();
    level_nElements[0] = pmesh->GetNE();

    if (nDimensions == 3)
        pmesh->ReorientTetMesh();

    ConstantCoefficient coeffSpace(1.0);
    ConstantCoefficient coeffDer(1.0);

    Vector ess_bc(n_bdr_attributes), nat_bc(n_bdr_attributes);
    ess_bc = 0.; nat_bc = 0.;

    for (int i(0); i < n_bdr_attributes; ++i)
        if (ess_attr[i] == 1)
            ess_bc(i) = 1.;

    for (int i(0); i < n_bdr_attributes; ++i)
        if (nat_attr[i] == 1)
            nat_bc(i) = -1.;

    PWConstCoefficient ubdr(ess_bc);
    PWConstCoefficient fbdr(nat_bc);
    // fbdr3d only need for form == 1
    Vector allones(nDimensions);
    allones = 1.;
    VectorConstantCoefficient tmp(allones);
    VectorRestrictedCoefficient fbdr3d(tmp, nat_attr);

    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    StopWatch chrono;
    chrono.Clear();
    chrono.Start();
    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;

    if (unstructured)
    {
        if (!myid)
            std::cout << "Doing unstructured coarsening...\n" << std::endl;
        std::vector<Array<MetisMaterialId>> info(nLevels);
        info[0].SetSize(pmesh->GetNE());
        MetisGraphPartitioner mpartitioner;
        mpartitioner.setFlags(MetisGraphPartitioner::KWAY ); // BISECTION
        mpartitioner.setOption(METIS_OPTION_SEED, 0);         // Fix the seed
        mpartitioner.setOption(METIS_OPTION_CONTIG,1);        // Ask metis to provide contiguous partitions
        mpartitioner.setOption(METIS_OPTION_MINCONN,1);       //
        mpartitioner.setUnbalanceToll(1.05); //

        LogicalPartitioner lpartitioner;
        std::vector<unique_ptr<CoarsenMetisMaterialId>>
                coarseningOp(nLevels-1);

        topology[0] = make_shared<AgglomeratedTopology>( pmesh, nDimensions );

        for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
        {
            Array<int> lpartitioning(
                    topology[ilevel]->GetNumberLocalEntities(at_elem));
            int lnum_partitions = lpartitioning.Size()/coarsening_factor;
            if(lnum_partitions == 0) lnum_partitions = 1;
            coarseningOp[ilevel] = make_unique<CoarsenMetisMaterialId>( mpartitioner,
                                            *(topology[ilevel]),
                                            lnum_partitions,
                                            info[ilevel]);
            // Setup logical info on fine grid
            if(ilevel == 0)
                coarseningOp[0]->FillFinestMetisMaterialId(*pmesh,info[0]);
            // Generate the metis partitioning
            lpartitioner.Partition<MetisMaterialId,CoarsenMetisMaterialId>(
                *(topology[ilevel]->LocalElementElementTable()),
                info[ilevel],
                *(coarseningOp[ilevel]),
                lpartitioning);
            // Build coarser topology based on partitioning
            topology[ilevel+1] = topology[ilevel]->
                CoarsenLocalPartitioning(lpartitioning, 1, 1);
            // Setup logical info for next level
            lpartitioner.ComputeCoarseLogical<MetisMaterialId,CoarsenMetisMaterialId>(
                *(coarseningOp[ilevel]),
                topology[ilevel]->AEntityEntity(at_elem),
                info[ilevel],
                info[ilevel+1]);
        }
    }
    else
    {
        if (!myid)
            std::cout << "Doing regular coarsening...\n" << std::endl;
        // Using LogicalPartitioner for structured coarsening
        MFEMRefinedMeshPartitioner partitioner(nDimensions);
        topology[0] = make_unique<AgglomeratedTopology>( pmesh, nDimensions );
        std::vector<Array<MFEMMaterialId>> info(nLevels);
        info[0].SetSize(pmesh->GetNE());

        LogicalPartitioner lpartitioner;
        std::vector<unique_ptr<CoarsenMFEMMaterialId>>
                coarseningOp(nLevels-1);

        for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
        {
            Array<int> lpartitioning(
                    topology[ilevel]->GetNumberLocalEntities(at_elem));
            coarseningOp[ilevel] = make_unique<CoarsenMFEMMaterialId>( partitioner,
                                            *(topology[ilevel]),
                                            level_nElements[ilevel+1],
                                            info[ilevel]);
           // Setup logical info on fine grid
            if(ilevel == 0)
                coarseningOp[0]->FillFinestMFEMMaterialId(*pmesh,info[0]);
            // Generate the metis partitioning
            lpartitioner.Partition<MFEMMaterialId,CoarsenMFEMMaterialId>(
                *(topology[ilevel]->LocalElementElementTable()),
                info[ilevel],
                *(coarseningOp[ilevel]),
                lpartitioning);
            // Build coarser topology based on partitioning
            topology[ilevel+1] = topology[ilevel]->
                CoarsenLocalPartitioning(lpartitioning, 1, 0);
            // Setup logical info for next level
            lpartitioner.ComputeCoarseLogical<MFEMMaterialId,CoarsenMFEMMaterialId>(
                *(coarseningOp[ilevel]),
                topology[ilevel]->AEntityEntity(at_elem),
                info[ilevel],
                info[ilevel+1]);
        }
    }

    chrono.Stop();
    if(myid == 0)
        std::cout<<"Timing ELEM_AGG: Mesh Agglomeration done in "
                 << chrono.RealTime() << " seconds.\n";

    for(int ilevel = 0; ilevel < nLevels; ++ilevel)
        if (do_visualize)
            ShowTopologyAgglomeratedElements(topology[ilevel].get(),pmesh.get());

    //-----------------------------------------------------//

    constexpr double tolSVD = 1e-9;
    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());
    if (nDimensions == 3)
        sequence[0] = make_shared<DeRhamSequence3D_FE>(
            topology[0], pmesh.get(), feorder);
    else
        sequence[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(
            topology[0], pmesh.get(), feorder);

    DeRhamSequenceFE* DRSequence_FE = sequence[0]->FemSequence();

    // for unstructured, jFormStart = form is possibly reasonable
    constexpr int jFormStart = 0;
    sequence[0]->SetjformStart(jFormStart);

    // replace mass integrators
    if (form == 0)
    {
        sequence[0]->FemSequence()->ReplaceMassIntegrator(
            at_elem, form, make_unique<MassIntegrator>(coeffSpace), false);
        sequence[0]->FemSequence()->ReplaceMassIntegrator(
            at_elem, form+1, make_unique<VectorFEMassIntegrator>(coeffDer), true);
    }
    else if (form == 1)
    {
        sequence[0]->FemSequence()->ReplaceMassIntegrator(
            at_elem, form, make_unique<VectorFEMassIntegrator>(coeffSpace), true);
        if (nDimensions == 3)
            sequence[0]->FemSequence()->ReplaceMassIntegrator(
                at_elem, form+1,
                make_unique<VectorFEMassIntegrator>(coeffDer), false);
        else
            sequence[0]->FemSequence()->ReplaceMassIntegrator(
                at_elem, form+1, make_unique<MassIntegrator>(coeffDer), false);
    }
    else if (form == 2)
    {
        sequence[0]->FemSequence()->ReplaceMassIntegrator(
            at_elem, form, make_unique<VectorFEMassIntegrator>(coeffSpace), true);
        sequence[0]->FemSequence()->ReplaceMassIntegrator(
            at_elem, form+1, make_unique<MassIntegrator>(coeffDer), false);
    }
    else
    {
        elag_error(1);
    }

    // set up coefficients / targets
    sequence[0]->FemSequence()->SetUpscalingTargets(nDimensions, upscalingOrder);

    chrono.Clear();
    chrono.Start();
    for (int i(0); i<nLevels-1; ++i)
    {
        sequence[i]->SetSVDTol(tolSVD);
        StopWatch chronoInterior;
        chronoInterior.Clear();
        chronoInterior.Start();
        if (!myid)
            std::cout << "Coarsening at level " << i << std::endl;
        sequence[i+1] = sequence[i]->Coarsen();
        chronoInterior.Stop();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL" << i << ": Coarsening done in "
                      << chronoInterior.RealTime() << " seconds.\n";
    }
    chrono.Stop();

    if (myid == 0)
        std::cout << "Timing ELEM_AGG: Coarsening done in "
                  << chrono.RealTime() << " seconds.\n";

    FiniteElementSpace * fespace = DRSequence_FE->GetFeSpace(form);
    auto b = make_unique<LinearForm>(fespace);
    if (form == 0)
        b->AddBoundaryIntegrator(new BoundaryLFIntegrator(fbdr));
    else if (form == 1)
        if (nDimensions == 3)
            b->AddBoundaryIntegrator(
                new VectorFEBoundaryTangentLFIntegrator(fbdr3d));
        else
            b->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fbdr));
    else // form == 2
        b->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fbdr));

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

    for (int i = 0; i < nLevels - 1; ++i)
        allP[i] = sequence[i]->GetP(form);

    for (int i = 0; i < nLevels; ++i)
        allD[i] = sequence[i]->GetDerivativeOperator(form);

    std::vector<unique_ptr<mfem::SparseMatrix>> Ml(nLevels);
    std::vector<unique_ptr<mfem::SparseMatrix>> Wl(nLevels);

    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        Ml[k] = sequence[k]->ComputeMassOperator(form);
        Wl[k] = sequence[k]->ComputeMassOperator(form+1);
        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k
                      << ": Assembly done in " << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;
    }
    std::vector<unique_ptr<Vector>> rhs(nLevels);
    std::vector<unique_ptr<Vector>> ess_data(nLevels);

    // 'b' and 'lift' now point to nothing, but are never used again.
    rhs[0] = std::move(b);
    ess_data[0] = std::move(lift);
    for (int i=0; i<nLevels-1; ++i)
    {
        rhs[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(form));
        ess_data[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(form));
        sequence[i]->GetP(form)->MultTranspose(*(rhs[i]), *(rhs[i+1]));
        sequence[i]->GetPi(form)->ComputeProjector();
        sequence[i]->GetPi(form)->GetProjectorMatrix().Mult(
            *(ess_data[i]), *(ess_data[i+1]));
    }

    std::vector<unique_ptr<mfem::Vector>> sol_EMPTY(nLevels);
    std::vector<unique_ptr<mfem::Vector>> sol_HYPRE(nLevels);
    std::vector<unique_ptr<mfem::Vector>> help(nLevels);

    for (int k(0); k < nLevels; ++k)
    {
        sol_EMPTY[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
        *(sol_EMPTY[k]) = 0.;
        sol_HYPRE[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
        *(sol_HYPRE[k]) = 0.;
        help[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
        *(help[k]) = 0.;
    }

    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();


        unique_ptr<HypreParMatrix> pA;
        const SharingMap & form_dofTrueDof(
            sequence[k]->GetDofHandler(form)->GetDofTrueDof());
        {
            SparseMatrix * M = Ml[k].get();
            SparseMatrix * W = Wl[k].get();
            SparseMatrix * D = allD[k];
            auto A = ToUnique(Add(*M, *ExampleRAP(*D,*W,*D)));

            const int nlocdofs = A->Height();
            Array<int> marker(nlocdofs);
            marker = 0;
            sequence[k]->GetDofHandler(form)->MarkDofsOnSelectedBndr(
                ess_attr, marker);

            for (int mm = 0; mm < nlocdofs; ++mm)
                if (marker[mm])
                    A->EliminateRowCol(
                        mm, ess_data[k]->operator ()(mm), *(rhs[k]));

            pA = Assemble(form_dofTrueDof, *A, form_dofTrueDof);
        }

        Vector prhs( form_dofTrueDof.GetTrueLocalSize() );
        form_dofTrueDof.Assemble(*(rhs[k]), prhs);

        elag_assert(prhs.Size() == pA->Height() );

        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k
                      << ": Assembly done in " << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;

        ndofs[k] = pA->GetGlobalNumRows();
        nnz[k] = pA->NNZ();

        {
            (*sol_EMPTY[k]) = 0.;
            timings(k, PRECONDITIONER_EMPTY) = 0.;
            timings(k,SOLVER_EMPTY) = 0.;
            iter(k,EMPTY) = 0;
        }

        iter(k,HYPRE) = UpscalingHypreSolver(
            form, pA.get(), prhs, sequence[k].get(),
            k, PRECONDITIONER_HYPRE, SOLVER_HYPRE,
            print_iter, max_num_iter, rtol, atol,
            timings, form_dofTrueDof, *(sol_HYPRE[k]));

        std::vector<unique_ptr<Vector> >& sol = sol_HYPRE;

        // error norms
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
                if (help[j]->Size() != sol[j]->Size() ||
                    sol[j]->Size() != allD[j]->Width() )
                    mfem_error("size don't match.\n");

                const int size  = sol[j]->Size();
                const int dsize = allD[j]->Height();
                Vector u_H(help[j]->GetData(), size);
                Vector u_h(sol[j]->GetData(), size);
                Vector u_diff(size), du_diff(dsize);
                u_diff = 0.; du_diff = 0.;

                subtract(u_H, u_h, u_diff);
                allD[j]->Mult(u_diff, du_diff);

                errors_L2_2(k,j) =  Ml[j]->InnerProduct(u_diff, u_diff);
                errors_div_2(k,j) =  Wl[j]->InnerProduct(du_diff, du_diff);
            }
        }

        // visualize solution
        if (do_visualize)
        {
            MultiVector tmp(sol[k]->GetData(), 1, sol[k]->Size());
            sequence[k]->show(form, tmp);
        }
    }
    OutputUpscalingTimings(ndofs, nnz, iter, timings,
                           solver_names, stage_names);

    ReduceAndOutputUpscalingErrors(errors_L2_2, norm_L2_2,
                                   errors_div_2, norm_div_2);

    return EXIT_SUCCESS;
}
