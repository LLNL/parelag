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

// Parelag Upscaling
//
// Canonical test runs (see also CMakeLists.txt):
//
// testsuite/UpscalingGeneralForm.exe --form 0 --nref_parallel 1
// testsuite/UpscalingGeneralForm.exe --form 1 --nref_parallel 1
// mpirun -np 2 testsuite/UpscalingGeneralForm.exe --form 1 --nref_parallel 1
// testsuite/UpscalingGeneralForm.exe --form 2 --nref_parallel 1


#include <fstream>
#include <sstream>

#include <mpi.h>

#include "../src/elag.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

enum {EMPTY = 0, HYPRE};
enum {ASSEMBLY = 0, PRECONDITIONER_EMPTY, PRECONDITIONER_HYPRE, SOLVER_EMPTY,
      SOLVER_HYPRE};

const int NSOLVERS = 2;
const char * solver_names[] = {"EMPTY","HYPRE"};

const int NSTAGES = 5;
const char * stage_names[] = {"ASSEMBLY", "prec EMPTY", "prec HYPRE",
                              "SOLVER_EMPTY", "SOLVER_HYPRE"};

void deformation3D(const Vector & in, Vector & out)
{
    out(1) = in(1) + .5*exp( in(2) );
    out(0) = in(0) + sin( out(1) );
}

void deformation2D(const Vector & in, Vector & out)
{
    out(0) = in(0) + sin( in(1) );
}

int main (int argc, char *argv[])
{
    // Initialize MPI
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

    // Program options
    const char* meshfile_c = "mesh.mesh3d";
    const char* petsc_opts_c = "";
    int form = 0;
    int ser_ref_levels = 0;
    int par_ref_levels = 2;
    int feorder = 0;
    int upscalingOrder = 0;
    bool do_visualize = true;
    bool unassembled = false;
    bool rhs_one = false;
    bool deformation = false;
    int debug_topology = -1;
    bool petsc = false;
    bool unstructured = false;
    bool geometric = false;
    OptionsParser args(argc, argv);
    args.AddOption(&meshfile_c, "-m", "--mesh",
                   "MFEM mesh file to load.");
    args.AddOption(&form, "--form", "--form",
                   "Form in de Rham sequence, 0=H1, 1=Hcurl, 2=Hdiv.");
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
    args.AddOption(&petsc_opts_c, "--petsc_opts", "--petsc_opts",
                   "Options file for PETSc solvers etc.");
    args.AddOption(&unassembled, "--unassembled", "--unassembled",
                   "--assembled", "--assembled",
                   "Whether to use PETSc unassembled or fully assembled matrix format.");
    args.AddOption(&rhs_one, "--rhs-one", "--rhs-one",
                   "--rhs-zero", "--rhs-zero",
                   "Include constant domain integrator on right-hand side.");
    args.AddOption(&deformation, "--deformation", "--deformation",
                   "--no-deformation", "--no-deformation",
                   "Apply artificial (curved) deformation to mesh.");
    args.AddOption(&unstructured, "--unstructured", "--unstructured",
                   "--structured", "--structured",
                   "Use unstructured (metis) partitioning for coarse level.");
    args.AddOption(&geometric, "--geometric", "--geometric",
                   "--no-geometric", "--no-geometric",
                   "Use geometric box partitioner coarsening.");
    args.AddOption(&debug_topology, "--debug-topology", "--debug-topology",
                   "Use specified partitioning and output tables (see badtopology/README).");
    args.AddOption(&petsc, "--use-petsc", "--use-petsc",
                   "--no-petsc", "--no-petsc",
                   "Enable PETSc solvers etc.");
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

#ifdef ParELAG_ENABLE_PETSC
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
#else
    if (petsc)
    {
        if (myid == 0)
        {
            std::cerr << "ParElag was not configured to be used with PETSc" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
#endif

    StopWatch chrono;

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

    // Read the (serial) mesh from the given mesh file and uniformly refine it.
    Array<int> ess_one;
    Array<int> ess_zeros;
    Array<int> nat_one;
    Array<int> nat_zeros;
    int nbdr = 6;
    ess_one.SetSize(nbdr);
    for (int i(0); i < nbdr; ++i)
        ess_one[i] = 0;
    ess_zeros.SetSize(nbdr);
    for (int i(0); i < nbdr; ++i)
        ess_zeros[i] = 0;
    nat_one.SetSize(nbdr);
    for (int i(0); i < nbdr; ++i)
        nat_one[i] = 0;
    nat_zeros.SetSize(nbdr);
    for (int i(0); i < nbdr; ++i)
        nat_zeros[i] = 0;

    Array<int> ess_attr(nbdr);
    Array<int> nat_attr(nbdr);
    shared_ptr<ParMesh> pmesh;
    {
        std::ifstream imesh(meshfile.c_str());
        unique_ptr<Mesh> mesh;
        if (imesh)
        {
            mesh = make_unique<Mesh>(imesh, 1, 1);
            imesh.close();
            if (debug_topology >= 0)
            {
                // todo: this is better handled with an XML input deck
                nat_one[0] = 1;
                nat_zeros[nbdr-1] = 1;
            }
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
            for (int i(1); i < nbdr-1; ++i)
                ess_zeros[i] = 1;
            nat_one[0] = 1;
            nat_zeros[nbdr-1] = 1;
        }

        for (int i(0); i < nbdr; ++i)
            ess_attr[i] = ess_one[i] + ess_zeros[i];
        for (int i(0); i < nbdr; ++i)
            nat_attr[i] = nat_one[i] + nat_zeros[i];

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        pmesh = make_shared<mfem::ParMesh>(comm,*mesh);
    }

    const int nDimensions = pmesh->Dimension();
    int nLevels;
    if (debug_topology >= 0)
        nLevels = 2;
    else
        nLevels = par_ref_levels+1;
    Array<int> level_nElements(nLevels);
    for (int l=0; l<par_ref_levels; l++)
    {
        if (unstructured || geometric) // coarsen more aggressively
            level_nElements[par_ref_levels-l] = pmesh->GetNE() / 2;
        else
            level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    pmesh->PrintCharacteristics();

    level_nElements[0] = pmesh->GetNE();

    if (nDimensions == 3)
        pmesh->ReorientTetMesh();

    if (deformation)
    {
        if (nDimensions == 2)
            pmesh->Transform(deformation2D);
        else
            pmesh->Transform(deformation3D);
    }

    ConstantCoefficient coeffSpace(1.0);
    ConstantCoefficient coeffDer(1.0);

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
    // fbdr3d only need for form == 1
    Vector allones(nDimensions);
    allones = 1.;
    VectorConstantCoefficient tmp(allones);
    VectorRestrictedCoefficient fbdr3d(tmp, nat_one);

    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    // Umberto has some _v2 tests where instead of refining the mesh,
    // and then calling MFEMRefinedMeshPartitioner to coarsen, you
    // refine the topology (also refining the mesh) with something
    // like the following:
    /*
      topology.Last() = new AgglomeratedTopology( pmesh, nDimensions );
      for (int ilevel = nLevels-1; ilevel > 0; --ilevel)
        topology[ilevel-1] = topology[ilevel]->UniformRefinement();
    */
    // this is the only difference between _v2 and regular, so there's
    // lots of duplicated code

    chrono.Clear();
    chrono.Start();
    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    const bool specified_coarsening = (debug_topology >= 0);
    if (specified_coarsening)
    {
        // see badtopology/README for explanation of this
        PARELAG_ASSERT(nLevels == 2);
        topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
        if (!myid)
            std::cout << "    Using specified coarsening..." << std::endl;
        Array<int> partitioning(
            topology[0]->GetNumberLocalEntities(at_elem));
        std::stringstream part_filename;
        part_filename << "specified_partitioning_pair_" << debug_topology;
        std::ifstream part_stream(part_filename.str().c_str());
        int c;
        bool looping = true;
        int total = 0;
        while (looping)
        {
            part_stream >> c;
            looping = (bool) part_stream;
            if (looping)
                partitioning[total++] = c;
        }
        PARELAG_ASSERT(total == partitioning.Size());
        constexpr bool check_topology = true;
        topology[1] = topology[0]->CoarsenLocalPartitioning(
            partitioning, check_topology, false);
    }
    else if (unstructured)
    {
        if (!myid)
            std::cout << "    Doing unstructured coarsening..." << std::endl;

        MetisGraphPartitioner partitioner;
        partitioner.setFlags(MetisGraphPartitioner::KWAY);// BISECTION
        partitioner.setOption(METIS_OPTION_SEED, 0);// Fix the seed
        partitioner.setOption(METIS_OPTION_CONTIG,1);// Contiguous partitions
        partitioner.setOption(METIS_OPTION_MINCONN,1);
        partitioner.setUnbalanceToll(1.05);

        topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
        // try nDimnsions - form, maybe
        for (int ilevel = 0; ilevel < nLevels-1; ++ilevel)
        {
            Array<int> partitioning(
                topology[ilevel]->GetNumberLocalEntities(at_elem));
            partitioner.doPartition(
                *(topology[ilevel]->LocalElementElementTable()),
                topology[ilevel]->Weight(at_elem),
                level_nElements[ilevel+1], partitioning);
            constexpr bool check_topology = true;
            topology[ilevel+1] = topology[ilevel]->CoarsenLocalPartitioning(
                partitioning, check_topology, false);
        }
    }
    else if (geometric)
    {
        if (!myid)
            std::cout << "    Doing geometric box coarsening..." << std::endl;
        GeometricBoxPartitioner partitioner;

        PARELAG_ASSERT(nLevels < 3);
        topology[0] = make_unique<AgglomeratedTopology>(pmesh, nDimensions);
        if (nLevels > 1)
        {
            constexpr int ilevel = 0;
            Array<int> partitioning(
                topology[ilevel]->GetNumberLocalEntities(at_elem));
            partitioner.doPartition(
                *pmesh, level_nElements[ilevel+1], partitioning);
            constexpr bool check_topology = true;
            topology[ilevel+1] = topology[ilevel]->CoarsenLocalPartitioning(
                partitioning, check_topology, false);
        }
    }
    else
    {
        if (!myid)
            std::cout << "    Doing regular (derefinement) coarsening..." << std::endl;
        MFEMRefinedMeshPartitioner partitioner(nDimensions);

        topology[0] = make_unique<AgglomeratedTopology>(pmesh, nDimensions);
        for (int ilevel = 0; ilevel < nLevels-1; ++ilevel)
        {
            Array<int> partitioning(
                topology[ilevel]->GetNumberLocalEntities(at_elem));
            partitioner.Partition(
                topology[ilevel]->GetNumberLocalEntities(at_elem),
                level_nElements[ilevel+1], partitioning);
            topology[ilevel+1] =
                topology[ilevel]->CoarsenLocalPartitioning(
                    partitioning, false, false);
        }
    }
    chrono.Stop();
    if (myid == 0)
        std::cout << "Timing ELEM_AGG: Mesh Agglomeration done in "
                  << chrono.RealTime() << " seconds.\n";

    if (debug_topology >= 0)
    {
        std::cout << "Fine topology characteristics:" << std::endl;
        topology[0]->ShowMe(std::cout);
        std::cout << "Coarse topology characteristics:" << std::endl;
        topology[1]->ShowMe(std::cout);
        std::ofstream AE_E_stream("AE_E.table");
        topology[0]->AEntityEntity(0).Print(AE_E_stream);
        std::ofstream AF_F_stream("AF_F.table");
        topology[0]->AEntityEntity(1).Print(AF_F_stream);
        std::ofstream Ae_e_stream("Aedge_edge.table");
        topology[0]->AEntityEntity(2).Print(Ae_e_stream);
        std::ofstream coarse_el_el_stream("coarse_el_el.table");
        topology[1]->LocalElementElementTable()->Print(coarse_el_el_stream);
        topology[0]->BuildConnectivity();
        auto fine_element_peak = topology[0]->GetConnectivity(
            AgglomeratedTopology::ELEMENT,
            AgglomeratedTopology::PEAK).AsCSRMatrix();
        std::ofstream fine_e_v_stream("fine_el_vertex.table");
        fine_element_peak->Print(fine_e_v_stream);
        auto fine_ridge_peak = topology[0]->GetConnectivity(
            AgglomeratedTopology::RIDGE,
            AgglomeratedTopology::PEAK).AsCSRMatrix();
        std::ofstream fine_r_p_stream("fine_edge_vertex.table");
        fine_ridge_peak->Print(fine_r_p_stream);
    }

    if (do_visualize)
    {
        for (int i=0; i<nLevels; ++i)
            ShowTopologyAgglomeratedElements(topology[i].get(), pmesh.get(), nullptr);
    }

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

    // for unstructured, you could try jFormStart = form, but you would need a
    // different solver than AMS/ADS because those rely on the rest of the
    // sequence
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
    auto b_form = make_unique<LinearForm>(fespace);
    if (form == 0)
    {
        b_form->AddBoundaryIntegrator(new BoundaryLFIntegrator(fbdr));
    }
    else if (form == 1)
    {
        if (nDimensions == 3)
        {
            b_form->AddBoundaryIntegrator(
                new VectorFEBoundaryTangentLFIntegrator(fbdr3d));
        }
        else
        {
            b_form->AddBoundaryIntegrator(
                new VectorFEBoundaryFluxLFIntegrator(fbdr));
        }
    }
    else // form == 2
    {
        b_form->AddBoundaryIntegrator(
            new VectorFEBoundaryFluxLFIntegrator(fbdr));
    }
    {
        ConstantCoefficient one_coefficient(1.0);
        Vector vector_ones(nDimensions);
        vector_ones = 1.0;
        VectorConstantCoefficient v_one_coefficient(vector_ones);
        if (rhs_one)
        {
            std::cout << "Adding domain integrator to right hand side."
                      << std::endl;
            if (form == 0)
                b_form->AddDomainIntegrator(
                    new DomainLFIntegrator(one_coefficient));
            else
                b_form->AddDomainIntegrator(
                    new VectorFEDomainLFIntegrator(v_one_coefficient));
        }
        b_form->Assemble();
    }

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

    // 'b_form' and 'lift' now point to nothing, but are never used again.
    rhs[0] = std::move(b_form);
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

        Array<int> marker;
#ifdef ParELAG_ENABLE_PETSC
        unique_ptr<PetscParMatrix> petscA;
#endif
        unique_ptr<HypreParMatrix> pA;
        const SharingMap & form_dofTrueDof(
            sequence[k]->GetDofHandler(form)->GetDofTrueDof());
        {
            SparseMatrix * M = Ml[k].get();
            SparseMatrix * W = Wl[k].get();
            SparseMatrix * D = allD[k];
            auto A = ToUnique(Add(*M, *ExampleRAP(*D,*W,*D)));

            const int nlocdofs = A->Height();
            marker.SetSize(nlocdofs);
            marker = 0;
            sequence[k]->GetDofHandler(form)->MarkDofsOnSelectedBndr(
                ess_attr, marker);

            for (int mm = 0; mm < nlocdofs; ++mm)
            {
                if (marker[mm])
                    A->EliminateRowCol(
                        mm, ess_data[k]->operator ()(mm), *(rhs[k]));
            }
#ifdef ParELAG_ENABLE_PETSC
            if (petsc)
            {
                petscA = AssemblePetsc(form_dofTrueDof, *A, form_dofTrueDof,
                                       petsc_tid);
            }
            else
#endif
            {
                pA = Assemble(form_dofTrueDof, *A, form_dofTrueDof);
            }
        }
        Vector prhs(form_dofTrueDof.GetTrueLocalSize());
        form_dofTrueDof.Assemble(*(rhs[k]), prhs);
#ifdef ParELAG_ENABLE_PETSC
        if (petsc)
        {
            PARELAG_ASSERT(prhs.Size() == petscA->Height());
        }
        else
#else
        {
            PARELAG_ASSERT(prhs.Size() == pA->Height());
        }
#endif

        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k
                      << ": Assembly done in " << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;

#ifdef ParELAG_ENABLE_PETSC
        if (petsc)
        {
            ndofs[k] = petscA->M();
            nnz[k] = petscA->NNZ();
        }
        else
#endif
        {
            ndofs[k] = pA->M();
            nnz[k] = pA->NNZ();
        }

        {
            (*sol_EMPTY[k]) = 0.;
            timings(k, PRECONDITIONER_EMPTY) = 0.;
            timings(k,SOLVER_EMPTY) = 0.;
            iter(k,EMPTY) = 0;
        }

#ifdef ParELAG_ENABLE_PETSC
        if (petsc)
        {
            iter(k,HYPRE) = UpscalingPetscSolver(
                form, petscA.get(), prhs, sequence[k].get(),
                k, &marker, NULL,
                PRECONDITIONER_HYPRE, SOLVER_HYPRE,
                print_iter, max_num_iter, rtol, atol,
                timings, form_dofTrueDof, *(sol_HYPRE[k]));
        }
        else
#endif
        {
            iter(k,HYPRE) = UpscalingHypreSolver(
                form, pA.get(), prhs, sequence[k].get(),
                k, PRECONDITIONER_HYPRE, SOLVER_HYPRE,
                print_iter, max_num_iter, rtol, atol,
                timings, form_dofTrueDof, *(sol_HYPRE[k]));
        }
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

#ifdef ParELAG_ENABLE_PETSC
    PetscFinalize();
#endif

    return EXIT_SUCCESS;
}
