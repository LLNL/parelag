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

#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"
#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "utilities/ParELAG_TimeManager.hpp"
#include "utilities/ParELAG_SimpleXMLParameterListReader.hpp"
#include "utilities/MPIDataTypes.hpp"

#include "testing_helpers/Build3DHexMesh.hpp"
#include "testing_helpers/CreateDarcyParameterList.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

enum {AMGe = 0, ADS};
enum {ASSEMBLY=0,PRECONDITIONER_AMGe,PRECONDITIONER_ADS,SOLVER_AMGe,SOLVER_ADS};

const int NSOLVERS = 2;
const char * solver_names[] = {"AMGe","ADS"};

const int NSTAGES = 5;
const char * stage_names[] = {"ASSEMBLY", "prec AMGe", "prec ADS",
                              "SOLVER_AMGe", "SOLVER_ADS"};


int main (int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_ranks, myid;
    MPI_Comm_size(comm, &num_ranks);
    MPI_Comm_rank(comm, &myid);

    // Get options from command line
    const char *xml_file_c = "BuildTestParameters";
    double W_weight = 0.0;
    bool reportTiming = true;
    mfem::OptionsParser args(argc, argv);
    args.AddOption(&xml_file_c, "-f", "--xml-file",
                   "XML parameter list.");
    // The constant weight in the system [M B^T; B -(W_weight*W)]
    args.AddOption(&W_weight, "-w", "--L2mass-weight",
                   "The constant weight in the system [M B^T; B -(W_weight*W)]");
    args.AddOption(&reportTiming, "--report_timing", "--report_timing",
                   "--no_report_timing", "--no_report_timing",
                   "Output timings to stdout.");
    args.Parse();
    PARELAG_ASSERT(args.Good());
    std::string xml_file(xml_file_c);

    // Read the parameter list from file
    unique_ptr<ParameterList> master_list;
    if (xml_file == "BuildTestParameters")
    {
        master_list = testhelpers::CreateDarcyTestParameters();
        if (!myid && W_weight == 0)
            std::cout << "Solving Darcy problem without L2 mass\n";
        else if (!myid && W_weight != 0)
            std::cout << "Solving Darcy problem with L2 mass (weight = "
                      << W_weight << ")\n";
    }
    else
    {
        std::ifstream xml_in(xml_file);
        if (!xml_in.good())
        {
            std::cerr << "ERROR: Unable to obtain a good filestream "
                      << "from input file: " << xml_file << ".\n";
            return EXIT_FAILURE;
        }

        SimpleXMLParameterListReader reader;
        master_list = reader.GetParameterList(xml_in);

        xml_in.close();
    }

    ParameterList& prob_list = master_list->Sublist("Problem parameters",true);

    // The file from which to read the mesh
    const std::string meshfile =
        prob_list.Get("Mesh file", "no mesh name found");

    // The number of times to refine in parallel
    const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2);

    // The level of the resulting hierarchy at which you would like
    // the solve to be performed. 0 is the finest grid.
    const int start_level = prob_list.Get("Solve level",0);

    ParameterList& output_list = master_list->Sublist("Output control");
    const bool print_time = output_list.Get("Print timings",true);
    const bool show_progress = output_list.Get("Show progress",true);

    const bool print_progress_report = (!myid && show_progress);

    if (print_progress_report)
        std::cout << "\n-- Hello!\n"
                  << "-- Welcome to RedistributeTopo!\n\n";

    std::ostringstream mesh_msg;
    if (!myid)
    {
        mesh_msg << '\n' << std::string(50,'*') << '\n'
                 << "*  Mesh: " << meshfile << "\n*\n";
    }

    shared_ptr<ParMesh> pmesh;
    {
        if (print_progress_report)
            std::cout << "-- Building and refining serial mesh...\n";

        std::unique_ptr<mfem::Mesh> mesh;
        if (meshfile == "TestingMesh")
        {
            mesh = testhelpers::Build3DHexMesh();

            if (print_progress_report)
                std::cout << "-- Built test mesh successfully." << std::endl;
        }
        else
        {
            std::ifstream imesh(meshfile.c_str());
            if (!imesh)
            {
                if (!myid)
                    std::cerr << std::endl << "Cannot open mesh file: "
                              << meshfile << std::endl << std::endl;
                return EXIT_FAILURE;
            }

            mesh = make_unique<mfem::Mesh>(imesh, 1, 1);
            imesh.close();

            if (print_progress_report)
                std::cout << "-- Read mesh \"" << meshfile
                          << "\" successfully.\n";
        }

        int ser_ref_levels =
            prob_list.Get("Serial refinement levels", -1);

        // This will do no refs if ser_ref_levels <= 0.
        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        // Negative means refine until mesh is big enough to distribute!
        if (ser_ref_levels < 0)
        {
            ser_ref_levels = 0;
            for ( ; mesh->GetNE() < 6*num_ranks; ++ser_ref_levels)
                mesh->UniformRefinement();
        }

        if (print_progress_report)
        {
            std::cout << "-- Refined mesh in serial " << ser_ref_levels
                      << " times.\n";
        }

        if (!myid)
        {
            mesh_msg << "*    Serial refinements: " << ser_ref_levels << '\n';
        }

        if (print_progress_report)
            std::cout << "-- Building parallel mesh...\n";

        pmesh = make_shared<ParMesh>(comm, *mesh);

        if (pmesh && print_progress_report)
            std::cout << "-- Built parallel mesh successfully.\n";
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

    if (print_progress_report)
        std::cout << "-- Refined mesh in parallel " << par_ref_levels
                  << " times.\n\n";

    if (nDimensions == 3)
        pmesh->ReorientTetMesh();


    {
        size_t my_num_elmts = pmesh->GetNE(), global_num_elmts;
        MPI_Reduce(&my_num_elmts,&global_num_elmts,1,GetMPIType<size_t>(0),
                   MPI_SUM,0,comm);

        if (!myid)
        {
            mesh_msg << "*  Parallel refinements: " << par_ref_levels << '\n'
                     << "*        Fine Mesh Size: " << global_num_elmts << '\n'
                     << std::string(50,'*') << '\n' << std::endl;
        }
    }

    if (!myid)
        std::cout << mesh_msg.str();

    const int num_redist_coarsen_levels = 3;
    auto elem_t = AgglomeratedTopology::ELEMENT;

    if (print_progress_report)
        std::cout << "-- Agglomerating topology to " << nLevels+num_redist_coarsen_levels
                  << " levels...\n";

    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels+num_redist_coarsen_levels);
    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());

    Timer outer = TimeManager::AddTimer("Mesh Agglomeration -- Total");
    {
        Timer inner = TimeManager::AddTimer("Mesh Agglomeration -- Level 0");
        topology[0] = make_shared<AgglomeratedTopology>(pmesh, 1);
        ShowTopologyAgglomeratedElements(topology[0].get(), pmesh.get(), nullptr);
    }

    MFEMRefinedMeshPartitioner mfem_partitioner(nDimensions);
    for (int i = 0; i < nLevels-1; ++i)
    {
        Array<int> partitioning(topology[i]->GetB(0).NumRows());
        mfem_partitioner.Partition(topology[i]->GetB(0).NumRows(),
                                   level_nElements[i+1], partitioning);

        topology[i+1] = topology[i]->CoarsenLocalPartitioning(
                    partitioning, 0, 0, nDimensions == 2 ? 0 : 2);
//        ShowTopologyAgglomeratedElements(topology[i+1].get(), pmesh.get(), nullptr);
    }

    if (print_progress_report)
        std::cout << "-- Successfully agglomerated topology before redistribution.\n";

    const int feorder = 0;
    const int upscalingOrder = 0;
    const int jFormStart = nDimensions-1;
    const int uform = nDimensions - 1;
    const int pform = nDimensions;

    if (nDimensions == 3)
    {
        sequence[0] = make_shared<DeRhamSequence3D_FE>(topology[0], pmesh.get(), feorder);
    }
    else
    {
        sequence[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(topology[0], pmesh.get(), feorder);
    }

    DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();
    PARELAG_ASSERT(DRSequence_FE);

    ConstantCoefficient coeffL2(1.);
    ConstantCoefficient coeffHdiv(1.);

    sequence[0]->SetjformStart(jFormStart);

    DRSequence_FE->ReplaceMassIntegrator(
                elem_t, pform, make_unique<MassIntegrator>(coeffL2), false);
    DRSequence_FE->ReplaceMassIntegrator(
                elem_t, uform, make_unique<VectorFEMassIntegrator>(coeffHdiv), true);

    // set up coefficients / targets
    DRSequence_FE->SetUpscalingTargets(nDimensions, upscalingOrder, jFormStart);

    StopWatch chrono;
    chrono.Clear();
    chrono.Start();
    constexpr double tolSVD = 1e-9;
    for (int i(0); i < nLevels-1; ++i)
    {
        sequence[i]->SetSVDTol( tolSVD );
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
        std::cout << "Timing ELEM_AGG: Coarsening before redistribution done in " << chrono.RealTime()
                  << " seconds.\n";

    chrono.Clear();
    chrono.Start();

    int num_redist_procs = 4;
    const int coarsening_factor = std::pow(2, nDimensions);
    MetisGraphPartitioner metis_partitioner;
    metis_partitioner.setFlags(MetisGraphPartitioner::RECURSIVE);
    for (int i = nLevels-1; i < topology.size()-1; ++i)
    {
        StopWatch chronoInterior;
        chronoInterior.Clear();
        chronoInterior.Start();

        std::vector<int> redistributed_procs(topology[i]->GetB(0).NumRows());
        std::fill_n(redistributed_procs.begin(), redistributed_procs.size(), myid % num_redist_procs);
        Redistributor redistributor(*topology[i], redistributed_procs);
        auto redist_topo = redistributor.Redistribute(*topology[i]);
        chronoInterior.Stop();
        if (myid == 0 && reportTiming)
            std::cout << "Timing ELEM_AGG_LEVEL" << i << ": Topology redistributed in "
                      << chronoInterior.RealTime() << " seconds.\n";

        chronoInterior.Clear();
        chronoInterior.Start();
        const int num_global_elem = topology[i]->GetNumberGlobalTrueEntities(elem_t);
        const int num_parts = num_global_elem / coarsening_factor / num_redist_procs;
        topology[i+1] = topology[i]->Coarsen(
                    redistributor, redist_topo, metis_partitioner, num_parts, 0, 0);
        chronoInterior.Stop();
        if (myid == 0 && reportTiming)
            std::cout << "Timing ELEM_AGG_LEVEL" << i << ": Topology coarsened in "
                      << chronoInterior.RealTime() << " seconds.\n";

        // ShowTopologyAgglomeratedElements(topology[i+1].get(), pmesh.get(), nullptr);

        chronoInterior.Clear();
        chronoInterior.Start();
        auto redist_seq = redistributor.Redistribute(*sequence[i], redist_topo);
        chronoInterior.Stop();
        if (myid == 0 && reportTiming)
            std::cout << "Timing ELEM_AGG_LEVEL" << i << ": DeRhamSequence redistributed"
                      << " in " << chronoInterior.RealTime() << " seconds.\n";

        chronoInterior.Clear();
        chronoInterior.Start();
        sequence[i+1] = sequence[i]->Coarsen(redistributor, redist_seq);
        chronoInterior.Stop();
        if (myid == 0 && reportTiming)
            std::cout << "Timing ELEM_AGG_LEVEL" << i << ": DeRhamSequence coarsened in "
                      << chronoInterior.RealTime() << " seconds.\n";

        num_redist_procs /= 2;
    }

    if (myid == 0 && reportTiming)
        std::cout << "Timing ELEM_AGG: Coarsening after redistribution done in " << chrono.RealTime()
                  << " seconds.\n";

    if (print_progress_report)
    {
        std::cout << "\n";
        for (int i = 0; i < topology.size()-1; ++i)
        {
            std::cout << "-- Number of agglomerates on level " << i+1 << " is "
                      << topology[i+1]->GetNumberGlobalTrueEntities(elem_t) <<".\n";
        }
    }

    {
        FiniteElementSpace * ufespace = DRSequence_FE->GetFeSpace(uform);
        FiniteElementSpace * pfespace = DRSequence_FE->GetFeSpace(pform);

        mfem::LinearForm bform(ufespace);
        ConstantCoefficient fbdr(0.0);
        bform.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fbdr));
        bform.Assemble();

        mfem::LinearForm qform(pfespace);
        ConstantCoefficient source(1.0);
        qform.AddDomainIntegrator(new DomainLFIntegrator(source));
        qform.Assemble();

        // Project rhs down to the level of interest
        auto rhs_u = make_unique<mfem::Vector>(sequence[0]->GetNumTrueDofs(uform));
        auto rhs_p = make_unique<mfem::Vector>(sequence[0]->GetNumTrueDofs(pform));
        *rhs_u = 0.0;
        *rhs_p = 0.0;
        sequence[0]->GetDofHandler(uform)->GetDofTrueDof().Assemble(bform, *rhs_u);
        sequence[0]->GetDofHandler(pform)->GetDofTrueDof().Assemble(qform, *rhs_p);

        for (int ii = 0; ii < start_level; ++ii)
        {
            auto tmp_u = make_unique<mfem::Vector>(
                sequence[ii+1]->GetNumTrueDofs(uform) );
            auto tmp_p = make_unique<mfem::Vector>(
                sequence[ii+1]->GetNumTrueDofs(pform) );
            sequence[ii]->GetTrueP(uform).MultTranspose(*rhs_u,*tmp_u);
            sequence[ii]->GetTrueP(pform).MultTranspose(*rhs_p,*tmp_p);
            rhs_u = std::move(tmp_u);
            rhs_p = std::move(tmp_p);
        }

        const SharingMap& hdiv_dofTrueDof
            = sequence[start_level]->GetDofHandler(uform)->GetDofTrueDof();
        const SharingMap& l2_dofTrueDof
            = sequence[start_level]->GetDofHandler(pform)->GetDofTrueDof();

        // Create the parallel linear system
        mfem::Array<int> true_block_offsets(3);
        true_block_offsets[0] = 0;
        true_block_offsets[1] = hdiv_dofTrueDof.GetTrueLocalSize();
        true_block_offsets[2] =
            true_block_offsets[1] + l2_dofTrueDof.GetTrueLocalSize();

        auto A = make_shared<MfemBlockOperator>(true_block_offsets);
        size_t local_nnz = 0;
        mfem::BlockVector prhs(true_block_offsets);
        {
            if (print_progress_report)
                std::cout << "-- Building operator on level " << start_level
                          << "...\n";

            // The blocks, managed here
            auto M = sequence[start_level]->ComputeMassOperator(uform);
            auto W = sequence[start_level]->ComputeMassOperator(pform);
            auto D = sequence[start_level]->GetDerivativeOperator(uform);

            auto B = ToUnique(Mult(*W, *D));
            auto Bt = ToUnique(Transpose(*B));

            auto pM = Assemble(hdiv_dofTrueDof, *M, hdiv_dofTrueDof);
            auto pB = Assemble(l2_dofTrueDof, *B, hdiv_dofTrueDof);
            auto pBt = Assemble(hdiv_dofTrueDof, *Bt, l2_dofTrueDof);

            hypre_ParCSRMatrix* tmp = *pM;
            local_nnz += tmp->diag->num_nonzeros;
            local_nnz += tmp->offd->num_nonzeros;
            tmp = *pB;
            local_nnz += tmp->diag->num_nonzeros;
            local_nnz += tmp->offd->num_nonzeros;
            tmp = *pBt;
            local_nnz += tmp->diag->num_nonzeros;
            local_nnz += tmp->offd->num_nonzeros;

            A->SetBlock(0,0,std::move(pM));
            A->SetBlock(0,1,std::move(pBt));
            A->SetBlock(1,0,std::move(pB));

            // Setup right-hand side
            prhs.GetBlock(0) = *rhs_u;
            prhs.GetBlock(1) = *rhs_p;

            if (print_progress_report)
                std::cout << "-- Built operator on level " << start_level
                          << ".\n"
                          <<"-- Assembled the linear system on level "
                          << start_level << ".\n\n";
        }

        // Report some stats on global problem size
        size_t global_height,global_width,global_nnz;
        {
            size_t local_height = A->Height(), local_width = A->Width();
            MPI_Reduce(&local_height,&global_height,1,GetMPIType(local_height),
                       MPI_SUM,0,comm);
            MPI_Reduce(&local_width,&global_width,1,GetMPIType(local_width),
                       MPI_SUM,0,comm);
            MPI_Reduce(&local_nnz,&global_nnz,1,GetMPIType<size_t>(local_nnz),
                       MPI_SUM,0,comm);
        }
        PARELAG_ASSERT(prhs.Size() == A->Height());

        //
        // Create the preconditioner
        //

        // Start with the solver library
        auto lib = SolverLibrary::CreateLibrary(
            master_list->Sublist("Preconditioner Library"));

        // Get the factory
        const std::string solver_type = prob_list.Get("Linear solver","Unknown");
        auto prec_factory = lib->GetSolverFactory(solver_type);
        auto solver_state = prec_factory->GetDefaultState();
        solver_state->SetDeRhamSequence(sequence[start_level]);
        solver_state->SetBoundaryLabels(
            std::vector<std::vector<int>>(2,std::vector<int>()));
        solver_state->SetForms({uform,pform});

        unique_ptr<mfem::Solver> solver;

        // Build the preconditioner
        if (print_progress_report)
            std::cout << "-- Building solver \"" << solver_type << "\"...\n";

        {
            Timer timer = TimeManager::AddTimer("Build Solver");
            solver = prec_factory->BuildSolver(A,*solver_state);
        }

        if (print_progress_report)
            std::cout << "-- Built solver \"" << solver_type << "\".\n";

//        mfem::BlockVector sol(block_offsets);
        mfem::BlockVector psol(true_block_offsets);
        psol = 0.;

        if (!myid)
            std::cout << '\n' << std::string(50,'*') << '\n'
                      << "*    Solving on level: " << start_level << '\n'
                      << "*              A size: "
                      << global_height << 'x' << global_width << '\n'
                      << "*               A NNZ: " << global_nnz << "\n*\n"
                      << "*              Solver: " << solver_type << "\n"
                      << std::string(50,'*') << '\n' << std::endl;

        if (print_progress_report)
            std::cout << "-- Solving system with " << solver_type << "...\n";
        {
            double local_norm = prhs.Norml2() * prhs.Norml2();
            double global_norm;
            MPI_Reduce(&local_norm,&global_norm,1,GetMPIType(local_norm),
                       MPI_SUM,0,comm);

            if (!myid)
                std::cout <<  "Initial residual norm: " << std::sqrt(global_norm)
                          << std::endl;
        }

        {
            Timer timer = TimeManager::AddTimer("Solve Linear System");
            solver->Mult(prhs,psol);
        }

        {
            mfem::Vector tmp(A->Height());
            A->Mult(psol,tmp);
            prhs -= tmp;
            double local_norm = prhs.Norml2() * prhs.Norml2();
            double global_norm;
            MPI_Reduce(&local_norm,&global_norm,1,GetMPIType(local_norm),
                       MPI_SUM,0,comm);

            if (!myid)
                std::cout << "Final residual norm: " << std::sqrt(global_norm)
                          << std::endl;
        }

        if (print_progress_report)
            std::cout << "-- Solver has exited.\n";
    }

//    {
//        const int form = 2;

//        const int nbdr = 6;
//        Array<int> ess_attr(nbdr);
//        ess_attr = 1;
//        ess_attr[0] = 0;
//        ess_attr[nbdr-1] = 0;

//        Vector ess_bc(nbdr), nat_bc(nbdr);
//        ess_bc = 0.0;
//        nat_bc = 0.0;
//        nat_bc[0] = -1.;

//        PWConstCoefficient ubdr(ess_bc);
//        PWConstCoefficient fbdr(nat_bc);

//        //  testUpscalingHdiv(sequence);
//        FiniteElementSpace * fespace = sequence[0]->FemSequence()->GetFeSpace(form);
//        auto b = make_unique<LinearForm>(fespace);
//        b->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fbdr));
//        b->Assemble();

//        auto lift = make_unique<GridFunction>(fespace);
//        lift->ProjectBdrCoefficient(ubdr, ess_attr);

//        DenseMatrix errors_L2_2(nLevels, nLevels);
//        errors_L2_2 = 0.0;
//        Vector norm_L2_2(nLevels);
//        norm_L2_2 = 0.;

//        DenseMatrix errors_div_2(nLevels, nLevels);
//        errors_div_2 = 0.0;
//        Vector norm_div_2(nLevels);
//        norm_div_2 = 0.;

//        DenseMatrix timings(nLevels, NSTAGES);
//        timings = 0.0;

//        Array2D<int> iter(nLevels, NSOLVERS);
//        iter = 0;
//        Array<int> ndofs(nLevels);
//        ndofs = 0;
//        Array<int> nnz(nLevels);
//        nnz = 0;

//        double tdiff;

//        Array<SparseMatrix *> allP(nLevels-1);
//        Array<SparseMatrix *> allD(nLevels);
//        std::vector<unique_ptr<HypreParMatrix>> allC(nLevels);

//        for (int i = 0; i < nLevels - 1; ++i)
//            allP[i] = sequence[i]->GetP(form);

//        for (int i = 0; i < nLevels; ++i)
//            allC[i] = sequence[i]->ComputeTrueDerivativeOperator(form-1);

//        for (int i = 0; i < nLevels; ++i)
//            allD[i] = sequence[i]->GetDerivativeOperator(form);

//        std::vector<unique_ptr<HypreParMatrix>> allP_bc(nLevels-1);
//        std::vector<unique_ptr<HypreParMatrix>> allC_bc(nLevels);

//        for (int i = 0; i < nLevels - 1; ++i)
//            allP_bc[i] = sequence[i]->ComputeTrueP(form, ess_attr);

//        for (int i = 0; i < nLevels; ++i)
//            allC_bc[i] = sequence[i]->ComputeTrueDerivativeOperator(form-1, ess_attr);

//        std::vector<unique_ptr<SparseMatrix>> Ml(nLevels);
//        std::vector<unique_ptr<SparseMatrix>> Wl(nLevels);

//        for (int k(0); k < nLevels; ++k)
//        {
//            chrono.Clear();
//            chrono.Start();
//            Ml[k] = sequence[k]->ComputeMassOperator(form);
//            Wl[k] = sequence[k]->ComputeMassOperator(form+1);
//            chrono.Stop();
//            tdiff = chrono.RealTime();
//            if (myid == 0 && reportTiming)
//                std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in "
//                          << tdiff << "s.\n";
//            timings(k, ASSEMBLY) += tdiff;
//        }

//        std::vector<unique_ptr<Vector>> rhs(nLevels);
//        std::vector<unique_ptr<Vector>> ess_data(nLevels);
//        rhs[0] = std::move(b);
//        ess_data[0] = std::move(lift);
//        for (int i = 0; i < nLevels-1; ++i)
//        {
//            rhs[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(form));
//            ess_data[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(form));
//            sequence[i]->GetP(form)->MultTranspose(*(rhs[i]), *(rhs[i+1]));
//            sequence[i]->GetPi(form)->ComputeProjector();
//            sequence[i]->GetPi(form)->GetProjectorMatrix().Mult(
//                *(ess_data[i]), *(ess_data[i+1]));
//        }

//        std::vector<unique_ptr<Vector>> sol_AMGe(nLevels);
//        std::vector<unique_ptr<Vector>> sol_ADS(nLevels);
//        std::vector<unique_ptr<Vector>> help(nLevels);

//        for (int k(0); k < nLevels; ++k)
//        {
//            sol_AMGe[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
//            *(sol_AMGe[k]) = 0.;
//            sol_ADS[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
//            *(sol_ADS[k]) = 0.;
//            help[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
//            *(help[k]) = 0.;
//        }

//        for (int k(0); k < nLevels; ++k)
//        {
//            chrono.Clear();
//            chrono.Start();

//            unique_ptr<HypreParMatrix> pA;
//            const SharingMap & hdiv_dofTrueDof(
//                sequence[k]->GetDofHandler(nDimensions-1)->GetDofTrueDof() );
//            {
//                auto M = Ml[k].get();
//                auto W = Wl[k].get();
//                auto D = allD[k];

//                auto A = ToUnique(Add(*M, *ExampleRAP(*D, *W, *D)));

//                const int nlocdofs = A->Height();
//                Array<int> marker(nlocdofs);
//                marker = 0;
//                sequence[k]->GetDofHandler(form)->MarkDofsOnSelectedBndr(
//                    ess_attr, marker);

//                for (int mm = 0; mm < nlocdofs; ++mm)
//                    if (marker[mm])
//                        A->EliminateRowCol(
//                            mm, ess_data[k]->operator ()(mm), *(rhs[k]) );

//                pA = Assemble(hdiv_dofTrueDof, *A, hdiv_dofTrueDof);
//            }
//            Vector prhs( hdiv_dofTrueDof.GetTrueLocalSize() );
//            hdiv_dofTrueDof.Assemble(*(rhs[k]), prhs);

//            elag_assert(prhs.Size() == pA->Height() );

//            chrono.Stop();
//            tdiff = chrono.RealTime();
//            if (myid == 0 && reportTiming)
//                std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in "
//                          << tdiff << "s.\n";
//            timings(k, ASSEMBLY) += tdiff;

//            ndofs[k] = pA->GetGlobalNumRows();
//            nnz[k] = pA->NNZ();

//            // AMGe SOLVER not implemented
//            {
//                (*sol_AMGe[k]) = 0.;
//                timings(k, PRECONDITIONER_AMGe) = 0.;
//                timings(k,SOLVER_AMGe) = 0.;
//                iter(k,AMGe) = 0;
//            }

//            // default linear solver options
//            constexpr int print_iter = 0;
//            constexpr int max_num_iter = 500;
//            constexpr double rtol = 1e-6;
//            constexpr double atol = 1e-12;

//            iter(k,ADS) = UpscalingHypreSolver(
//                form, pA.get(), prhs,
//                sequence[k].get(),
//                k, PRECONDITIONER_ADS, SOLVER_ADS,
//                print_iter, max_num_iter, rtol, atol,
//                timings, hdiv_dofTrueDof, *(sol_ADS[k]), reportTiming);

//            auto& sol = sol_ADS;
//            // ERROR NORMS
//            {
//                *(help[k]) = *(sol[k]);
//                for (int j = k; j > 0; --j)
//                    allP[j-1]->Mult( *(help[j]), *(help[j-1]) );

//                norm_L2_2(k) = Ml[k]->InnerProduct(*(sol[k]), *(sol[k]) );
//                Vector dsol( allD[k]->Height() );
//                allD[k]->Mult(*(sol[k]), dsol );
//                norm_div_2(k) = Wl[k]->InnerProduct(dsol, dsol );

//                for (int j(0); j < k; ++j)
//                {
//                    if (help[j]->Size() != sol[j]->Size() || sol[j]->Size() != allD[j]->Width() )
//                        mfem_error("size don't match \n");

//                    int size = sol[j]->Size();
//                    int dsize = allD[j]->Height();
//                    Vector u_H(help[j]->GetData(), size);
//                    Vector u_h(sol[j]->GetData(), size);
//                    Vector u_diff(size), du_diff(dsize);
//                    u_diff = 0.; du_diff = 0.;

//                    subtract(u_H, u_h, u_diff);
//                    allD[j]->Mult(u_diff, du_diff);

//                    errors_L2_2(k,j) =  Ml[j]->InnerProduct(u_diff, u_diff);
//                    errors_div_2(k,j) =  Wl[j]->InnerProduct(du_diff, du_diff);
//                }
//            }

//            //VISUALIZE SOLUTION
////            if (do_visualize)
////            {
////                MultiVector tmp(sol[k]->GetData(), 1, sol[k]->Size() );
////                sequence[k]->show(form, tmp);
////            }
//        }

//        ReduceAndOutputUpscalingErrors(errors_L2_2, norm_L2_2,
//                                       errors_div_2, norm_div_2);
//    }


    if (print_progress_report)
        std::cout << "-- Good bye!\n\n";

    return EXIT_SUCCESS;
}
