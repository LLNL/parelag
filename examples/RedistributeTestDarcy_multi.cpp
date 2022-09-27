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
    const bool visualize = output_list.Get("Visualize solution",false);

    const bool print_progress_report = (!myid && show_progress);

    if (print_progress_report)
        std::cout << "\n-- Hello!\n"
                  << "-- Welcome to RedistributeTestDarcy!\n\n";

    std::ostringstream mesh_msg;
    if (!myid)
    {
        mesh_msg << '\n' << std::string(50,'*') << '\n'
                 << "*  Mesh: " << meshfile << "\n*\n";
    }

    Array<int> par_partitioning;
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

        par_partitioning.MakeRef(mesh->GeneratePartitioning(num_ranks), mesh->GetNE());
        pmesh = make_shared<ParMesh>(comm, *mesh, par_partitioning);

        if (pmesh && print_progress_report)
            std::cout << "-- Built parallel mesh successfully.\n";
    }

    const int nDimensions = pmesh->Dimension();

    const int nGeometricLevels = par_ref_levels+1;
    std::vector<int> num_elements(nGeometricLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        num_elements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    num_elements[0] = pmesh->GetNE();

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

    const int uform = nDimensions - 1;
    const int pform = nDimensions;

    ConstantCoefficient coeffL2(1.);
    ConstantCoefficient coeffHdiv(1.);

    SequenceHierarchy hierarchy(pmesh, prob_list, print_progress_report);
    hierarchy.SetCoefficient(pform, coeffL2, false);
    hierarchy.SetCoefficient(uform, coeffHdiv, true);
    hierarchy.Build(move(num_elements));
    auto& sequence = hierarchy.GetDeRhamSequences();

    {
        PARELAG_ASSERT(start_level < sequence.size());

        auto DRSequence_FE = sequence[0]->FemSequence();
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
                hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(ii+1))[ii+1]->GetNumTrueDofs(uform) );
            auto tmp_p = make_unique<mfem::Vector>(
                hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(ii+1))[ii+1]->GetNumTrueDofs(pform) );
            // sequence[ii]->GetTrueP(uform).MultTranspose(*rhs_u,*tmp_u);
            // sequence[ii]->ApplyTruePTranspose(uform,*rhs_u,*tmp_u);
            hierarchy.ApplyTruePTranspose(ii, uform, *rhs_u, *tmp_u);
            // sequence[ii]->GetTrueP(pform).MultTranspose(*rhs_p,*tmp_p);
            // sequence[ii]->ApplyTruePTranspose(pform,*rhs_p,*tmp_p);
            hierarchy.ApplyTruePTranspose(ii, pform,*rhs_p,*tmp_p);
            rhs_u = std::move(tmp_u);
            rhs_p = std::move(tmp_p);
        }

        const SharingMap& hdiv_dofTrueDof
            = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetDofHandler(uform)->GetDofTrueDof();
        const SharingMap& l2_dofTrueDof
            = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetDofHandler(pform)->GetDofTrueDof();

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
            auto M = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->ComputeMassOperator(uform);
            auto W = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->ComputeMassOperator(pform);
            auto D = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetDerivativeOperator(uform);

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
        ParameterList& prec_list = master_list->Sublist("Preconditioner Library");
        auto lib = SolverLibrary::CreateLibrary(prec_list);

        // Get the factory
        const std::string solver_type = prob_list.Get("Linear solver","Unknown");
        auto prec_factory = lib->GetSolverFactory(solver_type);
        const int rescale_iter = prec_list.Sublist(solver_type).Sublist(
                "Solver Parameters").Get<int>("RescaleIteration");

        auto solver_state = prec_factory->GetDefaultState();
        solver_state->SetDeRhamSequence(hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]);
        solver_state->SetBoundaryLabels(
            std::vector<std::vector<int>>(2,std::vector<int>()));
        solver_state->SetForms({uform,pform});

        // These are for hybridization solver
        solver_state->SetExtraParameter("IsSameOrient",(start_level>0));
        solver_state->SetExtraParameter("ActOnTrueDofs",true);
        solver_state->SetExtraParameter("RescaleIteration", rescale_iter);

        unique_ptr<mfem::Solver> solver;

        // Build the preconditioner
        if (print_progress_report)
            std::cout << "-- Building solver \"" << solver_type << "\"...\n";

        {
            Timer timer = TimeManager::AddTimer("Build Solver");
            solver = prec_factory->BuildSolver(A,*solver_state);
            solver->iterative_mode=false;
        }

        if (print_progress_report)
            std::cout << "-- Built solver \"" << solver_type << "\".\n";

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

        solver->Mult(prhs,psol);

        {
            mfem::Vector tmp(A->Height());
            A->Mult(psol,tmp);
            prhs -= tmp;
            double local_norm = prhs.Norml2() * prhs.Norml2();
            double global_norm;
            MPI_Reduce(&local_norm,&global_norm,1,GetMPIType(local_norm),
                       MPI_SUM,0,comm);

            if (!myid)
            {
                std::cout << "Final residual norm: " << std::sqrt(global_norm)
                          << std::endl;
                auto hybrid_solver = dynamic_cast<HybridizationSolver*>(solver.get());
                if (hybrid_solver)
                {
                    std::cout << "Number of iterations: " << hybrid_solver->GetNumIters()
                              << std::endl;
                }
            }
        }

        if (visualize)
        {
            int k_l = hierarchy.GetRedistributionIndex(start_level);
            int level_numgroups = hierarchy.GetNumGlobalCopies(k_l);
            MultiVector u(psol.GetData(), 1, psol.BlockSize(0));
            MultiVector p(psol.GetBlock(1).GetData(), 1, psol.BlockSize(1));
            for (int groupid(0); groupid < level_numgroups; groupid++)
            {
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, uform, u);
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, pform, p);
            }
        }

        if (print_progress_report)
            std::cout << "-- Solver has exited.\n";
    }

    if (print_progress_report)
        std::cout << "-- Good bye!\n\n";

    return EXIT_SUCCESS;
}
