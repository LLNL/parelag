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

#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"
#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "utilities/ParELAG_TimeManager.hpp"
#include "utilities/ParELAG_SimpleXMLParameterListReader.hpp"
#include "utilities/MPIDataTypes.hpp"

#include "testing_helpers/Build3DHexMesh.hpp"
#include "testing_helpers/CreateDarcyParameterList.hpp"

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

    int num_ranks, myid;
    MPI_Comm_size(comm, &num_ranks);
    MPI_Comm_rank(comm, &myid);

    // Get options from command line
    const char *xml_file_c = "BuildTestParameters";
    double W_weight = 0.0;
    mfem::OptionsParser args(argc, argv);
    args.AddOption(&xml_file_c, "-f", "--xml-file",
                   "XML parameter list.");
    // The constant weight in the system [M B^T; B -(W_weight*W)]
    args.AddOption(&W_weight, "-w", "--L2mass-weight",
                   "The constant weight in the system [M B^T; B -(W_weight*W)]");
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

    // The order of the finite elements
    const int feorder = prob_list.Get("Finite element order", 0);

    // The order of the polynomials to include in the space
    const int upscalingOrder = prob_list.Get("Upscaling order", 0);

    const bool deformation = prob_list.Get("Deformation",false);

    ParameterList& output_list = master_list->Sublist("Output control");
    const bool print_time = output_list.Get("Print timings",true);
    const bool visualize = output_list.Get("Visualize solution",false);
    const bool show_progress = output_list.Get("Show progress",true);

    const bool print_progress_report = (!myid && show_progress);

    if (print_progress_report)
        std::cout << "\n-- Hello!\n"
                  << "-- Welcome to MultigridTestDarcy!\n\n";

    std::ostringstream mesh_msg;
    if (!myid)
    {
        mesh_msg << '\n' << std::string(50,'*') << '\n'
                 << "*  Mesh: " << meshfile << "\n*\n" << std::boolalpha
                 << "*           Deformation: " << deformation << '\n'
                 << "*              FE order: " << feorder << '\n'
                 << "*       Upscaling order: " << upscalingOrder << '\n'
                 << "*\n";
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
                      << " times.\n\n";
        }

        if (!myid)
        {
            mesh_msg << "*    Serial refinements: " << ser_ref_levels << '\n'
                     << "*      Coarse mesh size: " << mesh->GetNE() << '\n'
                     << "*\n";
        }

        if (print_progress_report)
            std::cout << "-- Building parallel mesh...\n";

        pmesh = make_shared<ParMesh>(comm, *mesh);

        if (pmesh && print_progress_report)
            std::cout << "-- Built parallel mesh successfully.\n\n";
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

    if (deformation)
    {
        if (print_progress_report)
            std::cout << "-- Transforming mesh...\n";
        if (nDimensions == 2)
            pmesh->Transform(deformation2D);
        else
            pmesh->Transform(deformation3D);
        if (print_progress_report)
            std::cout << "-- Transforming mesh...\n\n";
    }

    {
        size_t my_num_elmts = pmesh->GetNE(), global_num_elmts;
        MPI_Reduce(&my_num_elmts,&global_num_elmts,1,GetMPIType<size_t>(0),
                   MPI_SUM,0,comm);

        if (!myid)
        {
            mesh_msg << "*  Parallel refinements: " << par_ref_levels << '\n'
                     << "*        Fine Mesh Size: " << global_num_elmts << '\n'
                     << "*          Total levels: " << nLevels << '\n'
                     << std::string(50,'*') << '\n' << std::endl;
        }
    }

    if (!myid)
        std::cout << mesh_msg.str();

    if (print_progress_report)
        std::cout << "-- Agglomerating topology to " << nLevels
                  << " levels...\n";

    ConstantCoefficient coeffL2(1.);
    ConstantCoefficient coeffHdiv(1.);

    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    MFEMRefinedMeshPartitioner partitioner(nDimensions);
    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);
    {
        Timer outer = TimeManager::AddTimer("Mesh Agglomeration -- Total");
        {
            Timer inner = TimeManager::AddTimer("Mesh Agglomeration -- Level 0");

            topology[0] = make_shared<AgglomeratedTopology>(pmesh,nDimensions);
        }

        for (int ilevel = 0; ilevel < nLevels-1; ++ilevel)
        {
            Array<int> partitioning(
                topology[ilevel]->GetNumberLocalEntities(at_elem));

            std::ostringstream timer_name;
            timer_name << "Mesh Agglomeration -- Level " << ilevel+1;
            Timer inner = TimeManager::AddTimer(timer_name.str());

            partitioner.Partition(
                topology[ilevel]->GetNumberLocalEntities(at_elem),
                level_nElements[ilevel+1],
                partitioning);
            topology[ilevel+1] =
                topology[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0,
                                                           nDimensions == 2 ? 0 : 2);
        }
    }

    if (print_progress_report)
        std::cout << "-- Successfully agglomerated topology.\n\n"
                  << "-- Building the fine-grid DeRhamSequence...\n";

    const int uform = nDimensions - 1;
    const int pform = nDimensions;

    DeRhamSequenceFE * DRSequence_FE;
    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());
    {
        Timer timer = TimeManager::AddTimer(
            "DeRhamSequence Construction -- Level 0");
        Timer tot_timer = TimeManager::AddTimer(
            "DeRhamSequence Construction -- Total");

        if (nDimensions == 3)
        {
            sequence[0] = make_shared<DeRhamSequence3D_FE>(
                        topology[0], pmesh.get(), feorder);
        }
        else
        {
            sequence[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(
                        topology[0], pmesh.get(), feorder);
        }

        DRSequence_FE = sequence[0]->FemSequence();
        PARELAG_ASSERT(DRSequence_FE);

        const int jFormStart = nDimensions - 1;
        sequence[0]->SetjformStart(jFormStart);

        DRSequence_FE->ReplaceMassIntegrator(
            at_elem, pform, make_unique<MassIntegrator>(coeffL2), false);
        DRSequence_FE->ReplaceMassIntegrator(
            at_elem, uform, make_unique<VectorFEMassIntegrator>(coeffHdiv), true);

        if (print_progress_report)
            std::cout << "-- Building targets...\n";

        mfem::Array<mfem::Coefficient *> L2coeff;
        mfem::Array<mfem::VectorCoefficient *> Hdivcoeff;
        fillVectorCoefficientArray(nDimensions, upscalingOrder, Hdivcoeff);
        fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);

        std::vector<unique_ptr<MultiVector>>
            targets(sequence[0]->GetNumberOfForms());
        int jform(nDimensions-1);

        targets[jform] =
            DRSequence_FE->InterpolateVectorTargets(jform,Hdivcoeff);
        ++jform;

        targets[jform] =
            DRSequence_FE->InterpolateScalarTargets(jform,L2coeff);
        ++jform;

        freeCoeffArray(L2coeff);
        freeCoeffArray(Hdivcoeff);

        mfem::Array<MultiVector*> targets_in(targets.size());
        for (int ii = 0; ii < targets_in.Size(); ++ii)
            targets_in[ii] = targets[ii].get();

        if (print_progress_report)
            std::cout << "-- Setting targets...\n";

        sequence[0]->SetTargets(targets_in);
    }

    if (print_progress_report)
        std::cout << "-- Built fine grid DeRhamSequence.\n\n"
                  << "-- Coarsening the DeRhamSequence to all levels...\n";

    constexpr double tolSVD = 1e-9;
    for (int i(0); i < nLevels-1; ++i)
    {
        std::ostringstream timer_name;
        timer_name << "DeRhamSequence Construction -- Level " << i+1;
        Timer timer = TimeManager::AddTimer(timer_name.str());
        Timer tot_timer = TimeManager::GetTimer(
            "DeRhamSequence Construction -- Total");

        sequence[i]->SetSVDTol(tolSVD);
        sequence[i+1] = sequence[i]->Coarsen();
    }

    if (print_progress_report)
        std::cout << "-- Coarsened DeRhamSequence to all levels.\n\n"
                  << "-- Assembling the linear system on level "
                  << start_level << "...\n";

    FiniteElementSpace * ufespace = DRSequence_FE->GetFeSpace(uform);
    FiniteElementSpace * pfespace = DRSequence_FE->GetFeSpace(pform);

    auto bform = make_unique<mfem::LinearForm>(ufespace);
    ConstantCoefficient fbdr(0.);
    bform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fbdr));
    bform->Assemble();

    auto qform = make_unique<mfem::LinearForm>(pfespace);
    ConstantCoefficient source(1.);
    qform->AddDomainIntegrator(new DomainLFIntegrator(source));
    qform->Assemble();

    unique_ptr<mfem::Vector> rhs_u = std::move(bform);
    unique_ptr<mfem::Vector> rhs_p = std::move(qform);

    // Project rhs down to the level of interest
    for (int ii = 0; ii < start_level; ++ii)
    {
        auto tmp_u = make_unique<mfem::Vector>(
            sequence[ii+1]->GetNumberOfDofs(uform) );
        auto tmp_p = make_unique<mfem::Vector>(
            sequence[ii+1]->GetNumberOfDofs(pform) );
        sequence[ii]->GetP(uform)->MultTranspose(*rhs_u,*tmp_u);
        sequence[ii]->GetP(pform)->MultTranspose(*rhs_p,*tmp_p);
        rhs_u = std::move(tmp_u);
        rhs_p = std::move(tmp_p);
    }

    // Create the parallel linear system
    const SharingMap& hdiv_dofTrueDof
        = sequence[start_level]->GetDofHandler(uform)->GetDofTrueDof();
    const SharingMap& l2_dofTrueDof
        = sequence[start_level]->GetDofHandler(pform)->GetDofTrueDof();

    mfem::Array<int> block_offsets(3);
    block_offsets[0] = 0;
    block_offsets[1] = sequence[start_level]->GetNumberOfDofs(uform);
    block_offsets[2] = block_offsets[1] +
        sequence[start_level]->GetNumberOfDofs(pform);

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
        unique_ptr<HypreParMatrix> pM;
        unique_ptr<HypreParMatrix> pB;
        unique_ptr<HypreParMatrix> pBt;
        unique_ptr<HypreParMatrix> pW;

        {
            auto M = sequence[start_level]->ComputeMassOperator(uform),
                W = sequence[start_level]->ComputeMassOperator(pform);
            auto D = sequence[start_level]->GetDerivativeOperator(uform);

            auto B = ToUnique(Mult(*W, *D));
            auto Bt = ToUnique(Transpose(*B));

            pM = Assemble(hdiv_dofTrueDof, *M, hdiv_dofTrueDof);
            pB = Assemble(l2_dofTrueDof, *B, hdiv_dofTrueDof);
            pBt = Assemble(hdiv_dofTrueDof, *Bt, l2_dofTrueDof);

            if (W_weight > 0)
            {
                *W *= -W_weight;
                pW = Assemble(l2_dofTrueDof, *W, l2_dofTrueDof);
            }

            hypre_ParCSRMatrix* tmp = *pM;
            local_nnz += tmp->diag->num_nonzeros;
            local_nnz += tmp->offd->num_nonzeros;
            tmp = *pB;
            local_nnz += tmp->diag->num_nonzeros;
            local_nnz += tmp->offd->num_nonzeros;
            tmp = *pBt;
            local_nnz += tmp->diag->num_nonzeros;
            local_nnz += tmp->offd->num_nonzeros;
        }

        A->SetBlock(0,0,std::move(pM));
        A->SetBlock(0,1,std::move(pBt));
        A->SetBlock(1,0,std::move(pB));
        if (W_weight > 0)
            A->SetBlock(1,1,std::move(pW));

        // Setup right-hand side
        hdiv_dofTrueDof.Assemble(*rhs_u, prhs.GetBlock(0));
        l2_dofTrueDof.Assemble(*rhs_p, prhs.GetBlock(1));

        if (print_progress_report)
            std::cout << "-- Built operator on level " << start_level
                      << ".\n"
                      <<"-- Assembled the linear system on level "
                      << start_level << ".\n\n";
    }

    BlockVector rhs(block_offsets);
    rhs.GetBlock(0) = *rhs_u;
    rhs.GetBlock(1) = *rhs_p;

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
    const std::string prec_type = "Hybridization-Darcy";
    auto prec_factory = lib->GetSolverFactory(prec_type);
    const std::string solver_type = prec_list.Sublist(prec_type).Sublist(
    		"Solver Parameters").Get<std::string>("Solver");
    ParameterList& solver_list =
    		prec_list.Sublist(solver_type);
    const int max_num_iter = solver_list.Sublist(
    		"Solver Parameters").Get<int>("Maximum iterations");
    const double rtol = solver_list.Sublist(
    		"Solver Parameters").Get<double>("Relative tolerance");
    const double atol = solver_list.Sublist(
    		"Solver Parameters").Get<double>("Absolute tolerance");
    const int rescale_iter = prec_list.Sublist(prec_type).Sublist(
            "Solver Parameters").Get<int>("RescaleIteration");

    auto solver_state = prec_factory->GetDefaultState();
    solver_state->SetDeRhamSequence(sequence[start_level]);
    solver_state->SetBoundaryLabels(
    		std::vector<std::vector<int>>(2,std::vector<int>()));
    solver_state->SetForms({uform,pform});

    // Whether the element H(div) dofs have same orientation on shared facet
    solver_state->SetExtraParameter("IsSameOrient",(start_level>0));

    // The constant weight in the system [M B^T; B -(W_weight*W)]
    solver_state->SetExtraParameter("L2MassWeight",W_weight);

    // Number of smoothing steps in the generation of the rescaling vector
    // Such a rescaling can potentially improve the efficiency of the solver
    // If it is set to 0, then the original hybridized system is solved
    solver_state->SetExtraParameter("RescaleIteration",rescale_iter);

    // Scales of element H(div) mass matrices for the problem being solved
    // If not provided, the scale is treated as 1.0.
    auto elemMatrixScaling = make_shared<mfem::Vector>(
                topology[start_level]->GetNumberLocalEntities(at_elem));
    *elemMatrixScaling = 1.0;
    solver_state->SetVector("elemMatrixScaling",elemMatrixScaling);

    unique_ptr<mfem::Solver> hybrid_solver;
    {
    	Timer timer = TimeManager::AddTimer("Build Solver");
    	hybrid_solver = prec_factory->BuildSolver(A,*solver_state);
    	hybrid_solver->iterative_mode=false;
    }

    if (print_progress_report)
    	std::cout <<  "-- Solver has exited.\n";

    if (!myid)
        std::cout << '\n' << std::string(50,'*') << '\n'
                  << "*    Solving on level: " << start_level << '\n'
                  << "*              A size: "
                  << global_height << 'x' << global_width << '\n'
                  << "*               A NNZ: " << global_nnz << "\n*\n"
                  << "*              Solver: " << prec_type<< "\n"
                  << "*      Preconditioner: " << solver_type << '\n'
                  << "*  Relative Tolerance: " << rtol << '\n'
                  << "*  Absolute Tolerance: " << atol << '\n'
                  << "*  Maximum Iterations: " << max_num_iter << '\n'
                  << std::string(50,'*') << '\n' << std::endl;

    if (print_progress_report)
        std::cout << "-- Solving system with hybridization...\n";
    {
        double local_norm = prhs.Norml2() * prhs.Norml2();
        double global_norm;
        MPI_Reduce(&local_norm,&global_norm,1,GetMPIType(local_norm),
                   MPI_SUM,0,comm);

        if (!myid)
            std::cout <<  "Initial residual norm: " << std::sqrt(global_norm)
                      << std::endl;
    }

    mfem::BlockVector sol(block_offsets);
    sol = 0.;
    {
    	Timer timer = TimeManager::AddTimer("Solve Hybridized System");
    	hybrid_solver->Mult(rhs,sol);
    }

    {
        mfem::Vector tmp(A->Height());
        mfem::BlockVector psol(true_block_offsets);
        hdiv_dofTrueDof.IgnoreNonLocal(sol.GetBlock(0), psol.GetBlock(0));
        l2_dofTrueDof.IgnoreNonLocal(sol.GetBlock(1), psol.GetBlock(1));
        A->Mult(psol,tmp);
        prhs -= tmp;
        double local_norm = prhs.Norml2() * prhs.Norml2();
        double global_norm;
        MPI_Reduce(&local_norm,&global_norm,1,GetMPIType(local_norm),
                   MPI_SUM,0,comm);

        if (!myid)
            std::cout <<  "Final residual norm: " << std::sqrt(global_norm)
                      << std::endl;
    }

    if (print_time)
        TimeManager::Print(std::cout);

    if (visualize)
    {
        MultiVector u(sol.GetData(), 1, sol.GetBlock(0).Size() );
        sequence[start_level]->show(uform, u);
        MultiVector p(sol.GetBlock(1).GetData(), 1, sol.GetBlock(1).Size() );
        sequence[start_level]->show(pform, p);
    }

    if (print_progress_report)
        std::cout << "-- Good bye!\n\n";

    return EXIT_SUCCESS;
}
