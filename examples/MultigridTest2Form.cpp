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
#include <random>

#include <mpi.h>

#include "elag.hpp"

#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "utilities/MPIDataTypes.hpp"
#include "utilities/ParELAG_SimpleXMLParameterListReader.hpp"
#include "utilities/ParELAG_TimeManager.hpp"

#include "testing_helpers/Build3DHexMesh.hpp"
#include "testing_helpers/Create2FormParameterList.hpp"

using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

// Exact solution
void F_exact(const mfem::Vector &, mfem::Vector &);

// Exact RHS
void f_exact(const mfem::Vector &, mfem::Vector &);

double freq=1.0, kappa;

int main (int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_ranks, myid;
    MPI_Comm_size(comm, &num_ranks);
    MPI_Comm_rank(comm, &myid);

    Timer main_timer = TimeManager::AddTimer("Main()");

    // Get options from command line
    const char *xml_file_c = "BuildTestParameters";
    mfem::OptionsParser args(argc, argv);
    args.AddOption(&xml_file_c, "-f", "--xml-file",
                   "XML parameter list.");
    args.Parse();
    PARELAG_ASSERT(args.Good());
    std::string xml_file(xml_file_c);

    // Read the parameter list from file
    unique_ptr<ParameterList> master_list;
    if (xml_file == "BuildTestParameters")
    {
        master_list = testhelpers::Create2FormTestParameters();
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
    const int start_level = prob_list.Get("Start level",0);
    int stop_level = prob_list.Get("Stop level",start_level);

    if (stop_level < 0) stop_level = par_ref_levels;

    // The order of the finite elements
    const int feorder = prob_list.Get("Finite element order", 0);

    // The order of the polynomials to include in the space
    const int upscalingOrder = prob_list.Get("Upscaling order", 0);

    ParameterList& output_list = master_list->Sublist("Output control");
    const bool print_time = output_list.Get("Print timings",true);
    const bool visualize = output_list.Get("Visualize solution",false);
    const bool show_progress = output_list.Get("Show progress",true);

    const bool print_progress_report = (!myid && show_progress);

    if (print_progress_report)
        std::cout << "\n-- Hello!\n"
                  << "-- Welcome to MultigridTest2Form_ParameterList!\n\n";

    freq = prob_list.Get("Frequency",freq);
    kappa = freq*M_PI;

    std::ostringstream mesh_msg;
    if (!myid)
    {
        mesh_msg << '\n' << std::string(50,'*') << '\n'
                 << "*  Mesh: " << meshfile << "\n*\n"
#if 0
                 << "*             Frequency: " << freq << '\n'
#endif
                 << "*              FE order: " << feorder << '\n'
                 << "*       Upscaling order: " << upscalingOrder << '\n'
                 << "*\n";
    }

    // 2. Read the (serial) mesh from the given mesh file and
    //    uniformly refine it.
    shared_ptr<mfem::ParMesh> pmesh;
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
                      << " times.\n\n"
                      << "-- Building parallel mesh...\n";
        }

        if (!myid)
        {
            mesh_msg << "*    Serial refinements: " << ser_ref_levels << '\n'
                     << "*      Coarse mesh size: " << mesh->GetNE() << '\n'
                     << "*\n";
        }

        pmesh = make_shared<mfem::ParMesh>(comm,*mesh);

        if (pmesh && print_progress_report)
            std::cout << "-- Built parallel mesh successfully.\n\n";
    }

    // Mark all boundary attributes as essential
    std::vector<mfem::Array<int>> ess_attr(1);
    ess_attr[0].SetSize(pmesh->bdr_attributes.Max());
    ess_attr[0] = 1;

    // Refine the mesh in parallel
    const int nDimensions = pmesh->Dimension();
    const int nLevels = par_ref_levels+1;

    PARELAG_ASSERT(nDimensions == 3);

    // Check the start_level
    PARELAG_ASSERT(start_level < nLevels);

    std::vector<int> level_nElements(nLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }

    if (print_progress_report)
        std::cout << "-- Refined mesh in parallel " << par_ref_levels
                  << " times.\n\n";

    level_nElements[0] = pmesh->GetNE();

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

    // Tet meshes need to be reoriented before high-order nedelec
    // spaces can be defined on them...
    pmesh->ReorientTetMesh();

    if (print_progress_report)
        std::cout << "-- Agglomerating topology to " << nLevels
                  << " levels...\n";

    // 3. Partition the topology
    constexpr auto AT_elem = AgglomeratedTopology::ELEMENT;
    MFEMRefinedMeshPartitioner partitioner(nDimensions);
    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);
    {
        Timer outer = TimeManager::AddTimer("Mesh Agglomeration -- Total");
        {
            Timer inner = TimeManager::AddTimer("Mesh Agglomeration -- Level 0");

            topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
        }

        for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
        {
            mfem::Array<int> partitioning(
                topology[ilevel]->GetNumberLocalEntities(AT_elem) );

            std::ostringstream timer_name;
            timer_name << "Mesh Agglomeration -- Level " << ilevel+1;
            Timer inner = TimeManager::AddTimer(timer_name.str());

            partitioner.Partition(
                topology[ilevel]->GetNumberLocalEntities(AT_elem),
                level_nElements[ilevel+1],
                partitioning);
            topology[ilevel+1] =
                topology[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);
        }
    }

    if (print_progress_report)
        std::cout << "-- Successfully agglomerated topology.\n\n";

#ifdef ParELAG_ENABLE_PARMETIS
    int num_subdomains = prob_list.Get("Coefficient subdomains",4);

    if (print_progress_report)
        std::cout << "-- Generating " << num_subdomains
                  << " random coefficients...\n";

    mfem::Array<int> Coeff_Partition(pmesh->GetNE());
    mfem::PWConstCoefficient pw_coeff(num_subdomains > 0 ? num_subdomains : 1);

    if (num_subdomains > 1)
    {
        ParmetisGraphPartitioner pm_partitioner;
        pm_partitioner.doPartitionEqualWeight(
            *topology[0]->GlobalElementElementTable(),
            num_subdomains, Coeff_Partition);

        for (int elem = 0; elem < Coeff_Partition.Size(); ++elem)
            pmesh->GetElement(elem)->SetAttribute(Coeff_Partition[elem]+1);

        // interval = [0,1]
        std::mt19937_64 gen(13);
        std::uniform_real_distribution<double> dist(
            0.0,std::nextafter(1.0, std::numeric_limits<double>::max()));

        double min = 1.0, max = 0.0;
        std::vector<double> coeff_vec;
        for (int subd = 0; subd < num_subdomains; ++subd)
        {
            auto value = dist(gen);

            coeff_vec.push_back(value);

            pw_coeff(subd+1) = value;
            if (value > max)
                max = value;
            if (value < min)
                min = value;
        }

        if (!myid)
            std::cout << '\n' << std::string(50,'*') << '\n'
                      << "*    Num subdomains: " << num_subdomains << '\n'
                      << "*  Ratio of min/max: " << min/max << '\n'
                      << std::string(50,'*') << '\n'
                      << "Coefficients:\n" << coeff_vec << '\n' << std::endl;

    }
    else
        pw_coeff(1) = 1.0;

    if (print_progress_report)
        std::cout << "-- Successfully generated coefficients.\n\n";
#endif

    // Start setting the problem parameters
    mfem::ConstantCoefficient coeffH1(1.);
    mfem::ConstantCoefficient coeffDer(1.);
    mfem::ConstantCoefficient alpha(1.0);
    mfem::ConstantCoefficient beta(1.0);
    mfem::VectorFunctionCoefficient f(nDimensions, f_exact);

    // 4. Construct the DeRham hierarchy
    if (print_progress_report)
        std::cout << "-- Building the fine-grid DeRhamSequence...\n";

    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());

    const int nForms = nDimensions+1;
    DeRhamSequenceFE * DRSequence_FE;
    {
        Timer timer = TimeManager::AddTimer(
            "DeRhamSequence Construction -- Level 0");
        Timer tot_timer = TimeManager::AddTimer(
            "DeRhamSequence Construction -- Total");
        if(nDimensions == 3)
            sequence[0] = make_shared<DeRhamSequence3D_FE>(
                topology[0], pmesh.get(), feorder);
        else
            sequence[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(
                topology[0], pmesh.get(), feorder);

        DRSequence_FE = sequence[0]->FemSequence();
        PARELAG_ASSERT(DRSequence_FE);

        std::vector<unique_ptr<MultiVector>> targets(nForms);

        constexpr int jFormStart = 0;
        sequence[0]->SetjformStart(jFormStart);

#ifdef ParELAG_ENABLE_PARMETIS
        DRSequence_FE->ReplaceMassIntegrator(
            AT_elem,nForms-1,make_unique<mfem::MassIntegrator>(pw_coeff),false);
#else
        DRSequence_FE->ReplaceMassIntegrator(
            AT_elem,nForms-1,make_unique<mfem::MassIntegrator>(alpha),false);
#endif

        DRSequence_FE->ReplaceMassIntegrator(
            AT_elem,nForms-2,
            make_unique<mfem::VectorFEMassIntegrator>(beta),true);

        if (print_progress_report)
            std::cout << "-- Building targets...\n";

        mfem::Array<mfem::Coefficient *> L2coeff;
        mfem::Array<mfem::VectorCoefficient *> Hdivcoeff;
        mfem::Array<mfem::VectorCoefficient *> Hcurlcoeff;
        mfem::Array<mfem::Coefficient *> H1coeff;
        fillVectorCoefficientArray(nDimensions, upscalingOrder, Hcurlcoeff);
        fillVectorCoefficientArray(nDimensions, upscalingOrder, Hdivcoeff);
        fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);
        fillCoefficientArray(nDimensions, upscalingOrder+1, H1coeff);

        int jform(0);

        targets[jform] =
            DRSequence_FE->InterpolateScalarTargets(jform, H1coeff);
        ++jform;

        if(nDimensions == 3)
        {
            targets[jform] =
                DRSequence_FE->InterpolateVectorTargets(jform, Hcurlcoeff);
            ++jform;
        }

        targets[jform] =
            DRSequence_FE->InterpolateVectorTargets(jform, Hdivcoeff);
        ++jform;

        targets[jform] =
            DRSequence_FE->InterpolateScalarTargets(jform, L2coeff);
        ++jform;

        freeCoeffArray(L2coeff);
        freeCoeffArray(Hdivcoeff);
        freeCoeffArray(Hcurlcoeff);
        freeCoeffArray(H1coeff);

        mfem::Array<MultiVector *> targets_in(targets.size());
        for (int ii = 0; ii < targets_in.Size(); ++ii)
            targets_in[ii] = targets[ii].get();

        if (print_progress_report)
            std::cout << "-- Setting targets...\n";

        sequence[0]->SetTargets(targets_in);
    }

    if (print_progress_report)
        std::cout << "-- Built fine grid DeRhamSequence.\n\n"
                  << "-- Coarsening the DeRhamSequence to all levels...\n";

    for(int i(0); i < nLevels-1; ++i)
    {
        std::ostringstream timer_name;
        timer_name << "DeRhamSequence Construction -- Level " << i+1;
        Timer timer = TimeManager::AddTimer(timer_name.str());
        Timer tot_timer = TimeManager::GetTimer(
            "DeRhamSequence Construction -- Total");

        constexpr double tolSVD = 1e-9;
        sequence[i]->SetSVDTol(tolSVD);
        sequence[i+1] = sequence[i]->Coarsen();
    }

    if (print_progress_report)
        std::cout << "-- Coarsened DeRhamSequence to all levels.\n\n"
                  << "-- Assembling the linear system on level "
                  << start_level << "...\n";

    // Grab the FE space
    const int theform = nForms - 2;

    mfem::FiniteElementSpace * fespace
        = sequence[0]->FemSequence()->GetFeSpace(theform);

    // This is the FG RHS. Have to create this, regardless of solve level.
    auto fform = make_unique<mfem::LinearForm>(fespace);
    fform->AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f));
    fform->Assemble();

    // Move the bilinear form into the RHS vector
    unique_ptr<mfem::Vector> rhs = std::move(fform);

    // Create the boundary data
    mfem::VectorFunctionCoefficient F(nDimensions, F_exact);
    auto true_sol_gf = make_unique<mfem::GridFunction>(fespace);
    //true_sol->ProjectBdrCoefficientTangent(E, ess_attr);
    true_sol_gf->ProjectCoefficient(F);

    // Make a deep copy of the true solution to project
    auto true_sol = make_unique<mfem::Vector>(*true_sol_gf);

    // Project our way down to the start_level
    for (int ii = 0; ii < start_level; ++ii)
    {
        auto tmp = make_unique<mfem::Vector>(
            sequence[ii+1]->GetNumberOfDofs(theform) );
        sequence[ii]->GetP(theform)->MultTranspose(*rhs,*tmp);
        rhs = std::move(tmp);

        tmp = make_unique<mfem::Vector>(
            sequence[ii+1]->GetNumberOfDofs(theform) );
        sequence[ii]->GetPi(theform)->ComputeProjector();
        sequence[ii]->GetPi(theform)->GetProjectorMatrix().Mult(*true_sol,*tmp);
        true_sol = std::move(tmp);
    }

        // Create the solver library, used for any/all level(s)
    auto lib = SolverLibrary::CreateLibrary(
        master_list->Sublist("Preconditioner Library"));

    for (int level = start_level; level <= stop_level; ++level)
    {
        if (level != start_level)
        {
            auto tmp = make_unique<mfem::Vector>(
                sequence[level]->GetNumberOfDofs(theform) );
            sequence[level-1]->GetP(theform)->MultTranspose(*rhs,*tmp);
            rhs = std::move(tmp);

            tmp = make_unique<mfem::Vector>(
                sequence[level]->GetNumberOfDofs(theform) );
            sequence[level-1]->GetPi(theform)->ComputeProjector();
            sequence[level-1]->GetPi(theform)->GetProjectorMatrix().Mult(*true_sol,*tmp);
            true_sol = std::move(tmp);
        }

        if (print_progress_report)
            std::cout << std::endl
                      << "-- Starting solve phase on level " << level
                      << "...\n";

        // Create the parallel linear system
        const SharingMap& hdiv_dofTrueDof
            = sequence[level]->GetDofHandler(theform)->GetDofTrueDof();

        // First the RHS
        mfem::Vector B{hdiv_dofTrueDof.GetTrueLocalSize()};
        mfem::Vector rhs_tmp(*rhs);

        // Now system matrix, A
        shared_ptr<mfem::HypreParMatrix> A;
        {
            if (print_progress_report)
                std::cout << "-- Building operator on level " << level
                          << "...\n";

            // Get the mass operator
            auto M1 = sequence[level]->ComputeMassOperator(theform),
                M2 = sequence[level]->ComputeMassOperator(theform+1);
            auto D1 = sequence[level]->GetDerivativeOperator(theform);

            // spA = (div u, div v) + (u, v) = D1^T*M2*D1 + M1
            auto spA = ToUnique(mfem::Add(1.0,*M1,1.0,*ToUnique(mfem::RAP(*D1,*M2,*D1))));

            // Eliminate the boundary conditions
            mfem::Array<int> marker(spA->Height());
            marker = 0;
            sequence[level]->GetDofHandler(theform)->MarkDofsOnSelectedBndr(
                ess_attr[0], marker);

            for(int mm = 0; mm < spA->Height(); ++mm)
                if(marker[mm])
                    spA->EliminateRowCol(mm, true_sol->operator()(mm), rhs_tmp);

            A = Assemble(hdiv_dofTrueDof,*spA,hdiv_dofTrueDof);
            hdiv_dofTrueDof.Assemble(rhs_tmp,B);

            if (print_progress_report)
                std::cout << "-- Built operator on level " << level
                          << ".\n"
                          <<"-- Assembled the linear system on level "
                          << level << ".\n\n";
        }

        size_t global_height,global_width,global_nnz;
        {
            hypre_ParCSRMatrix* tmp = *A;

            size_t local_height = A->Height(), local_width = A->Width(),
                local_nnz = tmp->diag->num_nonzeros + tmp->offd->num_nonzeros;
            MPI_Reduce(&local_height,&global_height,1,GetMPIType(local_height),
                       MPI_SUM,0,comm);
            MPI_Reduce(&local_width,&global_width,1,GetMPIType(local_width),
                       MPI_SUM,0,comm);
            MPI_Reduce(&local_nnz,&global_nnz,1,GetMPIType<size_t>(local_nnz),
                       MPI_SUM,0,comm);
        }
        PARELAG_ASSERT(B.Size() == A->Height());

        //
        // Loop through the solvers!
        //
        auto list_of_solvers =
            prob_list.Get<std::list<std::string>>("List of linear solvers");

        for (const auto& solver_type : list_of_solvers)
        {
            auto suffix = std::string(" (\"").append(solver_type)
                          .append("\" -- Level ").append(std::to_string(level))
                          .append(")");
            // Get the factory
            auto solver_list = master_list->Sublist("Preconditioner Library")
                               .Sublist(solver_type);

            solver_list.Sublist("Solver Parameters")
            .Set("Timer name suffix",std::move(suffix));

            lib->AddSolver(solver_type,std::move(solver_list));

            // Get the factory
            auto prec_factory = lib->GetSolverFactory(solver_type);
            auto solver_state = prec_factory->GetDefaultState();
            solver_state->SetDeRhamSequence(sequence[level]);
            solver_state->SetBoundaryLabels(ess_attr);
            solver_state->SetForms({theform});

            unique_ptr<mfem::Solver> solver;

            // Build the preconditioner
            if (print_progress_report)
                std::cout << "-- Building solver \"" << solver_type
                          << "\"...\n";

            {
                Timer timer = TimeManager::AddTimer(
                    std::string("Build Solver \"").append(solver_type)
                    .append("\" -- Level ").append(std::to_string(level)));

                solver = prec_factory->BuildSolver(A,*solver_state);
            }

            if (print_progress_report)
                std::cout << "-- Built solver \"" << solver_type << "\".\n";

            if (!myid)
                std::cout << '\n' << std::string(50,'*') << '\n'
                          << "*    Solving on level: " << level << '\n'
                          << "*              A size: "
                          << global_height << 'x' << global_width << '\n'
                          << "*               A NNZ: " << global_nnz << "\n*\n"
                          << "*              Solver: " << solver_type << '\n'
                  << std::string(50,'*') << '\n' << std::endl;

            if (print_progress_report)
                std::cout << "-- Solving system with " << solver_type << "...\n";

            mfem::Vector X(A->Width()),
                x(sequence[level]->GetNumberOfDofs(theform));
            X=0.0;

            {
                double local_norm = B.Norml2() * B.Norml2();
                double global_norm;
                MPI_Reduce(&local_norm,&global_norm,1,GetMPIType(local_norm),
                           MPI_SUM,0,comm);

                if (!myid)
                    std::cout << "Initial residual norm: "
                              << std::sqrt(global_norm) << std::endl;
            }

            {
                Timer timer = TimeManager::AddTimer(
                                  std::string("Solve Linear System \"")
                                  .append(solver_type)
                                  .append("\" -- Level ")
                                  .append(std::to_string(level)));
                solver->Mult(B,X);
            }

            {
                mfem::Vector tmp(A->Height());
                A->Mult(X,tmp);
                tmp *= -1.0;
                tmp += B;

                double local_norm = tmp.Norml2() * tmp.Norml2();
                double global_norm;
                MPI_Reduce(&local_norm,&global_norm,1,GetMPIType(local_norm),
                           MPI_SUM,0,comm);

                if (!myid)
                    std::cout << "Final residual norm: " << std::sqrt(global_norm)
                              << std::endl;
            }

            if (print_progress_report)
                std::cout << "-- Solver has exited.\n";

            if (visualize)
            {
                hdiv_dofTrueDof.Distribute(X,x);

                MultiVector tmp(x.GetData(),1, x.Size());
                sequence[level]->show(theform,tmp);
            }
        }
    }

    if (print_time)
        TimeManager::Print(std::cout);

    if (print_progress_report)
        std::cout << "-- Goodbye!\n\n";

    return EXIT_SUCCESS;
}

// The exact solution (for non-surface meshes)
void F_exact(const mfem::Vector &p, mfem::Vector &F)
{
   int dim = p.Size();

   F(0) = 0.0;
   F(1) = 0.0;
   if (dim == 3)
   {
      F(2) = 0.0;
   }
}

// The right hand side
void f_exact(const mfem::Vector &p, mfem::Vector &f)
{
   int dim = p.Size();

   f(0) = 0.0;
   f(1) = 0.0;
   if (dim == 3)
   {
      f(2) = 1.0;
   }
}

#if 0
void F_exact(const mfem::Vector &p, mfem::Vector &F)
{
   int dim = p.Size();

   double x = p(0);
   double y = p(1);
   // double z = (dim == 3) ? p(2) : 0.0;

   F(0) = cos(kappa*x)*sin(kappa*y);
   F(1) = cos(kappa*y)*sin(kappa*x);
   if (dim == 3)
   {
      F(2) = 0.0;
   }
}

// The right hand side
void f_exact(const mfem::Vector &p, mfem::Vector &f)
{
   int dim = p.Size();

   double x = p(0);
   double y = p(1);
   // double z = (dim == 3) ? p(2) : 0.0;

   double temp = 1 + 2*kappa*kappa;

   f(0) = temp*cos(kappa*x)*sin(kappa*y);
   f(1) = temp*cos(kappa*y)*sin(kappa*x);
   if (dim == 3)
   {
      f(2) = 0;
   }
}
#endif
