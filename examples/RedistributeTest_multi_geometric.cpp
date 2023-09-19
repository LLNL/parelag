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
#include <unistd.h> 

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
using std::make_shared;
using std::shared_ptr;
using std::unique_ptr;

int main(int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    auto total_timer = TimeManager::AddTimer("ZTotal time");

    double sigma = 1;
    double kappa = 0.1 * M_PI;

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

    ParameterList &prob_list = master_list->Sublist("Problem parameters", true);

    // The file from which to read the mesh
    const std::string meshfile =
        prob_list.Get("Mesh file", "no mesh name found");

    // The number of times to refine in parallel
    const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2);

    // The level of the resulting hierarchy at which you would like
    // the solve to be performed. 0 is the finest grid.
    const int start_level = prob_list.Get("Solve level", 0);

    ParameterList &output_list = master_list->Sublist("Output control");
    const bool print_time = output_list.Get("Print timings", true);
    const bool show_progress = output_list.Get("Show progress", true);
    const bool visualize = output_list.Get("Visualize solution", false);
    const bool unstructured = prob_list.Get("Unstructured coarsening", false);

    const bool print_progress_report = (!myid && show_progress);

    if (print_progress_report)
        std::cout << "\n-- Hello!\n"
                  << "-- Welcome to RedistributeTestDarcy with multiple coarse copies!\n\n";

    std::ostringstream mesh_msg;
    if (!myid)
    {
        mesh_msg << '\n'
                 << std::string(50, '*') << '\n'
                 << "*  Mesh: " << meshfile << "\n*\n";
    }

    int ser_ref_levels;
    Array<int> partitioning_permuation;
    // Array<int> local2global, global2local;
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
                    std::cerr << std::endl
                              << "Cannot open mesh file: "
                              << meshfile << std::endl
                              << std::endl;
                return EXIT_FAILURE;
            }

            mesh = make_unique<mfem::Mesh>(imesh, 1, 1);
            imesh.close();

            if (print_progress_report)
                std::cout << "-- Read mesh \"" << meshfile
                          << "\" successfully.\n";
        }

        ser_ref_levels =
            prob_list.Get("Serial refinement levels", -1);

        // This will do no refs if ser_ref_levels <= 0.
        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        // Negative means refine until mesh is big enough to distribute!
        if (ser_ref_levels < 0)
        {
            ser_ref_levels = 0;
            for (; mesh->GetNE() < 6 * num_ranks; ++ser_ref_levels)
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
            std::cout << "-- Building parallel mesh...\n"
                      << std::flush;

        bool geometric_coarsening = prob_list.Get("Use geometric coarsening", false);
        if (geometric_coarsening)
            partitioning_permuation.SetSize(mesh->GetNE());
        // {
        //     auto timer = TimeManager::AddTimer("Mesh : build permutation map");
        //     Array<Array<int>*> tmp1(num_ranks);
        //     for (auto && arr : tmp1)
        //     {
        //         arr = new Array<int>(0);
        //         arr->Reserve(par_partitioning.Size() / num_ranks * 15 / 10);
        //     }

        //     for (int i = 0; i < par_partitioning.Size(); i++)
        //         tmp1[par_partitioning[i]]->Append(i);

        //     Array<int> tmp2;
        //     tmp2.Reserve(par_partitioning.Size());
        //     for (auto && arr : tmp1)
        //         tmp2.Append(*arr);

        //     partitioning_permuation.SetSize(tmp2.Size());
        //     for (int i = 0; i < partitioning_permuation.Size(); i++)
        //         partitioning_permuation[tmp2[i]] = i;
        //     tmp1.DeleteAll();
        // }


        // FIXME (aschaf 08/22/23) : when using METIS to generate the parallel distribution there are problems with processor boundaries when redistributing back to a single processor
        bool metis_partitioning = prob_list.Get("Use METIS for parallel partitioning", false);
        pmesh = BuildParallelMesh(comm, *mesh, partitioning_permuation, metis_partitioning);

        if (pmesh && print_progress_report)
            std::cout << "-- Built parallel mesh successfully.\n"
                      << std::flush;
    }

    // Mark no boundary attributes as essential
    std::vector<mfem::Array<int>> ess_attr(1);
    ess_attr[0].SetSize(pmesh->bdr_attributes.Max());
    ess_attr[0] = 0;

    const int nDimensions = pmesh->Dimension();

    const int nGeometricLevels = par_ref_levels + 1;
    std::vector<int> num_elements(nGeometricLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        num_elements[par_ref_levels - l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    if (unstructured)
        num_elements.resize(1);
    num_elements[0] = pmesh->GetNE();

    if (print_progress_report)
        std::cout << "-- Refined mesh in parallel " << par_ref_levels
                  << " times.\n\n";

    if (nDimensions == 3)
        pmesh->ReorientTetMesh();

    {
        size_t my_num_elmts = pmesh->GetNE(), global_num_elmts;
        MPI_Reduce(&my_num_elmts, &global_num_elmts, 1, GetMPIType<size_t>(0),
                   MPI_SUM, 0, comm);

        if (!myid)
        {
            mesh_msg << "*  Parallel refinements: " << par_ref_levels << '\n'
                     << "*        Fine Mesh Size: " << global_num_elmts << '\n'
                     << std::string(50, '*') << '\n'
                     << std::endl;
        }
    }

    if (!myid)
        std::cout << mesh_msg.str();

    const auto solve_form = prob_list.Get("Solve form", nDimensions - 1); // TODO: read from ParameterList

    const int uform = solve_form;
    const int pform = solve_form + 1;

    ConstantCoefficient coeffs[4] = {
        ConstantCoefficient(1.), // coeffL2
        ConstantCoefficient(1.), // coeffHdiv
        ConstantCoefficient(1.), // coeffH1
        ConstantCoefficient(1.)  // coeffHcurl
    };

    SequenceHierarchy hierarchy(pmesh, prob_list, print_progress_report);
    const auto start_form = prob_list.Get("Start form", nDimensions - 1); // TODO: read from ParameterList

    for (int jform = start_form; jform <= nDimensions; jform++)
        hierarchy.SetCoefficient(jform, coeffs[jform], (jform == nDimensions));

    hierarchy.SetSerialRefinementInfo(ser_ref_levels, partitioning_permuation);
    hierarchy.Build(move(num_elements));
    auto &sequence = hierarchy.GetDeRhamSequences();
    mfem::Vector ones({1.,1.,1.});
    VectorConstantCoefficient one_coeff(ones);

    if (solve_form == nDimensions - 1) // Darcy test
    {
        auto uFun_ex = [](const Vector &x, Vector &u)
        {
            double xi(x(0));
            double yi(x(1));
            double zi(0.0);
            if (x.Size() == 3)
            {
                zi = x(2);
            }

            u(0) = -exp(xi) * sin(yi) * cos(zi);
            u(1) = -exp(xi) * cos(yi) * cos(zi);

            if (x.Size() == 3)
            {
                u(2) = exp(xi) * sin(yi) * sin(zi);
            }
        };

        // Change if needed
        auto pFun_ex = [](const Vector &x)
        {
            double xi(x(0));
            double yi(x(1));
            double zi(0.0);

            if (x.Size() == 3)
            {
                zi = x(2);
            }

            return exp(xi) * sin(yi) * cos(zi);
        };

        auto fFun = [](const Vector &x, Vector &f)
        {
            f = 0.0;
        };

        auto gFun = [&pFun_ex](const Vector &x)
        {
            if (x.Size() == 3)
            {
                return -pFun_ex(x);
            }
            else
            {
                return 0.;
            }
        };

        auto f_natural = [&pFun_ex](const Vector &x)
        {
            return (-pFun_ex(x));
        };

        PARELAG_ASSERT(start_level < sequence.size());

        auto DRSequence_FE = sequence[0]->FemSequence();
        FiniteElementSpace *ufespace = DRSequence_FE->GetFeSpace(uform);
        FiniteElementSpace *pfespace = DRSequence_FE->GetFeSpace(pform);

        mfem::LinearForm bform(ufespace);
        FunctionCoefficient fbdr(f_natural);
        bform.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fbdr));
        bform.Assemble();

        mfem::LinearForm qform(pfespace);
        FunctionCoefficient source(gFun);
        qform.AddDomainIntegrator(new DomainLFIntegrator(source));
        qform.Assemble();

        FunctionCoefficient truep_coeff(pFun_ex);
        VectorFunctionCoefficient trueu_coeff(nDimensions, uFun_ex);
        auto truep_gf = make_unique<GridFunction>(pfespace);
        truep_gf->ProjectCoefficient(truep_coeff);
        auto trueu_gf = make_unique<GridFunction>(ufespace);
        trueu_gf->ProjectCoefficient(trueu_coeff);
        auto hdiv_const_gf = make_unique<GridFunction>(ufespace);
        hdiv_const_gf->ProjectCoefficient(one_coeff);

        // Project rhs down to the level of interest
        auto rhs_u = make_unique<mfem::Vector>(sequence[0]->GetNumTrueDofs(uform));
        auto rhs_p = make_unique<mfem::Vector>(sequence[0]->GetNumTrueDofs(pform));
        *rhs_u = 0.0;
        *rhs_p = 0.0;
        sequence[0]->GetDofHandler(uform)->GetDofTrueDof().Assemble(bform, *rhs_u);
        sequence[0]->GetDofHandler(pform)->GetDofTrueDof().Assemble(qform, *rhs_p);
        auto truep = make_unique<mfem::Vector>(sequence[0]->GetNumTrueDofs(pform));
        auto trueu = make_unique<mfem::Vector>(sequence[0]->GetNumTrueDofs(uform));
        *trueu = 0.0;
        *truep = 0.0;
        sequence[0]->GetDofHandler(pform)->GetDofTrueDof().IgnoreNonLocal(*truep_gf, *truep);
        sequence[0]->GetDofHandler(uform)->GetDofTrueDof().IgnoreNonLocal(*trueu_gf, *trueu);
        auto hdiv_const = make_unique<mfem::Vector>(sequence[0]->GetNumTrueDofs(uform));
        sequence[0]->GetDofHandler(uform)->GetDofTrueDof().IgnoreNonLocal(*hdiv_const_gf, *hdiv_const);


        for (int ii = 0; ii < start_level; ++ii)
        {
            auto tmp_u = make_unique<mfem::Vector>(
                hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(ii + 1))[ii + 1]->GetNumTrueDofs(uform));
            auto tmp_p = make_unique<mfem::Vector>(
                hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(ii + 1))[ii + 1]->GetNumTrueDofs(pform));
            // sequence[ii]->GetTrueP(uform).MultTranspose(*rhs_u,*tmp_u);
            // sequence[ii]->ApplyTruePTranspose(uform,*rhs_u,*tmp_u);
            hierarchy.ApplyTruePTranspose(ii, uform, *rhs_u, *tmp_u);
            // sequence[ii]->GetTrueP(pform).MultTranspose(*rhs_p,*tmp_p);
            // sequence[ii]->ApplyTruePTranspose(pform,*rhs_p,*tmp_p);
            hierarchy.ApplyTruePTranspose(ii, pform, *rhs_p, *tmp_p);
            rhs_u = std::move(tmp_u);
            rhs_p = std::move(tmp_p);
            tmp_u = make_unique<mfem::Vector>(
                hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(ii + 1))[ii + 1]->GetNumTrueDofs(uform));
            tmp_p = make_unique<mfem::Vector>(
                hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(ii + 1))[ii + 1]->GetNumTrueDofs(pform));
            hierarchy.ApplyTruePi(ii, uform, *trueu, *tmp_u);
            hierarchy.ApplyTruePi(ii, pform, *truep, *tmp_p);
            truep = std::move(tmp_p);
            trueu = std::move(tmp_u);
            tmp_u = make_unique<mfem::Vector>(
                hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(ii + 1))[ii + 1]->GetNumTrueDofs(uform));
            hierarchy.ApplyTruePi(ii, uform, *hdiv_const, *tmp_u);
            hdiv_const = std::move(tmp_u);
        }

        const SharingMap &hdiv_dofTrueDof = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetDofHandler(uform)->GetDofTrueDof();
        const SharingMap &l2_dofTrueDof = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetDofHandler(pform)->GetDofTrueDof();

        // Create the parallel linear system
        mfem::Array<int> true_block_offsets(3);
        true_block_offsets[0] = 0;
        true_block_offsets[1] = hdiv_dofTrueDof.GetTrueLocalSize();
        true_block_offsets[2] =
            true_block_offsets[1] + l2_dofTrueDof.GetTrueLocalSize();

        auto A = make_shared<MfemBlockOperator>(true_block_offsets);
        size_t local_nnz = 0;
        mfem::BlockVector prhs(true_block_offsets);
        mfem::BlockVector ptruesol(true_block_offsets);
        {
            if (print_progress_report)
                std::cout << "-- Building operator on level " << start_level
                          << "...\n";

            // The blocks, managed here
            auto M = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->ComputeMassOperator(uform);
            auto W = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->ComputeMassOperator(pform);
            auto D = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetDerivativeOperator(uform);

            auto B = ToUnique(Mult(*W, *D));
            *B *= -1.;
            auto Bt = ToUnique(Transpose(*B));

            auto pM = Assemble(hdiv_dofTrueDof, *M, hdiv_dofTrueDof);
            auto pB = Assemble(l2_dofTrueDof, *B, hdiv_dofTrueDof);
            auto pBt = Assemble(hdiv_dofTrueDof, *Bt, l2_dofTrueDof);

            hypre_ParCSRMatrix *tmp = *pM;
            local_nnz += tmp->diag->num_nonzeros;
            local_nnz += tmp->offd->num_nonzeros;
            tmp = *pB;
            local_nnz += tmp->diag->num_nonzeros;
            local_nnz += tmp->offd->num_nonzeros;
            tmp = *pBt;
            local_nnz += tmp->diag->num_nonzeros;
            local_nnz += tmp->offd->num_nonzeros;

            A->SetBlock(0, 0, std::move(pM));
            A->SetBlock(0, 1, std::move(pBt));
            A->SetBlock(1, 0, std::move(pB));

            // Setup right-hand side
            prhs.GetBlock(0) = *rhs_u;
            prhs.GetBlock(1) = *rhs_p;

            ptruesol.GetBlock(0) = *trueu;
            ptruesol.GetBlock(1) = *truep;

            if (print_progress_report)
                std::cout << "-- Built operator on level " << start_level
                          << ".\n"
                          << "-- Assembled the linear system on level "
                          << start_level << ".\n\n";
        }

        // Report some stats on global problem size
        size_t global_height, global_width, global_nnz;
        {
            size_t local_height = A->Height(), local_width = A->Width();
            MPI_Reduce(&local_height, &global_height, 1, GetMPIType(local_height),
                       MPI_SUM, 0, comm);
            MPI_Reduce(&local_width, &global_width, 1, GetMPIType(local_width),
                       MPI_SUM, 0, comm);
            MPI_Reduce(&local_nnz, &global_nnz, 1, GetMPIType<size_t>(local_nnz),
                       MPI_SUM, 0, comm);
        }
        PARELAG_ASSERT(prhs.Size() == A->Height());

        //
        // Create the preconditioner
        //

        // Start with the solver library
        ParameterList &prec_list = master_list->Sublist("Preconditioner Library");
        auto lib = SolverLibrary::CreateLibrary(prec_list);

        // Get the factory
        const std::string solver_type = prob_list.Get("Linear solver", "Unknown");
        auto prec_factory = lib->GetSolverFactory(solver_type);
        const int rescale_iter = prec_list.Sublist(solver_type).Sublist("Solver Parameters").Get("RescaleIteration", -20);

        auto solver_state = prec_factory->GetDefaultState();
        solver_state->SetDeRhamSequence(hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]);
        solver_state->SetBoundaryLabels(
            std::vector<std::vector<int>>(2, std::vector<int>()));
        solver_state->SetForms({uform, pform});

        // These are for hybridization solver
        solver_state->SetExtraParameter("IsSameOrient", (start_level > 0));
        solver_state->SetExtraParameter("ActOnTrueDofs", true);
        solver_state->SetExtraParameter("RescaleIteration", rescale_iter);

        unique_ptr<mfem::Solver> solver;

        // Build the preconditioner
        if (print_progress_report)
            std::cout << "-- Building solver \"" << solver_type << "\"...\n";

        {
            Timer timer = TimeManager::AddTimer("Build Solver");
            solver = prec_factory->BuildSolver(A, *solver_state);
            solver->iterative_mode = false;
        }

        if (print_progress_report)
            std::cout << "-- Built solver \"" << solver_type << "\".\n";

        mfem::BlockVector psol(true_block_offsets);
        psol = 0.;

        if (!myid)
            std::cout << '\n'
                      << std::string(50, '*') << '\n'
                      << "*    Solving on level: " << start_level << '\n'
                      << "*              A size: "
                      << global_height << 'x' << global_width << '\n'
                      << "*               A NNZ: " << global_nnz << "\n*\n"
                      << "*              Solver: " << solver_type << "\n"
                      << std::string(50, '*') << '\n'
                      << std::endl;

        if (print_progress_report)
            std::cout << "-- Solving system with " << solver_type << "...\n";
        {
            double local_norm = prhs.Norml2() * prhs.Norml2();
            double global_norm;
            MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
                       MPI_SUM, 0, comm);

            if (!myid)
                std::cout << "Initial residual norm: " << std::sqrt(global_norm)
                          << std::endl;
        }

        {
            auto timer = TimeManager::AddTimer("Solve");
            solver->Mult(prhs, psol);
        }

        {
            mfem::Vector tmp(A->Height());
            A->Mult(psol, tmp);
            prhs -= tmp;
            double local_norm = prhs.Norml2() * prhs.Norml2();
            double global_norm;
            MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
                       MPI_SUM, 0, comm);

            if (!myid)
            {
                std::cout << "Final residual norm: " << std::sqrt(global_norm)
                          << std::endl;
                auto hybrid_solver = dynamic_cast<HybridizationSolver *>(solver.get());
                if (hybrid_solver)
                {
                    std::cout << "Number of iterations: " << hybrid_solver->GetNumIters()
                              << std::endl;
                }
            }
        }
        {
            mfem::Vector err(psol);
            err -= ptruesol;
            double local_norm = err.Normlinf();
            double global_norm;
            MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
            MPI_MAX, 0, comm);

            if (!myid)
            {
                std::cout << "||u* - uh||_infty : " << (global_norm) << std::endl;
            }
        }


        if (visualize)
        {
            psol = 1.;
            psol.GetBlock(0) = *hdiv_const;
            int k_l = hierarchy.GetRedistributionIndex(start_level);
            int level_numgroups = hierarchy.GetNumGlobalCopies(k_l);
            Vector tmp(hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetNumTrueDofs(nDimensions));
            MultiVector u(psol.GetData(), 1, psol.BlockSize(0));
            MultiVector p(psol.GetBlock(1).GetData(), 1, psol.BlockSize(1));
            MultiVector elems(tmp.GetData(), 1, tmp.Size());

            int loc = psol.BlockSize(1);
            int o = 0;
            MPI_Exscan(&loc, &o, 1, MPI_INT, MPI_SUM, hierarchy.GetComm(hierarchy.GetRedistributionIndex(start_level)));
            for (auto &&pp : elems)
                pp = o++;
            if (!prob_list.Get("Visualize multiple copies", false))
                level_numgroups = 1;
            for (int groupid(0); groupid < level_numgroups; groupid++)
            {
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, uform, u);
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, pform, p);
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, nDimensions, elems);
            }
        }

        if (print_progress_report)
            std::cout << "-- Solver has exited.\n";
    }
    else if (solve_form == 1) // H(curl) test
    {
        const auto theform = solve_form;
        auto DRSequence_FE = sequence[0]->FemSequence();
        FiniteElementSpace *fespace = DRSequence_FE->GetFeSpace(theform); // H(curl)

        auto truesol_gf = make_unique<GridFunction>(fespace);
        auto onevec_gf = make_unique<GridFunction>(fespace);

        mfem::LinearForm bform(fespace);
        auto source_vec = [&nDimensions, &sigma, &kappa](const Vector &x, Vector &f)
        {
            if (nDimensions == 3)
            {
                f(0) = (sigma + kappa * kappa) * sin(kappa * x(1));
                f(1) = (sigma + kappa * kappa) * sin(kappa * x(2));
                f(2) = (sigma + kappa * kappa) * sin(kappa * x(0));
            }
            else
            {
                f(0) = (sigma + kappa * kappa) * sin(kappa * x(1));
                f(1) = (sigma + kappa * kappa) * sin(kappa * x(0));
                if (x.Size() == 3)
                {
                    f(2) = 0.0;
                }
            }
        };

        auto exact_sol = [&nDimensions, &sigma, &kappa](const Vector &x, Vector &E)
        {
            if (nDimensions == 3)
            {
                E(0) = sin(kappa * x(1));
                E(1) = sin(kappa * x(2));
                E(2) = sin(kappa * x(0));
            }
            else
            {
                E(0) = sin(kappa * x(1));
                E(1) = sin(kappa * x(0));
                if (x.Size() == 3)
                {
                    E(2) = 0.0;
                }
            }
        };
        auto exact_Dsol = [&nDimensions, &sigma, &kappa](const Vector &x, Vector &E)
        {
            if (nDimensions == 3)
            {
                E(0) = -kappa * cos(kappa * x(2));
                E(1) = -kappa * cos(kappa * x(0));
                E(2) = -kappa * cos(kappa * x(1));
            }
            else
            {
                E(0) = sin(kappa * x(1));
                E(1) = sin(kappa * x(0));
                if (x.Size() == 3)
                {
                    E(2) = 0.0;
                }
            }
        };
        VectorFunctionCoefficient exact(nDimensions, exact_sol);
        VectorFunctionCoefficient Dexact(nDimensions, exact_Dsol);
        ScalarVectorProductCoefficient mDexact(-1., Dexact);
        truesol_gf->ProjectCoefficient(exact);
        onevec_gf->ProjectCoefficient(one_coeff);

        VectorFunctionCoefficient source(nDimensions, source_vec);
        bform.AddDomainIntegrator(new VectorFEDomainLFIntegrator(source));
        bform.AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(mDexact));

        bform.Assemble();

        // Project rhs down to the level of interest
        auto rhs_u = make_unique<mfem::Vector>(sequence[0]->GetNumTrueDofs(theform));
        *rhs_u = 0.0;
        sequence[0]->GetDofHandler(theform)->GetDofTrueDof().Assemble(bform, *rhs_u);
        auto truesol = make_unique<mfem::Vector>(sequence[0]->GetNumTrueDofs(theform));
        sequence[0]->GetDofHandler(theform)->GetDofTrueDof().IgnoreNonLocal(*truesol_gf, *truesol);
        auto true_onevec = make_unique<mfem::Vector>(sequence[0]->GetNumTrueDofs(theform));
        sequence[0]->GetDofHandler(theform)->GetDofTrueDof().IgnoreNonLocal(*onevec_gf, *true_onevec);

        for (int ii = 0; ii < start_level; ++ii)
        {
            auto tmp = make_unique<mfem::Vector>(
                hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(ii + 1))[ii + 1]->GetNumTrueDofs(theform));
            hierarchy.ApplyTruePTranspose(ii, theform, *rhs_u, *tmp);
            rhs_u = std::move(tmp);
            tmp = make_unique<mfem::Vector>(
                hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(ii + 1))[ii + 1]->GetNumTrueDofs(theform));
            hierarchy.ApplyTruePi(ii, theform, *truesol, *tmp);
            truesol = std::move(tmp);
            tmp = make_unique<mfem::Vector>(
                hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(ii + 1))[ii + 1]->GetNumTrueDofs(theform));
            hierarchy.ApplyTruePi(ii, theform, *true_onevec, *tmp);
            true_onevec = std::move(tmp);
        }

        const SharingMap &hcurl_dofTrueDof = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetDofHandler(theform)->GetDofTrueDof();

        // Create the parallel linear system
        shared_ptr<mfem::HypreParMatrix> pA;
        size_t local_nnz = 0;
        mfem::Vector prhs;
        {
            auto timer = TimeManager::AddTimer(string("Building operator on level ").append(to_string(start_level)));
            Vector truesoltmp(hcurl_dofTrueDof.GetLocalSize()), rhstmp(hcurl_dofTrueDof.GetLocalSize());
            hcurl_dofTrueDof.DisAssemble(*rhs_u, rhstmp);
            hcurl_dofTrueDof.DisAssemble(*truesol, truesoltmp);
            if (print_progress_report)
                std::cout << "-- Building operator on level " << start_level
                          << "...\n";

            // The blocks, managed here
            auto M = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->ComputeMassOperator(theform);
            auto W = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->ComputeMassOperator(theform + 1);
            auto D = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetDerivativeOperator(theform);

            auto A = ToUnique(Add(sigma, *M, 1.0, *RAP(*D, *W, *D)));

            mfem::Array<int> marker(A->Height());
            marker = 0;
            hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetDofHandler(theform)->MarkDofsOnSelectedBndr(
                ess_attr[0], marker);

            for (int mm = 0; mm < A->Height(); ++mm)
                if (marker[mm])
                    A->EliminateRowCol(mm, truesoltmp(mm), rhstmp);

            pA = Assemble(hcurl_dofTrueDof, *A, hcurl_dofTrueDof);

            hypre_ParCSRMatrix *tmp = *pA;
            local_nnz += tmp->diag->num_nonzeros;
            local_nnz += tmp->offd->num_nonzeros;

            // Setup right-hand side
            hcurl_dofTrueDof.IgnoreNonLocal(rhstmp, *rhs_u);
            prhs.MakeRef(*rhs_u, 0, rhs_u->Size());

            if (print_progress_report)
                std::cout << "-- Built operator on level " << start_level
                          << ".\n"
                          << "-- Assembled the linear system on level "
                          << start_level << ".\n\n";
        }

        // Report some stats on global problem size
        size_t global_height, global_width, global_nnz;
        {
            size_t local_height = pA->Height(), local_width = pA->Width();
            MPI_Reduce(&local_height, &global_height, 1, GetMPIType(local_height),
                       MPI_SUM, 0, comm);
            MPI_Reduce(&local_width, &global_width, 1, GetMPIType(local_width),
                       MPI_SUM, 0, comm);
            MPI_Reduce(&local_nnz, &global_nnz, 1, GetMPIType<size_t>(local_nnz),
                       MPI_SUM, 0, comm);
        }
        PARELAG_ASSERT(prhs.Size() == pA->Height());

        //
        // Create the preconditioner
        //

        // Start with the solver library
        ParameterList &prec_list = master_list->Sublist("Preconditioner Library");
        auto lib = SolverLibrary::CreateLibrary(prec_list);

        // Get the factory
        const std::string solver_type =  prob_list.Get("Linear solver 1-Form", "CG-AMG");// "PCG with Auxiliary Space Preconditioner"; // "CG-AMG";
        auto prec_factory = lib->GetSolverFactory(solver_type);
        // const int rescale_iter = prec_list.Sublist(solver_type).Sublist(
        //         "Solver Parameters").Get<int>("RescaleIteration");

        auto solver_state = prec_factory->GetDefaultState();
        solver_state->SetDeRhamSequence(hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]);
        solver_state->SetBoundaryLabels(ess_attr);
        solver_state->SetForms({theform});

        unique_ptr<mfem::Solver> solver;

        // Build the preconditioner
        if (print_progress_report)
            std::cout << "-- Building solver \"" << solver_type << "\"...\n";

        {
            Timer timer = TimeManager::AddTimer("Build Solver");
            solver = prec_factory->BuildSolver(pA, *solver_state);
            solver->iterative_mode = false;
        }

        if (print_progress_report)
            std::cout << "-- Built solver \"" << solver_type << "\".\n";

        mfem::Vector psol(prhs.Size());
        psol = 0.;

        if (!myid)
            std::cout << '\n'
                      << std::string(50, '*') << '\n'
                      << "*    Solving on level: " << start_level << '\n'
                      << "*              A size: "
                      << global_height << 'x' << global_width << '\n'
                      << "*               A NNZ: " << global_nnz << "\n*\n"
                      << "*              Solver: " << solver_type << "\n"
                      << std::string(50, '*') << '\n'
                      << std::endl;

        if (print_progress_report)
            std::cout << "-- Solving system with " << solver_type << "...\n";
        {
            double local_norm = prhs.Norml2() * prhs.Norml2();
            double global_norm;
            MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
                       MPI_SUM, 0, comm);

            if (!myid)
                std::cout << "Initial residual norm: " << std::sqrt(global_norm)
                          << std::endl;
        }

        {
            auto timer = TimeManager::AddTimer("Solve");
            solver->Mult(prhs, psol);
        }

        {
            mfem::Vector tmp(pA->Height());
            pA->Mult(psol, tmp);
            prhs -= tmp;
            double local_norm = prhs.Norml2() * prhs.Norml2();
            double global_norm;
            MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
                       MPI_SUM, 0, comm);

            if (!myid)
            {
                std::cout << "Final residual norm: " << std::sqrt(global_norm)
                          << std::endl;
                auto hybrid_solver = dynamic_cast<HybridizationSolver *>(solver.get());
                if (hybrid_solver)
                {
                    std::cout << "Number of iterations: " << hybrid_solver->GetNumIters()
                              << std::endl;
                }
            }
        }
        {
            auto timer = TimeManager::AddTimer("Postprocessing : compute errors");
            mfem::Vector err(psol);
            err -= *truesol;
            // double local_norm = err.Normlinf();
            double global_norm;
            // MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
            // MPI_MAX, 0, comm);
            auto M = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->ComputeMassOperator(theform);
            auto pM = Assemble(hcurl_dofTrueDof, *M, hcurl_dofTrueDof);

            mfem::Vector Merr(err.Size());
            pM->Mult(err, Merr);
            global_norm = sqrt(mfem::InnerProduct(hierarchy.GetComm(hierarchy.GetRedistributionIndex(start_level)), Merr, err));
            if (!myid)
            {
                std::cout << "||u* - uh||_L2 : " << (global_norm) << std::endl;
            }
        }

        if (visualize)
        {
            int k_l = hierarchy.GetRedistributionIndex(start_level);
            int level_numgroups = hierarchy.GetNumGlobalCopies(k_l);
            Vector elno_vec(hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetNumTrueDofs(nDimensions));
            MultiVector u(psol.GetData(), 1, psol.Size());
            MultiVector elno(elno_vec.GetData(), 1, elno_vec.Size());
            MultiVector onevec(true_onevec->GetData(), 1, true_onevec->Size());

            // {
            //     auto fname = mfem::MakeParFilename("sol.gf", myid, string("of").append(to_string(num_ranks)), to_string(num_ranks).length());
            //     ofstream fout(fname);
            //     truesol->Print_HYPRE(fout);
            //     fout.close();
            // }

            int loc = elno_vec.Size();
            int o = 0;
            MPI_Exscan(&loc, &o, 1, MPI_INT, MPI_SUM, hierarchy.GetComm(hierarchy.GetRedistributionIndex(start_level)));
            for (auto &&pp : elno_vec)
                pp = o++;
            if (!prob_list.Get("Visualize multiple copies", false))
                level_numgroups = 1;
            for (int groupid(0); groupid < level_numgroups; groupid++)
            {
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, theform, u);
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, theform, onevec);
                // hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, nDimensions, elno);
            }

            // int glob_dofs = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetNumTrueDofs(theform);
            // MPI_Allreduce(MPI_IN_PLACE, &glob_dofs, 1, MPI_INT, MPI_SUM, hierarchy.GetComm(hierarchy.GetRedistributionIndex(start_level)));

            // for (int idof = 0; idof < glob_dofs; idof++)
            // {
            //     if (true_onevec->Size())
            //     {
            //         *true_onevec = 0.;
            //         (*true_onevec)(idof) = 1.;
            //     }
            //     hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), 0, theform, onevec);
            //     sleep(.5);
            // }
        }

        // if (num_ranks == 1)
        // {
        //     sequence[1]->GetTopology()->TrueB(0).Print("B0");
        //     sequence[1]->GetTopology()->TrueB(1).Print("B1");
        //     sequence[1]->GetTopology()->TrueB(2).Print("B2");
        // }

        if (print_progress_report)
            std::cout << "-- Solver has exited.\n";
    }
    else if (solve_form == 0) // H1 test
    {
        const auto theform = solve_form;
        auto DRSequence_FE = sequence[0]->FemSequence();
        FiniteElementSpace *fespace = DRSequence_FE->GetFeSpace(theform); // H(curl)

        auto truesol_gf = make_unique<GridFunction>(fespace);

        mfem::LinearForm bform(fespace);
        auto source = [&nDimensions, &sigma, &kappa](const Vector &x)
        {
            if (nDimensions == 3)
            {
                return (sigma + 3*(kappa * kappa)) * sin(kappa * x(1))*sin(kappa * x(2))* sin(kappa * x(0));
            }
            else
            {
                return (sigma + 2*(kappa * kappa)) * sin(kappa * x(1)) * sin(kappa * x(0));
            }
        };

        auto exact_sol = [&nDimensions, &sigma, &kappa](const Vector &x)
        {
            if (nDimensions == 3)
            {
                return sin(kappa * x(1)) * sin(kappa * x(2)) * sin(kappa * x(0));
            }
            else
            {
                return sin(kappa * x(1)) * sin(kappa * x(0));
            }
        };
        auto exact_Dsol = [&nDimensions, &sigma, &kappa](const Vector &x, Vector &Du)
        {
            if (nDimensions == 3)
            {
                Du(0) = kappa * sin(kappa * x(1)) * sin(kappa * x(2)) * cos(kappa * x(0));
                Du(1) = kappa * cos(kappa * x(1)) * sin(kappa * x(2)) * sin(kappa * x(0));
                Du(2) = kappa * sin(kappa * x(1)) * cos(kappa * x(2)) * sin(kappa * x(0));
            }
            else
            {
                Du(0) = kappa * sin(kappa * x(1)) * cos(kappa * x(0));
                Du(1) = kappa * cos(kappa * x(1)) * sin(kappa * x(0));
            }
        };
        VectorFunctionCoefficient Dexact(nDimensions, exact_Dsol);

        FunctionCoefficient exact(exact_sol);
        truesol_gf->ProjectCoefficient(exact);

        FunctionCoefficient source_coeff(source);
        bform.AddDomainIntegrator(new DomainLFIntegrator(source_coeff));
        bform.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(Dexact));

        bform.Assemble();

        // Project rhs down to the level of interest
        auto rhs_u = make_unique<mfem::Vector>(sequence[0]->GetNumTrueDofs(theform));
        *rhs_u = 0.0;
        sequence[0]->GetDofHandler(theform)->GetDofTrueDof().Assemble(bform, *rhs_u);
        auto truesol = make_unique<mfem::Vector>(sequence[0]->GetNumTrueDofs(theform));
        sequence[0]->GetDofHandler(theform)->GetDofTrueDof().IgnoreNonLocal(*truesol_gf, *truesol);

        for (int ii = 0; ii < start_level; ++ii)
        {
            auto tmp = make_unique<mfem::Vector>(
                hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(ii + 1))[ii + 1]->GetNumTrueDofs(theform));
            hierarchy.ApplyTruePTranspose(ii, theform, *rhs_u, *tmp);
            rhs_u = std::move(tmp);
            tmp = make_unique<mfem::Vector>(
                hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(ii + 1))[ii + 1]->GetNumTrueDofs(theform));
            hierarchy.ApplyTruePi(ii, theform, *truesol, *tmp);
            truesol = std::move(tmp);
        }

        const SharingMap &h1_dofTrueDof = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetDofHandler(theform)->GetDofTrueDof();

        // Create the parallel linear system
        shared_ptr<mfem::HypreParMatrix> pA;
        size_t local_nnz = 0;
        mfem::Vector prhs;
        {
            Vector truesoltmp(h1_dofTrueDof.GetLocalSize()), rhstmp(h1_dofTrueDof.GetLocalSize());
            h1_dofTrueDof.DisAssemble(*rhs_u, rhstmp);
            h1_dofTrueDof.DisAssemble(*truesol, truesoltmp);
            if (print_progress_report)
                std::cout << "-- Building operator on level " << start_level
                          << "...\n";

            // The blocks, managed here
            auto M = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->ComputeMassOperator(theform);
            auto W = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->ComputeMassOperator(theform + 1);
            auto D = hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetDerivativeOperator(theform);

            auto A = ToUnique(Add(sigma, *M, 1.0, *RAP(*D, *W, *D)));

            mfem::Array<int> marker(A->Height());
            marker = 0;
            hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetDofHandler(theform)->MarkDofsOnSelectedBndr(
                ess_attr[0], marker);

            for (int mm = 0; mm < A->Height(); ++mm)
                if (marker[mm])
                    A->EliminateRowCol(mm, truesoltmp(mm), rhstmp);

            pA = Assemble(h1_dofTrueDof, *A, h1_dofTrueDof);

            hypre_ParCSRMatrix *tmp = *pA;
            local_nnz += tmp->diag->num_nonzeros;
            local_nnz += tmp->offd->num_nonzeros;

            // Setup right-hand side
            h1_dofTrueDof.Assemble(rhstmp, *rhs_u);
            prhs.MakeRef(*rhs_u, 0, rhs_u->Size());

            if (print_progress_report)
                std::cout << "-- Built operator on level " << start_level
                          << ".\n"
                          << "-- Assembled the linear system on level "
                          << start_level << ".\n\n";
        }

        // Report some stats on global problem size
        size_t global_height, global_width, global_nnz;
        {
            size_t local_height = pA->Height(), local_width = pA->Width();
            MPI_Reduce(&local_height, &global_height, 1, GetMPIType(local_height),
                       MPI_SUM, 0, comm);
            MPI_Reduce(&local_width, &global_width, 1, GetMPIType(local_width),
                       MPI_SUM, 0, comm);
            MPI_Reduce(&local_nnz, &global_nnz, 1, GetMPIType<size_t>(local_nnz),
                       MPI_SUM, 0, comm);
        }
        PARELAG_ASSERT(prhs.Size() == pA->Height());

        //
        // Create the preconditioner
        //

        // Start with the solver library
        ParameterList &prec_list = master_list->Sublist("Preconditioner Library");
        auto lib = SolverLibrary::CreateLibrary(prec_list);

        // Get the factory
        const std::string solver_type = "CG-AMG";
        auto prec_factory = lib->GetSolverFactory(solver_type);
        // const int rescale_iter = prec_list.Sublist(solver_type).Sublist(
        //         "Solver Parameters").Get<int>("RescaleIteration");

        auto solver_state = prec_factory->GetDefaultState();
        solver_state->SetDeRhamSequence(hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]);
        solver_state->SetBoundaryLabels(ess_attr);
        solver_state->SetForms({theform});

        unique_ptr<mfem::Solver> solver;

        // Build the preconditioner
        if (print_progress_report)
            std::cout << "-- Building solver \"" << solver_type << "\"...\n";

        {
            Timer timer = TimeManager::AddTimer("Build Solver");
            solver = prec_factory->BuildSolver(pA, *solver_state);
            solver->iterative_mode = false;
        }

        if (print_progress_report)
            std::cout << "-- Built solver \"" << solver_type << "\".\n";

        mfem::Vector psol(prhs.Size());
        psol = 0.;

        if (!myid)
            std::cout << '\n'
                      << std::string(50, '*') << '\n'
                      << "*    Solving on level: " << start_level << '\n'
                      << "*              A size: "
                      << global_height << 'x' << global_width << '\n'
                      << "*               A NNZ: " << global_nnz << "\n*\n"
                      << "*              Solver: " << solver_type << "\n"
                      << std::string(50, '*') << '\n'
                      << std::endl;

        if (print_progress_report)
            std::cout << "-- Solving system with " << solver_type << "...\n";
        {
            double local_norm = prhs.Norml2() * prhs.Norml2();
            double global_norm;
            MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
                       MPI_SUM, 0, comm);

            if (!myid)
                std::cout << "Initial residual norm: " << std::sqrt(global_norm)
                          << std::endl;
        }

        solver->Mult(prhs, psol);

        {
            mfem::Vector tmp(pA->Height());
            pA->Mult(psol, tmp);
            prhs -= tmp;
            double local_norm = prhs.Norml2() * prhs.Norml2();
            double global_norm;
            MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
                       MPI_SUM, 0, comm);

            if (!myid)
            {
                std::cout << "Final residual norm: " << std::sqrt(global_norm)
                          << std::endl;
                auto hybrid_solver = dynamic_cast<HybridizationSolver *>(solver.get());
                if (hybrid_solver)
                {
                    std::cout << "Number of iterations: " << hybrid_solver->GetNumIters()
                              << std::endl;
                }
            }
        }

        {
            mfem::Vector err(psol);
            err -= *truesol;
            double local_norm = err.Normlinf();
            double global_norm;
            MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
            MPI_MAX, 0, comm);

            if (!myid)
            {
                std::cout << "||u* - uh||_infty : " << (global_norm) << std::endl;
            }
        }

        if (visualize)
        {
            int k_l = hierarchy.GetRedistributionIndex(start_level);
            int level_numgroups = hierarchy.GetNumGlobalCopies(k_l);
            Vector elno_vec(hierarchy.GetDeRhamSequences(hierarchy.GetRedistributionIndex(start_level))[start_level]->GetNumTrueDofs(nDimensions));
            MultiVector u(psol.GetData(), 1, psol.Size());
            MultiVector elno(elno_vec.GetData(), 1, elno_vec.Size());

            {
                auto fname = mfem::MakeParFilename("sol.gf", myid, string("of").append(to_string(num_ranks)), to_string(num_ranks).length());
                ofstream fout(fname);
                truesol->Print_HYPRE(fout);
                fout.close();
            }

            *truesol = 1.;
            MultiVector trueU(truesol->GetData(), 1, truesol->Size());

            int loc = elno_vec.Size();
            int o = 0;
            MPI_Exscan(&loc, &o, 1, MPI_INT, MPI_SUM, hierarchy.GetComm(hierarchy.GetRedistributionIndex(start_level)));
            for (auto &&pp : elno_vec)
                pp = o++;
            if (!prob_list.Get("Visualize multiple copies", false))
                level_numgroups = 1;
            for (int groupid(0); groupid < level_numgroups; groupid++)
            {
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, theform, u);
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, theform, trueU);
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, nDimensions, elno);
            }
        }

        if (print_progress_report)
            std::cout << "-- Solver has exited.\n";
    }


    if (print_progress_report)
        std::cout << "-- Good bye!\n\n";

    total_timer.Stop();

    if (print_time)
        TimeManager::Print(std::cout);

    return EXIT_SUCCESS;
}
