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
#include <csignal>

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

void debug_break_on_me(MPI_Comm *comm, int *, ...)
{
    // break on me
    std::raise(SIGABRT);
}

mfem::Vector ones({1.,1.,1.});
VectorConstantCoefficient one_coeff(ones);

double sigma = 1;
double kappa = 0.1 * M_PI;

double solve_darcy(int nDimensions, SequenceHierarchy &hierarchy, const int solve_level, MPI_Comm comm, ParameterList* master_list, ParameterList& prob_list, int myid, int num_ranks);

double solve_Hdiv(int nDimensions, SequenceHierarchy &hierarchy, const int solve_level, MPI_Comm comm, ParameterList* master_list, ParameterList& prob_list, int myid, int num_ranks, std::vector<mfem::Array<int>> &ess_attr);

double solve_Hcurl(int nDimensions, SequenceHierarchy &hierarchy, const int solve_level, MPI_Comm comm, ParameterList* master_list, ParameterList& prob_list, int myid, int num_ranks, std::vector<mfem::Array<int>> &ess_attr);

double solve_H1(int nDimensions, SequenceHierarchy &hierarchy, const int solve_level, MPI_Comm comm, ParameterList* master_list, ParameterList& prob_list, int myid, int num_ranks, std::vector<mfem::Array<int>> &ess_attr);

int main(int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    auto total_timer = TimeManager::AddTimer("ZTotal time");

    // install our own error handler (for easier debugging)
    MPI_Errhandler errhandle;
    MPI_Comm_create_errhandler(debug_break_on_me, &errhandle);
    MPI_Comm_set_errhandler(comm, errhandle);

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
                  << "-- Welcome to RedistributeTestGeometric with FE solutions!\n\n";

    std::ostringstream mesh_msg;
    if (!myid)
    {
        mesh_msg << '\n'
                 << std::string(50, '*') << '\n'
                 << "*  Mesh: " << meshfile << "\n*\n";
    }

    int ser_ref_levels;
    std::vector<SerialRefinementInfo> serial_refinements;
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
            {
                auto hilbtimer = TimeManager::AddTimer("Mesh : reorder");
                Array<int> hilb;
                mesh->GetHilbertElementOrdering(hilb);
                mesh->ReorderElements(hilb);
            }
            // mesh->EnsureNCMesh();
            imesh.close();

            if (print_progress_report)
                std::cout << "-- Read mesh \"" << meshfile
                          << "\" successfully.\n";
        }

        ser_ref_levels =
            prob_list.Get("Serial refinement levels", -1);

        // Array<int> part;
        // if (part.Size() == 0 && mesh->GetNE() >= num_ranks)
        // {
        //     part.MakeRef(mesh->GeneratePartitioning(num_ranks), mesh->GetNE());
        // }

        // This will do no refs if ser_ref_levels <= 0.
        for (int l = 0; l < ser_ref_levels; l++)
        {
            mesh->UniformRefinement();
            // if (part.Size() == 0 && mesh->GetNE() >= num_ranks)
            // {
            //     part.MakeRef(mesh->GeneratePartitioning(num_ranks), mesh->GetNE());
            // }
        }

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
        {
            serial_refinements.resize(ser_ref_levels);
            // if (ser_ref_levels)
            //     part.Copy(serial_refinements[0].partition);
        }


        // FIXME (aschaf 08/22/23) : when using METIS to generate the parallel distribution there are problems with processor boundaries when redistributing back to a single processor
        pmesh = BuildParallelMesh(comm, *mesh, serial_refinements, prob_list);

        if (pmesh && print_progress_report)
            std::cout << "-- Built parallel mesh successfully.\n"
                      << std::flush;
    }

    // Mark no boundary attributes as essential
    std::vector<mfem::Array<int>> ess_attr(1);
    ess_attr[0].SetSize(pmesh->bdr_attributes.Max());
    ess_attr[0] = 0;

    const int nDimensions = pmesh->Dimension();
    // verify nDimensions == 3

    const int nGeometricLevels = par_ref_levels + 1;
    std::vector<int> num_elements(nGeometricLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        num_elements[par_ref_levels - l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    if (unstructured)
    {
        num_elements.resize(1);
        serial_refinements.resize(0);
    }
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

    ConstantCoefficient coeffs[4] = {
        ConstantCoefficient(1.), // coeffL2
        ConstantCoefficient(1.), // coeffHdiv
        ConstantCoefficient(1.), // coeffH1
        ConstantCoefficient(1.)  // coeffHcurl
    };

    SequenceHierarchy hierarchy(pmesh, prob_list, print_progress_report);
    const auto start_form = prob_list.Get("Start form", nDimensions - 1); // TODO: read from ParameterList

    {
        auto timer = TimeManager::AddTimer("SequenceHierarchy : set coefficients");
        for (int jform = start_form; jform <= nDimensions; jform++)
            hierarchy.SetCoefficient(jform, coeffs[jform], (jform == nDimensions));
    }

    for (auto &&sr : serial_refinements)
        sr.elem_redist_procs.resize(0);

    hierarchy.SetSerialRefinementInfos(serial_refinements);
    hierarchy.Build(move(num_elements));

    std::list<std::string> list_of_problems;
    if (prob_list.IsParameter("List of problems"))
        list_of_problems = prob_list.Get<std::list<std::string>>("List of problems");

    std::map<std::string, std::vector<double>> result_table;

    for (int ilevel = start_level; ilevel < hierarchy.GetDeRhamSequences().size(); ilevel++)
    {
        for (const auto& problem : list_of_problems)
        {
            if ((problem.compare("Darcy") == 0) && start_form <= 2) // Darcy test
            {
                result_table[problem].push_back(solve_darcy(nDimensions, hierarchy, ilevel, comm, master_list.get(), prob_list, myid, num_ranks));
            }
            else if ((problem.compare("Hcurl") == 0) && start_form <= 1) // H(curl) test
            {
                result_table[problem].push_back(solve_Hcurl(nDimensions, hierarchy, ilevel, comm, master_list.get(), prob_list, myid, num_ranks, ess_attr));
            }
            else if ((problem.compare("H1") == 0) && start_form <= 0) // H1 test
            {
                result_table[problem].push_back(solve_H1(nDimensions, hierarchy, ilevel, comm, master_list.get(), prob_list, myid, num_ranks, ess_attr));
            }
            else if ((problem.compare("Hdiv") == 0) && start_form <= 2)
                result_table[problem].push_back(solve_Hdiv(nDimensions, hierarchy, ilevel, comm, master_list.get(), prob_list, myid, num_ranks, ess_attr));
        }
    }

    MPI_Barrier(comm);
    if (visualize)
    {
        auto &seq = hierarchy.GetDeRhamSequences()[start_level];
        Vector elem_no(seq->GetNumTrueDofs(nDimensions));
        Vector rank_no(seq->GetNumTrueDofs(nDimensions));
        MultiVector elems(elem_no.GetData(), 1, elem_no.Size());
        MultiVector rank(rank_no.GetData(), 1, rank_no.Size());


        int loc = elem_no.Size();
        int o = 0;
        MPI_Exscan(&loc, &o, 1, MPI_INT, MPI_SUM, comm);
        for (auto &&pp : elems)
            pp = o++;
        for (auto &&pp : rank)
            pp = myid;
        {
            seq->ShowTrueData(nDimensions, rank);
            // hierarchy.ShowTrueData(start_level, , groupid, nDimensions, rank);
            // sleep(1);
            // MPI_Barrier(comm);
            // hierarchy.ShowTrueData(start_level, , groupid, nDimensions, elems);
        }
    }

    if (!myid)
    {
        std::cout << "\n-- Summary of results: Table of L2 errors" << std::endl;
        const int maxlevel = hierarchy.GetDeRhamSequences().size();
        std::cout << std::left << std::setw(std::to_string(maxlevel).size()) << "l";
        for (auto && p : result_table)
            std::cout << " | " << std::left << std::setw(12) << p.first;
        std::cout << endl;

        std::cout.precision(6);
        std::cout << std::scientific;
        for (int ilevel = start_level; ilevel < maxlevel; ilevel++)
        {
            std::cout << ilevel;
            for (auto && p : result_table)
                std::cout << " | " << p.second[ilevel - start_level];
            std::cout << endl;
        }
    }

    if (print_progress_report)
        std::cout << "-- Good bye!\n\n";

    total_timer.Stop();

    if (print_time)
        TimeManager::Print(std::cout);

    return EXIT_SUCCESS;
}


double solve_darcy(int nDimensions, SequenceHierarchy &hierarchy, const int solve_level, MPI_Comm comm, ParameterList* master_list, ParameterList& prob_list, int myid, int num_ranks)
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
        u = -1.;
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

        return xi + yi + zi; //exp(xi) * sin(yi) * cos(zi);
    };

    auto fFun = [](const Vector &x, Vector &f)
    {
        f = 0.0;
    };

    auto gFun = [&pFun_ex](const Vector &x)
    {
        return 0.;
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

    const int uform = 2;
    const int pform = 2 + 1;
    const int start_level = solve_level;
    auto &sequence = hierarchy.GetDeRhamSequences();
    const bool show_progress = master_list->Sublist("Output control").Get("Show progress Darcy", false);
    const bool visualize = master_list->Sublist("Output control").Get("Visualize solution Darcy", false);
    const bool print_progress_report = (!myid && show_progress);
    double global_error;

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
            sequence[ii + 1]->GetNumTrueDofs(uform));
        auto tmp_p = make_unique<mfem::Vector>(
            sequence[ii + 1]->GetNumTrueDofs(pform));
        sequence[ii]->GetTrueP(uform).MultTranspose(*rhs_u,*tmp_u);
        // sequence[ii]->ApplyTruePTranspose(uform,*rhs_u,*tmp_u);
        // sequence[ii]->GetTrueP(uform).MultTranspose(*rhs_u, *tmp_u);
        sequence[ii]->GetTrueP(pform).MultTranspose(*rhs_p,*tmp_p);
        // sequence[ii]->ApplyTruePTranspose(pform,*rhs_p,*tmp_p);
        // sequence[ii]->GetTrueP(pform).MultTranspose(*rhs_p, *tmp_p);
        rhs_u = std::move(tmp_u);
        rhs_p = std::move(tmp_p);
        tmp_u = make_unique<mfem::Vector>(
            sequence[ii + 1]->GetNumTrueDofs(uform));
        tmp_p = make_unique<mfem::Vector>(
            sequence[ii + 1]->GetNumTrueDofs(pform));
        sequence[ii]->GetTruePi(uform).Mult(*trueu, *tmp_u);
        sequence[ii]->GetTruePi(pform).Mult(*truep, *tmp_p);
        // sequence[ii]->GetTruePi(tform, ).MultTranspose(rueu, *tmp_u);
        // sequence[ii]->GetTruePi(tform, ).MultTranspose(ruep, *tmp_p);
        truep = std::move(tmp_p);
        trueu = std::move(tmp_u);
        tmp_u = make_unique<mfem::Vector>(
            sequence[ii + 1]->GetNumTrueDofs(uform));
        // sequence[ii]->GetTruePi(tform, ).MultTranspose(div_const, *tmp_u);
        sequence[ii]->GetTruePi(uform).Mult(*hdiv_const, *tmp_u);
        hdiv_const = std::move(tmp_u);
    }

    const SharingMap &hdiv_dofTrueDof = hierarchy.GetDeRhamSequences()[start_level]->GetDofHandler(uform)->GetDofTrueDof();
    const SharingMap &l2_dofTrueDof = hierarchy.GetDeRhamSequences()[start_level]->GetDofHandler(pform)->GetDofTrueDof();

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
        auto M = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(uform);
        auto W = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(pform);
        auto D = hierarchy.GetDeRhamSequences()[start_level]->GetDerivativeOperator(uform);

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
    const std::string solver_type = prob_list.Get("Linear solver Darcy", "MINRES-BlkJacobi-GS-AMG");
    auto prec_factory = lib->GetSolverFactory(solver_type);
    const int rescale_iter = prec_list.Sublist(solver_type).Sublist("Solver Parameters").Get("RescaleIteration", -20);

    auto solver_state = prec_factory->GetDefaultState();
    solver_state->SetDeRhamSequence(hierarchy.GetDeRhamSequences()[start_level]);
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
        Timer timer = TimeManager::AddTimer("Darcy : Build Solver");
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
        auto timer = TimeManager::AddTimer("Darcy : Solve");
        solver->Mult(prhs, psol);
    }

    {
        if (solver_type.compare("Hybridization-Darcy-CG-AMG") == 0)
            psol.GetBlock(1) *= -1.;
        mfem::Vector tmp(A->Height());
        A->Mult(psol, tmp);
        tmp -= prhs;
        double local_norm = tmp.Norml2() * tmp.Norml2();
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
        mfem::BlockVector perr(psol);
        perr -= ptruesol;
        auto M = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(uform);
        auto W = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(pform);
        auto pM = Assemble(hdiv_dofTrueDof, *M, hdiv_dofTrueDof);
        auto pW = Assemble(l2_dofTrueDof, *W, l2_dofTrueDof);
        mfem::Vector err(perr.GetData(), psol.BlockSize(0));
        mfem::Vector Merr(err.Size());
        pM->Mult(err, Merr);
        double global_norm = global_error = sqrt(InnerProduct(comm, Merr, err));
        if (print_progress_report)
        {
            std::cout << "||u* - uh||_L2 : " << (global_norm) << std::endl;
        }
        err.SetDataAndSize(perr.GetData() + psol.BlockSize(0), psol.BlockSize(1));
        Merr.SetSize(err.Size());
        pW->Mult(err, Merr);
        global_norm = sqrt(InnerProduct(comm, Merr, err));
        global_error = sqrt(global_error*global_error + global_norm*global_norm);
        if (print_progress_report)
        {
            std::cout << "||p* - ph||_L2 : " << (global_norm) << std::endl;

#ifdef PARELAG_DEBUG_darcy_output
            ofstream ofile;
            if (num_ranks > 1)
                ofile.open("psol.darcy.parallel.00000");
            else
                ofile.open("psol.darcy.serial.00000");
            psol.Print_HYPRE(ofile);
            ofile.close();
            if (num_ranks > 1)
                ofile.open("prhs.darcy.parallel.00000");
            else
                ofile.open("prhs.darcy.serial.00000");
            prhs.Print_HYPRE(ofile);
            ofile.close();
#endif // PARELAG_DEBUG_darcy_output
        }
    }


    if (visualize)
    {
        auto seql = hierarchy.GetDeRhamSequences()[start_level];
        auto size = seql->GetNumTrueDofs(nDimensions);
        MultiVector el_myids(1, size);
        MultiVector el_nos(1, size);

        int loc = size;
        int o = 0;
        MPI_Exscan(&loc, &o, 1, MPI_INT, MPI_SUM, comm);
        for (auto &&elno : el_nos)
            elno = o++;
        for (auto &&eid : el_myids)
            eid = myid;
        {
            seql->ShowTrueData(nDimensions, el_myids);
            seql->ShowTrueData(nDimensions, el_nos);
        }
    }

    if (print_progress_report)
        std::cout << "-- Darcy Solver has exited.\n" << std::endl;

    return global_error;
}

double solve_Hcurl(int nDimensions, SequenceHierarchy &hierarchy, const int solve_level, MPI_Comm comm, ParameterList* master_list, ParameterList& prob_list, int myid, int num_ranks, std::vector<mfem::Array<int>> &ess_attr)
{
    const bool print_progress_report = false;
    const bool visualize = false;
    if (print_progress_report)
        std::cout << std::endl << "-- H(curl) Solver started!" << std::endl;
    const int theform = 1;
    const int start_level = solve_level;
    auto &sequence = hierarchy.GetDeRhamSequences();
    double global_error;

    auto DRSequence_FE = sequence[0]->FemSequence();
    FiniteElementSpace *fespace = DRSequence_FE->GetFeSpace(theform); // H(curl)

    auto truesol_gf = make_unique<GridFunction>(fespace);
    auto onevec_gf = make_unique<GridFunction>(fespace);

    mfem::LinearForm bform(fespace);
    auto source_vec = [&nDimensions](const Vector &x, Vector &f)
    {
        if (nDimensions == 3)
        {
            f(0) = sigma*x(1); // (sigma + kappa * kappa) * sin(kappa * x(1));
            f(1) = sigma*x(2); // (sigma + kappa * kappa) * sin(kappa * x(2));
            f(2) = sigma*x(0); // (sigma + kappa * kappa) * sin(kappa * x(0));
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

    auto exact_sol = [&nDimensions](const Vector &x, Vector &E)
    {
        if (nDimensions == 3)
        {
            E(0) = x(1); // sin(kappa * x(1));
            E(1) = x(2); // sin(kappa * x(2));
            E(2) = x(0); // sin(kappa * x(0));
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
    auto exact_Dsol = [&nDimensions](const Vector &x, Vector &E)
    {
        if (nDimensions == 3)
        {
            E(0) = -1.; // -kappa * cos(kappa * x(2));
            E(1) = -1.; // -kappa * cos(kappa * x(0));
            E(2) = -1.; // -kappa * cos(kappa * x(1));
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
            sequence[ii + 1]->GetNumTrueDofs(theform));
        sequence[ii]->GetTrueP(theform).MultTranspose(*rhs_u, *tmp);
        rhs_u = std::move(tmp);
        tmp = make_unique<mfem::Vector>(
            sequence[ii + 1]->GetNumTrueDofs(theform));
        sequence[ii]->GetTruePi(theform).Mult(*truesol, *tmp);
        truesol = std::move(tmp);
        tmp = make_unique<mfem::Vector>(
            sequence[ii + 1]->GetNumTrueDofs(theform));
        sequence[ii]->GetTruePi(theform).Mult(*true_onevec, *tmp);
        true_onevec = std::move(tmp);
    }

    const SharingMap &hcurl_dofTrueDof = hierarchy.GetDeRhamSequences()[start_level]->GetDofHandler(theform)->GetDofTrueDof();

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
        auto M = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(theform);
        auto W = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(theform + 1);
        auto D = hierarchy.GetDeRhamSequences()[start_level]->GetDerivativeOperator(theform);

        auto A = ToUnique(Add(sigma, *M, 1.0, *RAP(*D, *W, *D)));

        mfem::Array<int> marker(A->Height());
        marker = 0;
        hierarchy.GetDeRhamSequences()[start_level]->GetDofHandler(theform)->MarkDofsOnSelectedBndr(
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
    mfem::Vector psol(pA->Width());
    psol = 0.;
    bool solve = prob_list.Get("Solve the problem", true);
    if (solve)
    {
        //
        // Create the preconditioner
        //

        // Start with the solver library
        ParameterList &prec_list = master_list->Sublist("Preconditioner Library");
        const std::string solver_type = prob_list.Get("Linear solver 1-Form", "CG-AMG");
        {
            auto &solver_parameter_list = prec_list.Sublist(solver_type).Sublist("Solver Parameters");
            solver_parameter_list.Set("Relative tolerance", 1e-16);
            solver_parameter_list.Set("Absolute tolerance", 1e-16);
            solver_parameter_list.Set("Maximum iterations", 1000);
            solver_parameter_list.Set("Print level", 2);
        }

        auto lib = SolverLibrary::CreateLibrary(prec_list);

        // Get the factory
        auto prec_factory = lib->GetSolverFactory(solver_type);
        // const int rescale_iter = prec_list.Sublist(solver_type).Sublist(
        //         "Solver Parameters").Get<int>("RescaleIteration");

        auto solver_state = prec_factory->GetDefaultState();
        solver_state->SetDeRhamSequence(hierarchy.GetDeRhamSequences()[start_level]);
        // if (ess_attr)
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

        if (print_progress_report)
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

            if (print_progress_report)
                std::cout << "Initial residual norm: " << std::sqrt(global_norm)
                            << std::endl;
        }

        {
            auto timer = TimeManager::AddTimer("H(curl) : Solve");
            solver->Mult(prhs, psol);
        }

        {
            mfem::Vector tmp(pA->Height());
            pA->Mult(psol, tmp);
            tmp -= prhs;
            double local_norm = tmp.Norml2() * tmp.Norml2();
            double global_norm;
            MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
                        MPI_SUM, 0, comm);

            if (print_progress_report)
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
            auto timer = TimeManager::AddTimer("H(curl) : compute errors");
            mfem::Vector err(psol);
            err -= *truesol;
            auto M = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(theform);
            auto pM = Assemble(hcurl_dofTrueDof, *M, hcurl_dofTrueDof);

            mfem::Vector Merr(err.Size());
            pM->Mult(err, Merr);
            global_error = sqrt(mfem::InnerProduct(comm, Merr, err));
            if (print_progress_report)
            {
                std::cout << "||u* - uh||_L2 = " << (global_error) << std::endl;

#ifdef PARELAG_DEBUG_1form_output
                ofstream ofile;
                if (num_ranks > 1)
                    ofile.open("psol.1form.parallel.00000");
                else
                    ofile.open("psol.1form.serial.00000");
                psol.Print_HYPRE(ofile);
                ofile.close();
                if (num_ranks > 1)
                    ofile.open("prhs.1form.parallel.00000");
                else
                    ofile.open("prhs.1form.serial.00000");
                prhs.Print_HYPRE(ofile);
                ofile.close();
#endif
            }
#ifdef PARELAG_DEBUG_1form_output
            auto *topo = sequence[start_level]->GetTopology();
            if (num_ranks > 1)
            {
                topo->TrueB(0).Print("trueB0.parallel");
                topo->TrueB(1).Print("trueB1.parallel");
            }
            else
            {
                topo->TrueB(0).Print("trueB0.serial");
                topo->TrueB(1).Print("trueB1.serial");
            }
#endif
        }
    }

    if (visualize)
    {
        MultiVector u(psol.GetData(), 1, psol.Size());
        int o = 0;
        for (auto &&uval : u)
            uval = o++;

        {
            // hierarchy.ShowTrueData(start_level, , groupid, theform, u);
        }
    }

    if (print_progress_report)
        std::cout << "-- H(curl) Solver has exited.\n" << std::endl;

    return global_error;
}

double solve_H1(int nDimensions, SequenceHierarchy &hierarchy, const int solve_level, MPI_Comm comm, ParameterList* master_list, ParameterList& prob_list, int myid, int num_ranks, std::vector<mfem::Array<int>> &ess_attr)
{
    const bool print_progress_report = false;
    const bool visualize = false;
    const int start_level = solve_level;
    auto &sequence = hierarchy.GetDeRhamSequences();
    const auto theform = 0;
    auto DRSequence_FE = sequence[0]->FemSequence();
    FiniteElementSpace *fespace = DRSequence_FE->GetFeSpace(theform); // H1
    double global_error;

    auto truesol_gf = make_unique<GridFunction>(fespace);

    mfem::LinearForm bform(fespace);
    auto source = [&nDimensions](const Vector &x)
    {
        if (nDimensions == 3)
        {
            return sigma * (x(0) + x(1) + x(2));
            return (sigma + 3*(kappa * kappa)) * sin(kappa * x(1))*sin(kappa * x(2))* sin(kappa * x(0));
        }
        else
        {
            return (sigma + 2*(kappa * kappa)) * sin(kappa * x(1)) * sin(kappa * x(0));
        }
    };

    auto exact_sol = [&nDimensions](const Vector &x)
    {
        if (nDimensions == 3)
        {
            return x(0) + x(1) + x(2);
            return sin(kappa * x(1)) * sin(kappa * x(2)) * sin(kappa * x(0));
        }
        else
        {
            return sin(kappa * x(1)) * sin(kappa * x(0));
        }
    };
    auto exact_Dsol = [&nDimensions](const Vector &x, Vector &Du)
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
        Du = 1.;
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
            sequence[ii + 1]->GetNumTrueDofs(theform));
        sequence[ii]->GetTrueP(theform).MultTranspose(*rhs_u, *tmp);
        rhs_u = std::move(tmp);
        tmp = make_unique<mfem::Vector>(
            sequence[ii + 1]->GetNumTrueDofs(theform));
        sequence[ii]->GetTruePi(theform).Mult(*truesol, *tmp);
        truesol = std::move(tmp);
    }

    const SharingMap &h1_dofTrueDof = hierarchy.GetDeRhamSequences()[start_level]->GetDofHandler(theform)->GetDofTrueDof();

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
        auto M = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(theform);
        auto W = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(theform + 1);
        auto D = hierarchy.GetDeRhamSequences()[start_level]->GetDerivativeOperator(theform);

        auto A = ToUnique(Add(sigma, *M, 1.0, *RAP(*D, *W, *D)));

        mfem::Array<int> marker(A->Height());
        marker = 0;
        hierarchy.GetDeRhamSequences()[start_level]->GetDofHandler(theform)->MarkDofsOnSelectedBndr(
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
    const std::string solver_type = "CG-AMG";
    {
        auto &solver_parameter_list = prec_list.Sublist(solver_type).Sublist("Solver Parameters");
        solver_parameter_list.Set("Relative tolerance", 1e-16);
        solver_parameter_list.Set("Absolute tolerance", 1e-16);
        solver_parameter_list.Set("Maximum iterations", 1000);
        solver_parameter_list.Set("Print level", 2);
    }

    auto lib = SolverLibrary::CreateLibrary(prec_list);

    // Get the factory
    auto prec_factory = lib->GetSolverFactory(solver_type);
    // const int rescale_iter = prec_list.Sublist(solver_type).Sublist(
    //         "Solver Parameters").Get<int>("RescaleIteration");

    auto solver_state = prec_factory->GetDefaultState();
    solver_state->SetDeRhamSequence(hierarchy.GetDeRhamSequences()[start_level]);
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

    if (print_progress_report)
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

        if (print_progress_report)
            std::cout << "Initial residual norm: " << std::sqrt(global_norm)
                        << std::endl;
    }

    {
        auto timer = TimeManager::AddTimer("H1 : Solve");
        solver->Mult(prhs, psol);
    }

    {
        mfem::Vector tmp(pA->Height());
        pA->Mult(psol, tmp);
        tmp -= prhs;
        double local_norm = tmp.Norml2() * tmp.Norml2();
        double global_norm;
        MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
                    MPI_SUM, 0, comm);

        if (print_progress_report)
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
        auto M = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(theform);
        auto pM = Assemble(h1_dofTrueDof, *M, h1_dofTrueDof);

        mfem::Vector Merr(err.Size());
        pM->Mult(err, Merr);
        global_error = sqrt(mfem::InnerProduct(comm, Merr, err));
        if (print_progress_report)
        {
            std::cout << "||u* - uh||_L2 : " << (global_error) << std::endl;
#ifdef PARELAG_DEBUG_0form_output
            ofstream ofile;
            if (num_ranks > 1)
                ofile.open("psol.0form.parallel.00000");
            else
                ofile.open("psol.0form.serial.00000");
            psol.Print_HYPRE(ofile);
            ofile.close();
            if (num_ranks > 1)
                ofile.open("prhs.0form.parallel.00000");
            else
                ofile.open("prhs.0form.serial.00000");
            prhs.Print_HYPRE(ofile);
            ofile.close();
            if (num_ranks > 1)
                ofile.open("truesol.0form.parallel.00000");
            else
                ofile.open("truesol.0form.serial.00000");
            truesol->Print_HYPRE(ofile);
            ofile.close();
#endif
        }

    }

    if (visualize)
    {
        MultiVector u(psol.GetData(), 1, psol.Size());

        MultiVector trueU(truesol->GetData(), 1, truesol->Size());
        int loc = truesol->Size();
        int o = 0;
        MPI_Exscan(&loc, &o, 1, MPI_INT, MPI_SUM, comm);
        for (auto &&a : *truesol)
            a = o++;

        {
            // hierarchy.ShowTrueData(start_level, , groupid, theform, u);
            // hierarchy.ShowTrueData(start_level, , groupid, theform, trueU);
        }
    }

    if (print_progress_report)
        std::cout << "-- H1 Solver has exited.\n" << std::endl;

    return global_error;
}

double solve_Hdiv(int nDimensions, SequenceHierarchy &hierarchy, const int solve_level, MPI_Comm comm, ParameterList* master_list, ParameterList& prob_list, int myid, int num_ranks, std::vector<mfem::Array<int>> &ess_attr)
{
    const bool print_progress_report = false;
    const bool visualize = false;
    if (print_progress_report)
        std::cout << std::endl << "-- H(div) Solver started!" << std::endl;
    const int theform = 2;
    const int start_level = solve_level;
    auto &sequence = hierarchy.GetDeRhamSequences();
    double global_error;

    auto DRSequence_FE = sequence[0]->FemSequence();
    FiniteElementSpace *fespace = DRSequence_FE->GetFeSpace(theform); // H(div)

    auto truesol_gf = make_unique<GridFunction>(fespace);
    auto onevec_gf = make_unique<GridFunction>(fespace);

    mfem::LinearForm bform(fespace);
    auto source_vec = [&nDimensions](const Vector &x, Vector &f)
    {
        if (nDimensions == 3)
        {
            f(0) = sigma*x(0);
            f(1) = sigma*x(1);
            f(2) = sigma*x(2);
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

    auto exact_sol = [&nDimensions](const Vector &x, Vector &E)
    {
        if (nDimensions == 3)
        {
            E(0) = x(0);
            E(1) = x(1);
            E(2) = x(2);
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
    auto exact_Dsol = [&nDimensions](const Vector &x)
    {
        if (nDimensions == 3)
        {
            return 3.;
        }
        else
        {
            return 2.;
        }
        return 3.;
    };
    VectorFunctionCoefficient exact(nDimensions, exact_sol);
    FunctionCoefficient Dexact(exact_Dsol);
    truesol_gf->ProjectCoefficient(exact);
    onevec_gf->ProjectCoefficient(one_coeff);

    VectorFunctionCoefficient source(nDimensions, source_vec);
    bform.AddDomainIntegrator(new VectorFEDomainLFIntegrator(source));
    bform.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(Dexact));

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
            sequence[ii + 1]->GetNumTrueDofs(theform));
        sequence[ii]->GetTrueP(theform).MultTranspose(*rhs_u, *tmp);
        rhs_u = std::move(tmp);
        tmp = make_unique<mfem::Vector>(
            sequence[ii + 1]->GetNumTrueDofs(theform));
        sequence[ii]->GetTruePi(theform).Mult(*truesol, *tmp);
        truesol = std::move(tmp);
        tmp = make_unique<mfem::Vector>(
            sequence[ii + 1]->GetNumTrueDofs(theform));
        sequence[ii]->GetTruePi(theform).Mult(*true_onevec, *tmp);
        true_onevec = std::move(tmp);
    }

    const SharingMap &hdiv_dofTrueDof = hierarchy.GetDeRhamSequences()[start_level]->GetDofHandler(theform)->GetDofTrueDof();

    // Create the parallel linear system
    shared_ptr<mfem::HypreParMatrix> pA;
    size_t local_nnz = 0;
    mfem::Vector prhs;
    {
        auto timer = TimeManager::AddTimer(string("Building operator on level ").append(to_string(start_level)));
        Vector truesoltmp(hdiv_dofTrueDof.GetLocalSize()), rhstmp(hdiv_dofTrueDof.GetLocalSize());
        hdiv_dofTrueDof.DisAssemble(*rhs_u, rhstmp);
        hdiv_dofTrueDof.DisAssemble(*truesol, truesoltmp);
        if (print_progress_report)
            std::cout << "-- Building operator on level " << start_level
                        << "...\n";

        // The blocks, managed here
        auto M = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(theform);
        auto W = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(theform + 1);
        auto D = hierarchy.GetDeRhamSequences()[start_level]->GetDerivativeOperator(theform);

        auto A = ToUnique(Add(sigma, *M, 1.0, *RAP(*D, *W, *D)));

        mfem::Array<int> marker(A->Height());
        marker = 0;
        hierarchy.GetDeRhamSequences()[start_level]->GetDofHandler(theform)->MarkDofsOnSelectedBndr(
            ess_attr[0], marker);

        for (int mm = 0; mm < A->Height(); ++mm)
            if (marker[mm])
                A->EliminateRowCol(mm, truesoltmp(mm), rhstmp);

        pA = Assemble(hdiv_dofTrueDof, *A, hdiv_dofTrueDof);

        hypre_ParCSRMatrix *tmp = *pA;
        local_nnz += tmp->diag->num_nonzeros;
        local_nnz += tmp->offd->num_nonzeros;

        // Setup right-hand side
        hdiv_dofTrueDof.IgnoreNonLocal(rhstmp, *rhs_u);
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
    mfem::Vector psol(pA->Width());
    psol = 0.;
    bool solve = prob_list.Get("Solve the problem", true);
    if (solve)
    {
        //
        // Create the preconditioner
        //

        // Get the factory
        const std::string solver_type = prob_list.Get("Linear solver 2-Form", "CG-AMG");
        // Start with the solver library
        ParameterList &prec_list = master_list->Sublist("Preconditioner Library");
        {
            auto &solver_parameter_list = prec_list.Sublist(solver_type).Sublist("Solver Parameters");
            solver_parameter_list.Set("Relative tolerance", 1e-16);
            solver_parameter_list.Set("Absolute tolerance", 1e-16);
            solver_parameter_list.Set("Maximum iterations", 1000);
            solver_parameter_list.Set("Print level", 2);
        }

        auto lib = SolverLibrary::CreateLibrary(prec_list);

        auto prec_factory = lib->GetSolverFactory(solver_type);
        // const int rescale_iter = prec_list.Sublist(solver_type).Sublist(
        //         "Solver Parameters").Get<int>("RescaleIteration");

        auto solver_state = prec_factory->GetDefaultState();
        solver_state->SetDeRhamSequence(hierarchy.GetDeRhamSequences()[start_level]);
        // if (ess_attr)
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

        if (print_progress_report)
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

            if (print_progress_report)
                std::cout << "Initial residual norm: " << std::sqrt(global_norm)
                            << std::endl;
        }

        {
            auto timer = TimeManager::AddTimer("H(div) : Solve");
            solver->Mult(prhs, psol);
        }

        {
            mfem::Vector tmp(pA->Height());
            pA->Mult(psol, tmp);
            tmp -= prhs;
            double local_norm = tmp.Norml2() * tmp.Norml2();
            double global_norm;
            MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
                        MPI_SUM, 0, comm);

            if (print_progress_report)
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
            auto timer = TimeManager::AddTimer("H(div) : compute errors");
            mfem::Vector err(psol);
            err -= *truesol;
            auto M = hierarchy.GetDeRhamSequences()[start_level]->ComputeMassOperator(theform);
            auto pM = Assemble(hdiv_dofTrueDof, *M, hdiv_dofTrueDof);

            mfem::Vector Merr(err.Size());
            pM->Mult(err, Merr);
            global_error = sqrt(mfem::InnerProduct(comm, Merr, err));
            if (print_progress_report)
            {
                std::cout << "||u* - uh||_L2 = " << (global_error) << std::endl;

#ifdef PARELAG_DEBUG_2form_output
                ofstream ofile;
                if (num_ranks > 1)
                    ofile.open("psol.2form.parallel.00000");
                else
                    ofile.open("psol.2form.serial.00000");
                psol.Print_HYPRE(ofile);
                ofile.close();
                if (num_ranks > 1)
                    ofile.open("prhs.2form.parallel.00000");
                else
                    ofile.open("prhs.2form.serial.00000");
                prhs.Print_HYPRE(ofile);
                ofile.close();
#endif
            }
        }
    }

    if (visualize)
    {
        MultiVector u(psol.GetData(), 1, psol.Size());

        int loc = psol.Size();
        int o = 0;
        MPI_Exscan(&loc, &o, 1, MPI_INT, MPI_SUM, comm);
        for (int i = 0; i < psol.Size(); i++)
            psol[i] = o++;

        {
            // hierarchy.ShowTrueData(start_level, , groupid, theform, u);
        }
    }

    if (print_progress_report)
        std::cout << "-- H(div) Solver has exited.\n" << std::endl;

    return global_error;
}
