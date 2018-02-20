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

//                       Elag Upscaling - Parallel Version
//
// solves a mixed Darcy problem with unstructured coarsening, using
// an H1-L2 solver.
//
// This is one of our Umberto benchmarks, which he documented on the
// confluence pages. It should be combined with testsuite/unstructuredDarcy.cpp

#include <fstream>
#include <list>
#include <sstream>

#include <mpi.h>

#include "elag.hpp"

#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"
#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "utilities/MPIDataTypes.hpp"
#include "utilities/ParELAG_SimpleXMLParameterListReader.hpp"
#include "utilities/ParELAG_TimeManager.hpp"

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

    Timer main_timer = TimeManager::AddTimer("Main()");

    int num_ranks, myid;
    MPI_Comm_size(comm, &num_ranks);
    MPI_Comm_rank(comm, &myid);

    // Get options from command line
    const char *xml_file_c = "example_parameters.xml";
    mfem::OptionsParser args(argc, argv);
    args.AddOption(&xml_file_c, "-f", "--xml-file",
                   "XML parameter list.");
    args.Parse();
    PARELAG_ASSERT(args.Good());
    std::string xml_file(xml_file_c);

    // Read the parameter list from file
    unique_ptr<ParameterList> master_list;
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

    // Standard problem parameters
    ParameterList& prob_list = master_list->Sublist("Problem parameters",true);
    // SPE10 specific params
    ParameterList& spe_list
        = master_list->Sublist("SPE10 problem parameters",false);
    std::string permFile = spe_list.Get("SPE10 PermFile", "data/spe_perm.dat");
    const int slice = spe_list.Get("SPE10 Slice", -1);
    std::vector<int> num_elements
        = spe_list.Get("Number of elements", std::vector<int>{60,220,85});
    std::vector<int> element_size
        = spe_list.Get("Element sizes", std::vector<int>{20,10,2});

    const auto nDimensions = (slice < 0 ? 3u : 2u);

    const int n_bdr_attributes = spe_list.Get("Number Boundary Attributes", 6);
    std::vector<int> v_ess_attr
        = spe_list.Get("Essential Attributes", std::vector<int>{1,0,1,0,1,1});
    std::vector<int> v_obs_attr
        = spe_list.Get("Observational Attributes",
                        std::vector<int>{0,1,0,0,0,0});
    std::vector<int> v_inflow_attr
        = spe_list.Get("Inflow Attributes", std::vector<int>{0,0,0,1,0,0});

    std::vector<mfem::Array<int>> ess_attr(2);
    ess_attr[0].SetSize(n_bdr_attributes);
    std::copy(v_ess_attr.begin(),v_ess_attr.end(),ess_attr[0].GetData());

    mfem::Array<int> inflow_attr(v_inflow_attr.data(),n_bdr_attributes);
    mfem::Array<int> obs_attr(v_obs_attr.data(), n_bdr_attributes);

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

    PARELAG_ASSERT(start_level <= stop_level && stop_level <= par_ref_levels);

    // The order of the finite elements
    const int feorder = prob_list.Get("Finite element order", 0);

    // The order of the polynomials to include in the space
    const int upscaling_order = prob_list.Get("Upscaling order", 0);

    const int coarsening_factor = prob_list.Get("Coarsening factor",8);
    const int agg_coarsening_levels =
        prob_list.Get("Aggressive coarsening levels",1);
    const bool derefinement = prob_list.Get("Use derefinement",false);

    ParameterList& output_list = master_list->Sublist("Output control");
    const bool print_time = output_list.Get("Print timings",true);
    const bool visualize = output_list.Get("Visualize solution",false);
    const bool show_progress = output_list.Get("Show progress",true);
    const bool save_solution = output_list.Get("Save solution",false);
    const std::string save_file_prefix = output_list.Get("Save directory","./");
    const bool print_progress_report = (!myid && show_progress);

    if (print_progress_report)
        std::cout << "\n-- Hello!\n"
                  << "-- Welcome to MultigridTestSPE10!\n\n";

    std::ostringstream mesh_msg;
    if (!myid)
    {
        mesh_msg << '\n' << std::string(50,'*') << '\n'
                 << "*  Mesh: " << meshfile << "\n*\n"
                 << "*              FE order: " << feorder << '\n'
                 << "*       Upscaling order: " << upscaling_order << '\n'
                 << "*\n" << std::boolalpha
                 << "*          Derefinement: " << derefinement << '\n'
                 << "*     Coarsening factor: " << coarsening_factor << '\n'
                 << "*   Agg. Coarsen Levels: " << agg_coarsening_levels << '\n'
                 << "*\n";
    }


    // cell sizes for fine cartesian grid (do not change this)
    Array<double> h_array(3);
    h_array[0] = element_size[0];
    h_array[1] = element_size[1];
    h_array[2] = element_size[2];

    // Load the SPE10 data set
#if 0
    {
        InversePermeabilityFunction::SetNumberCells(
            num_elements[0],num_elements[1],num_elements[2]);
        InversePermeabilityFunction::SetMeshSizes(
            element_size[0],element_size[1],element_size[2]);
        InversePermeabilityFunction::ReadPermeabilityFile(permFile, comm);
    }
#endif

    // This ensures the whole file is read in each time, regardless of sizes.
    InversePermeabilityFunction::SetNumberCells( 60, 220, 85);
    InversePermeabilityFunction::SetMeshSizes( 20, 10, 2);
    InversePermeabilityFunction::ReadPermeabilityFile(permFile, comm);

    if(nDimensions == 2)
        InversePermeabilityFunction::Set2DSlice(
            InversePermeabilityFunction::XY, slice);

    shared_ptr<ParMesh> pmesh;
    {
        if (print_progress_report)
            std::cout << "-- Building and refining serial mesh...\n";

        // Create the finite element mesh
        std::unique_ptr<Mesh> mesh;
        if (nDimensions == 3)
            mesh = make_unique<Mesh>(
                num_elements[0], num_elements[1], num_elements[2],
                Element::HEXAHEDRON, 1,
                num_elements[0]*element_size[0],
                num_elements[1]*element_size[1],
                num_elements[2]*element_size[2]);
        else
            mesh = make_unique<Mesh>(
                num_elements[0], num_elements[1],
                Element::QUADRILATERAL, 1,
                num_elements[0]*element_size[0],
                num_elements[1]*element_size[1]);

        if (print_progress_report)
            std::cout << "-- Created SPE10 mesh successfully.\n";

        if (mesh->Dimension() < 3)
        {
            std::cerr << "Must use a 3D mesh for this example." << std::endl;
            return EXIT_FAILURE;
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

        pmesh = make_shared<ParMesh>(comm, *mesh);

        if (pmesh && print_progress_report)
            std::cout << "-- Built parallel mesh successfully.\n";
    }

    // Temporary
    PARELAG_ASSERT(nDimensions == 3);

    std::vector<int> level_nElements(derefinement ? par_ref_levels+1 : 0);
    for (int l = 0; l < par_ref_levels; l++)
    {
        if (derefinement) level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    if (derefinement) level_nElements[0] = pmesh->GetNE();

    if (nDimensions > 2)
        pmesh->ReorientTetMesh();

    if (print_progress_report)
        std::cout << "-- Refined mesh in parallel " << par_ref_levels
                  << " times.\n";


    if (not derefinement)
    {
        int this_num_elements = pmesh->GetNE();
        level_nElements.push_back(this_num_elements);
        for (int i = 0; i < agg_coarsening_levels; ++i)
        {
            this_num_elements /= (coarsening_factor)*(coarsening_factor);
            level_nElements.push_back(std::max(this_num_elements,1));
            if (this_num_elements < coarsening_factor)
                break;
        }
        while (this_num_elements > coarsening_factor)
        {
            this_num_elements /= coarsening_factor;
            level_nElements.push_back(std::max(this_num_elements,1));
        }
    }

    // Print out level information
    const int nLevels = level_nElements.size();
    if (!myid)
        std::cout << "Number of levels: " << nLevels << "\n"
                  << "level_nElements: " << level_nElements << "\n";

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

    // Create the partitioner to be used
    MFEMRefinedMeshPartitioner mfem_partitioner(nDimensions);
    MetisGraphPartitioner partitioner;
    Array<int> partitioning;
    partitioner.setFlags(MetisGraphPartitioner::KWAY );// BISECTION
    partitioner.setOption(METIS_OPTION_SEED, 0);// Fix the seed
    partitioner.setOption(METIS_OPTION_CONTIG,1);// Contiguous partitions
    partitioner.setOption(METIS_OPTION_MINCONN,1);
    partitioner.setUnbalanceToll(1.05);

    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);
    constexpr auto AT_elem = AgglomeratedTopology::ELEMENT;
    {
        Timer outer = TimeManager::AddTimer("Mesh Agglomeration -- Total");
        {
            Timer inner = TimeManager::AddTimer("Mesh Agglomeration -- Level 0");
            topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
        }

        for (int ilevel = 0; ilevel < nLevels-1; ++ilevel)
        {
            if (print_progress_report)
                std::cout << "  --  Level " << ilevel + 1 << "...";

            std::ostringstream timer_name;
            timer_name << "Mesh Agglomeration -- Level " << ilevel+1;
            Timer inner = TimeManager::AddTimer(timer_name.str());

            partitioning.SetSize(level_nElements[ilevel]);

            if (derefinement)
                mfem_partitioner.Partition(
                    topology[ilevel]->GetNumberLocalEntities(AT_elem),
                    level_nElements[ilevel+1],
                    partitioning);
            else
                partitioner.doPartition(
                    *(topology[ilevel]->LocalElementElementTable()),
                    topology[ilevel]->Weight(AT_elem),
                    level_nElements[ilevel+1], partitioning);
            topology[ilevel+1] =
                topology[ilevel]->CoarsenLocalPartitioning(
                    partitioning,false,false);

            if (print_progress_report)
                std::cout << " Done!" << std::endl;
        }
    }

    if (print_progress_report)
        std::cout << "-- Successfully agglomerated topology.\n\n"
                  << "-- Building the fine-grid DeRhamSequence...\n";

    const int uform = nDimensions - 1;
    const int pform = nDimensions;
    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());
    DeRhamSequenceFE* DRSequence_FE;
    mfem::FiniteElementSpace * uspace, * pspace;

    // Set the inverse permeability function
//    VectorFunctionCoefficient kinv(
//        nDimensions, InversePermeabilityFunction::InversePermeability);
    ConstantCoefficient kinv(1.0);
    ConstantCoefficient coeffL2(1.0);
    {
        Timer timer = TimeManager::AddTimer(
                          "DeRhamSequence Construction -- Level 0");
        Timer tot_timer = TimeManager::AddTimer(
                              "DeRhamSequence Construction -- Total");

        sequence[0] = make_shared<DeRhamSequence3D_FE>(
                          topology[0], pmesh.get(), feorder);
        DRSequence_FE = sequence[0]->FemSequence();

        DRSequence_FE->ReplaceMassIntegrator(
            AT_elem, pform, make_unique<MassIntegrator>(coeffL2), false);
        DRSequence_FE->ReplaceMassIntegrator(
            AT_elem, uform, make_unique<VectorFEMassIntegrator>(kinv), true);

        if (print_progress_report)
            std::cout << "-- Building targets...\n";

        mfem::Array<mfem::Coefficient *> L2coeff;
        mfem::Array<mfem::VectorCoefficient *> Hdivcoeff;
        mfem::Array<mfem::VectorCoefficient *> Hcurlcoeff;
        mfem::Array<mfem::Coefficient *> H1coeff;
        fillVectorCoefficientArray(nDimensions, upscaling_order, Hcurlcoeff);
        fillVectorCoefficientArray(nDimensions, upscaling_order, Hdivcoeff);
        fillCoefficientArray(nDimensions, upscaling_order, L2coeff);
        fillCoefficientArray(nDimensions, upscaling_order+1, H1coeff);

        std::vector<unique_ptr<MultiVector>>
            targets(sequence[0]->GetNumberOfForms());
        int jform(0);

        targets[jform] =
            DRSequence_FE->InterpolateScalarTargets(jform, H1coeff);
        ++jform;

        targets[jform] =
            DRSequence_FE->InterpolateVectorTargets(jform, Hcurlcoeff);
        ++jform;

        targets[jform] =
            DRSequence_FE->InterpolateVectorTargets(jform,Hdivcoeff);
        ++jform;

        targets[jform] =
            DRSequence_FE->InterpolateScalarTargets(jform,L2coeff);
        ++jform;

        freeCoeffArray(L2coeff);
        freeCoeffArray(Hdivcoeff);
        freeCoeffArray(Hcurlcoeff);
        freeCoeffArray(H1coeff);

        Array<MultiVector*> targets_in(targets.size());
        for (int ii = 0; ii < targets_in.Size(); ++ii)
            targets_in[ii] = targets[ii].get();

        if (print_progress_report)
            std::cout << "-- Setting targets...\n";

        sequence[0]->SetjformStart(derefinement ? 0 : 2);
        sequence[0]->SetTargets(targets_in);

        uspace = DRSequence_FE->GetFeSpace(uform);
        pspace = DRSequence_FE->GetFeSpace(pform);
    }

    if (print_progress_report)
        std::cout << "-- Built fine grid DeRhamSequence.\n\n"
                  << "-- Coarsening the DeRhamSequence to all levels...\n";

    constexpr double tolSVD = 1e-9;
    for (int i(0); i < nLevels-1; ++i)
    {
        if (print_progress_report)
            std::cout << "  -- Level " << i+1 << "...";

        std::ostringstream timer_name;
        timer_name << "DeRhamSequence Construction -- Level " << i+1;
        Timer timer = TimeManager::AddTimer(timer_name.str());
        Timer tot_timer = TimeManager::GetTimer(
                              "DeRhamSequence Construction -- Total");

        sequence[i]->SetSVDTol(tolSVD);
        sequence[i+1] = sequence[i]->Coarsen();
        //sequence[i]->CheckInvariants();

        if (print_progress_report)
            std::cout << " Done!" << std::endl;
    }

    if (print_progress_report)
        std::cout << "-- Coarsened DeRhamSequence to all levels.\n\n"
                  << "-- Assembling the linear system on level "
                  << start_level << "...\n";


    // Setup extra coefficients
    ConstantCoefficient zero(0.);
    ConstantCoefficient one(1.);
    ConstantCoefficient minus_one(-1.);
    RestrictedCoefficient obs_coeff(one, obs_attr);
    RestrictedCoefficient pinflow_coeff(minus_one, inflow_attr);
    Vector zeros_nDim(nDimensions);
    zeros_nDim = 0.;
    VectorConstantCoefficient zero_vcoeff(zeros_nDim);


    // Essential boundary data
    std::vector<std::unique_ptr<mfem::Vector>> ess_data(nLevels);

    {
        for(int i(0); i < nLevels; ++i)
            ess_data[i] = make_unique<Vector>(
                sequence[i]->GetNumberOfDofs(uform));

        *(ess_data[0]) = 0.0;
        GridFunction ugf;
        ugf.MakeRef(uspace, *(ess_data[0]), 0 );
        ugf.ProjectBdrCoefficientNormal(zero_vcoeff, ess_attr[0]);

        for(int i(0); i < nLevels-1; ++i)
            sequence[i]->GetPi(uform)->GetProjectorMatrix().Mult( *(ess_data[i]), *(ess_data[i+1]));
    }

    auto bform = make_unique<mfem::LinearForm>(uspace);
    bform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(zero_vcoeff));
    bform->AddBoundaryIntegrator(
        new VectorFEBoundaryFluxLFIntegrator(pinflow_coeff));
    bform->Assemble();

    auto qform = make_unique<mfem::LinearForm>(pspace);
    qform->AddDomainIntegrator(new DomainLFIntegrator(zero));
    qform->Assemble();

    unique_ptr<mfem::Vector> rhs_u = std::move(bform);
    unique_ptr<mfem::Vector> rhs_p = std::move(qform);

    // Project rhs down to the starting level.
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

    // Loop over all levels of interest
    auto lib = SolverLibrary::CreateLibrary(
                   master_list->Sublist("Preconditioner Library"));

    //std::vector<std::unordered_map<std::string,unique_ptr<mfem::Vector>>>
    //solutions_by_level(stop_level-start_level+1);

    for (int level = start_level; level <= stop_level; ++level)
    {
        // Project RHS down to this level
        //
        // Do NOT time this because not timing the previous loop
        if (level != start_level)
        {
            auto tmp_u = make_unique<mfem::Vector>(
                             sequence[level]->GetNumberOfDofs(uform) );
            auto tmp_p = make_unique<mfem::Vector>(
                             sequence[level]->GetNumberOfDofs(pform) );
            sequence[level-1]->GetP(uform)->MultTranspose(*rhs_u,*tmp_u);
            sequence[level-1]->GetP(pform)->MultTranspose(*rhs_p,*tmp_p);
            rhs_u = std::move(tmp_u);
            rhs_p = std::move(tmp_p);
        }

        Timer total_time = TimeManager::AddTimer(
                               std::string("Solve Time \"Total\" -- Level ")
                               .append(std::to_string(level)));

        if (print_progress_report)
            std::cout << std::endl
                      << "-- Starting solve phase on level " << level
                      << "...\n";

        // Create the parallel linear system
        const SharingMap& hdiv_dofTrueDof
            = sequence[level]->GetDofHandler(uform)->GetDofTrueDof();
        const SharingMap& l2_dofTrueDof
            = sequence[level]->GetDofHandler(pform)->GetDofTrueDof();

        mfem::Array<int> block_offsets(3);
        block_offsets[0] = 0;
        block_offsets[1] = sequence[level]->GetNumberOfDofs(uform);
        block_offsets[2] = block_offsets[1] +
                           sequence[level]->GetNumberOfDofs(pform);

        mfem::Array<int> true_block_offsets(3);
        true_block_offsets[0] = 0;
        true_block_offsets[1] = hdiv_dofTrueDof.GetTrueLocalSize();
        true_block_offsets[2] =
            true_block_offsets[1] + l2_dofTrueDof.GetTrueLocalSize();

        auto A = make_shared<MfemBlockOperator>(true_block_offsets);
        size_t local_nnz = 0;

        // Serial solution vector
        mfem::BlockVector sol(block_offsets);
        sol = 0.0;

        // Serial RHS vector
        mfem::BlockVector rhs(block_offsets);
        rhs.GetBlock(0).SetData(rhs_u->GetData());
        rhs.GetBlock(1).SetData(rhs_p->GetData());

        // Parallel RHS
        mfem::BlockVector prhs(true_block_offsets);
        {
            if (print_progress_report)
                std::cout << "-- Building operator on level " << level
                          << "...\n";

            // The blocks, managed here
            unique_ptr<HypreParMatrix> pM;
            unique_ptr<HypreParMatrix> pB;
            unique_ptr<HypreParMatrix> pBt;

            {
                auto M = sequence[level]->ComputeMassOperator(uform),
                     W = sequence[level]->ComputeMassOperator(pform);
                auto D = sequence[level]->GetDerivativeOperator(uform);

                auto B = ToUnique(Mult(*W, *D));
                *B *= -1.0;
                auto Bt = ToUnique(Transpose(*B));

                BlockMatrix A(block_offsets);
                A.owns_blocks = 0;
                A.SetBlock(0,0,M.get());
                A.SetBlock(0,1,Bt.get());
                A.SetBlock(1,0,B.get());

                Array<int> ess_dofs(A.Height());
                ess_dofs = 0;
                Array<int> ess_udofs(ess_dofs.GetData(), M->Height());
                sequence[level]->GetDofHandler(uform)->MarkDofsOnSelectedBndr(
                    ess_attr[0], ess_udofs);
                A.EliminateRowCol(ess_dofs, *(ess_data[level]), rhs);

                pM = Assemble(hdiv_dofTrueDof, *M, hdiv_dofTrueDof);
                pB = Assemble(l2_dofTrueDof, *B, hdiv_dofTrueDof);
                pBt = Assemble(hdiv_dofTrueDof, *Bt, l2_dofTrueDof);

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

            // Setup right-hand side
            hdiv_dofTrueDof.Assemble(*rhs_u, prhs.GetBlock(0));
            l2_dofTrueDof.Assemble(*rhs_p, prhs.GetBlock(1));

            if (print_progress_report)
                std::cout << "-- Built operator on level " << level
                          << ".\n"
                          <<"-- Assembled the linear system on level "
                          << level << ".\n\n";

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

            auto prec_factory = lib->GetSolverFactory(solver_type);
            auto solver_state = prec_factory->GetDefaultState();
            solver_state->SetDeRhamSequence(sequence[level]);
            solver_state->SetBoundaryLabels(ess_attr);
            solver_state->SetForms({uform,pform});

            unique_ptr<mfem::Solver> solver;

            // Build the preconditioner
            if (print_progress_report)
                std::cout << "-- Building solver \"" << solver_type << "\" "
                          << "on level " << level << "...\n";

            {
                Timer timer = TimeManager::AddTimer(
                    std::string("Build Solver \"").append(solver_type)
                    .append("\" -- Level ").append(std::to_string(level)));

                solver = prec_factory->BuildSolver(A,*solver_state);
            }

            if (print_progress_report)
                std::cout << "-- Built solver \"" << solver_type << "\".\n";

            auto psol = make_unique<mfem::BlockVector>(true_block_offsets);
            *psol = 0.;

            if (!myid)
                std::cout << '\n' << std::string(50,'*') << '\n'
                          << "*    Solving on level: " << level << '\n'
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
                Timer timer = TimeManager::AddTimer(
                                  std::string("Solve Linear System \"")
                                  .append(solver_type)
                                  .append("\" -- Level ")
                                  .append(std::to_string(level)));
                solver->Mult(prhs,*psol);
            }

            {
                mfem::Vector tmp(A->Height());
                A->Mult(*psol,tmp);
                tmp -= prhs;
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

            if (visualize || save_solution)
            {
                hdiv_dofTrueDof.Distribute(psol->GetBlock(0), sol.GetBlock(0));
                l2_dofTrueDof.Distribute(psol->GetBlock(1), sol.GetBlock(1));
            }

            if (visualize)
            {
                hdiv_dofTrueDof.Distribute(psol->GetBlock(0), sol.GetBlock(0));
                l2_dofTrueDof.Distribute(psol->GetBlock(1), sol.GetBlock(1));

                MultiVector u(sol.GetData(), 1, sol.GetBlock(0).Size() );
                sequence[level]->show(uform, u);
                MultiVector p(sol.GetBlock(1).GetData(), 1, sol.GetBlock(1).Size() );
                sequence[level]->show(pform, p);
            }

            if (save_solution && level == 0)
            {
                // std::unique_ptr<Vector> feCoeff, help;

                // help = make_unique<Vector>();
                // help->SetDataAndSize( coeff.GetData(), coeff.Size() );

                // for(int lev(ilevel); lev > 0; --lev)
                // {
                //     PARELAG_ASSERT(help->Size() == P[lev-1]->Width());
                //     feCoeff = new Vector(P[lev-1]->Height() );
                //     P[lev-1]->Mult(*help, *feCoeff);
                //     delete help;
                //     help = feCoeff;
                // }

                // PARELAG_ASSERT(x.Size()==feCoeff->Size());
                // x = *feCoeff;

                // delete feCoeff;

                std::ostringstream mesh_name;
                mesh_name << save_file_prefix << "mesh."
                          << std::setfill('0') << std::setw(6) << myid;

                std::ofstream mesh_ofs(mesh_name.str().c_str());
                mesh_ofs.precision(8);
                pmesh->Print(mesh_ofs);

                GridFunction gf;
                {
                    gf.MakeRef(uspace, sol.GetBlock(0), 0);
                    std::ostringstream fid_name;
                    fid_name << save_file_prefix << "u_fine"
                             << std::setfill('0') << std::setw(2)
                             << level << "." << std::setw(6) << myid;
                    std::ofstream fid(fid_name.str().c_str());
                    fid.precision(8);
                    gf.Save(fid);
                }
                {
                    gf.MakeRef(pspace, sol.GetBlock(1), 0);
                    std::ostringstream fid_name;
                    fid_name << save_file_prefix << "p_fine"
                             << std::setfill('0') << std::setw(2)
                             << level << "." << std::setw(6) << myid;
                    std::ofstream fid(fid_name.str().c_str());
                    fid.precision(8);
                    gf.Save(fid);
                }
            }

            // Cache the solution
            // solutions_by_level[level - start_level][solver_type] = std::move(psol);
        }
    }

    if (print_time)
        TimeManager::Print(std::cout);

    if (print_progress_report)
        std::cout << "-- Good bye!\n\n";

    return EXIT_SUCCESS;
}
