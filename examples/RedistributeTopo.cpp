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
//#include "matred.hpp"

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
    for (int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        std::ostringstream timer_name;
        timer_name << "Mesh Agglomeration -- Level " << ilevel+1;
        Timer inner = TimeManager::AddTimer(timer_name.str());

        Array<int> partitioning(topology[ilevel]->GetB(0).NumRows());
        mfem_partitioner.Partition(topology[ilevel]->GetB(0).NumRows(),
                                   level_nElements[ilevel+1], partitioning);

        topology[ilevel+1] = topology[ilevel]->CoarsenLocalPartitioning(
                    partitioning, 0, 0, nDimensions == 2 ? 0 : 2);
        ShowTopologyAgglomeratedElements(topology[ilevel+1].get(), pmesh.get(), nullptr);
        if (print_progress_report)
            std::cout << "-- Number of agglomerates on level " << ilevel+1 << " is "
                      << topology[ilevel+1]->GetNumberGlobalTrueEntities(elem_t) <<".\n";
    }

    if (print_progress_report)
        std::cout << "-- Successfully agglomerated topology before redistribution.\n";

    const int feorder = 0;
    const int upscalingOrder = 0;
    const int jFormStart = nDimensions-1;

    sequence[0] = make_shared<DeRhamSequence3D_FE>(
                topology[0], pmesh.get(), feorder);

    DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();
    PARELAG_ASSERT(DRSequence_FE);

    ConstantCoefficient coeffL2(1.);
    ConstantCoefficient coeffHdiv(1.);

    sequence[0]->SetjformStart(jFormStart);

    DRSequence_FE->ReplaceMassIntegrator(
                elem_t, 3, make_unique<MassIntegrator>(coeffL2), false);
    DRSequence_FE->ReplaceMassIntegrator(
                elem_t, 2, make_unique<VectorFEMassIntegrator>(coeffHdiv), true);

    // set up coefficients / targets
    sequence[0]->FemSequence()->SetUpscalingTargets(nDimensions, upscalingOrder, 2);

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
    MetisGraphPartitioner metis_partitioner;
    metis_partitioner.setFlags(MetisGraphPartitioner::RECURSIVE);
    for (int ilevel = nLevels-1; ilevel < topology.size()-1; ++ilevel)
    {
        std::ostringstream timer_name;
        timer_name << "Mesh Agglomeration -- Level " << ilevel+1;
        Timer inner = TimeManager::AddTimer(timer_name.str());

        std::vector<int> redistributed_procs(topology[ilevel]->GetB(0).NumRows());
        std::fill_n(redistributed_procs.begin(), redistributed_procs.size(), myid % num_redist_procs);

        num_redist_procs /= 2;

        Redistributor redistributor(*topology[ilevel], redistributed_procs);
        auto redist_topo = redistributor.Redistribute(*topology[ilevel]);
        topology[ilevel+1] = topology[ilevel]->Coarsen(
                 redistributor, redist_topo, metis_partitioner, 2, 0, 0);

        ShowTopologyAgglomeratedElements(topology[ilevel+1].get(), pmesh.get(), nullptr);

        if (print_progress_report)
            std::cout << "-- Number of agglomerates on level " << ilevel+1 << " is "
                      << topology[ilevel+1]->GetNumberGlobalTrueEntities(elem_t) <<".\n";

        auto redist_seq = redistributor.Redistribute(*sequence[ilevel], redist_topo);
        sequence[ilevel+1] = sequence[ilevel]->Coarsen(redist_seq);
    }

    if (myid == 0 && reportTiming)
        std::cout << "Timing ELEM_AGG: Coarsening before redistribution done in " << chrono.RealTime()
                  << " seconds.\n";


    if (print_time)
        TimeManager::Print(std::cout);

    if (print_progress_report)
        std::cout << "-- Good bye!\n\n";

    return EXIT_SUCCESS;
}
