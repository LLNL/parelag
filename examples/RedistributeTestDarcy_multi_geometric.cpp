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


#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

#include <mpi.h>
#include <string>
#include <utility>
#include <vector>

#include "elag.hpp"

#include "linalg/dense/ParELAG_MultiVector.hpp"
#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"
#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "utilities/ParELAG_TimeManager.hpp"
#include "utilities/ParELAG_SimpleXMLParameterListReader.hpp"
#include "utilities/MPIDataTypes.hpp"

#include "testing_helpers/Build3DHexMesh.hpp"
#include "testing_helpers/CreateDarcyParameterList.hpp"
#include "utilities/elagError.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;
namespace MPIErrorHandler
{
// subclass so we can specifically catch MPI errors
class Exception : public std::exception
{
  public:
    Exception(std::string const &what) : std::exception(), m_what(what) {}
    virtual ~Exception() throw() {}
    virtual const char *what() const throw()
    {
        return m_what.c_str();
    }

  protected:
    std::string m_what;
};

void convertToException(MPI_Comm *comm, int *err, ...)
{
    throw Exception(std::string("MPI Error."));
}
} // namespace MPIErrorHandler

class ParMeshExtension : public mfem::ParMesh
{
  public:
    ParMeshExtension() = delete;

    explicit ParMeshExtension(const ParMesh &pmesh) : ParMesh(pmesh)
    {
    }

  public:
    void PrepareRepartitioning(Array<int> &partitioning, const int group_offset)
    {
        ExchangeFaceNbrData();
        const int nsharedfaces = this->GetNSharedFaces();
        std::unique_ptr<Table> squad_group{Transpose(group_squad)};
        // group_squad
        const int num_face_nbrs = GetNFaceNeighbors();
        vector<Array<int>> new_face_ranks_groups(num_face_nbrs);
        Array<int> tmp;
        for (int i = 0; i < num_face_nbrs; ++i)
        {
            group_squad.GetRow(i, tmp);
            new_face_ranks_groups[i].SetSize(tmp.Size());
            for (int j = 0; j < tmp.Size(); ++j)
            {
                const int face = this->GetSharedFace(tmp[j]);
                new_face_ranks_groups[i][j] = partitioning[faces_info[face].Elem1No];
            }
            // std::cout << "with rank " << GetFaceNbrRank(i) << " : ";
            // new_face_ranks_groups[i].Print(out, new_face_ranks_groups[i].Size());
        }
        vector<Array<int>> new_face_nbr_ranks_groups(num_face_nbrs);
        {
            MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
            MPI_Request *send_requests = requests;
            MPI_Request *recv_requests = requests + num_face_nbrs;
            MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

            for (int fn = 0; fn < num_face_nbrs; ++fn)
            {
                new_face_nbr_ranks_groups[fn].SetSize(group_squad.RowSize(fn));
                int tag = 234;

                MPI_Isend(new_face_ranks_groups[fn].GetData(), new_face_ranks_groups[fn].Size(), MPI_INT, GetFaceNbrRank(fn), tag, MyComm, &send_requests[fn]);

                MPI_Irecv(new_face_nbr_ranks_groups[fn].GetData(), new_face_nbr_ranks_groups[fn].Size(), MPI_INT, GetFaceNbrRank(fn), tag, MyComm, &recv_requests[fn]);
            }
            MPI_Waitall(num_face_nbrs, recv_requests, statuses);
        }
        // for (int i = 0; i < num_face_nbrs; ++i)
        // {
        //     std::cout << "from rank " << GetFaceNbrRank(i) << " : ";
        //     new_face_nbr_ranks_groups[i].Print(out, new_face_nbr_ranks_groups[i].Size());
        // }


        int partioningsize = partitioning.Size();
        // for (int i = 0; i < num_face_nbrs; ++i)
        //     partitioning.Append(GetFaceNbrRank(i));
        std::map<int,int> tmpmap;

        for (int fn = 0; fn < num_face_nbrs; ++fn)
        {
            group_squad.GetRow(fn, tmp);
            for (auto j = 0; j < tmp.Size(); ++j)
            {
                const int face = this->GetSharedFace(tmp[j]);
                if (tmpmap.count(new_face_nbr_ranks_groups[fn][j]) == 0)
                {
                    tmpmap[new_face_nbr_ranks_groups[fn][j]] = partioningsize++;
                }
                faces_info[face].Elem2No = tmpmap[new_face_nbr_ranks_groups[fn][j]];
            }
        }
        partitioning.SetSize(partioningsize);
        for (auto &p : tmpmap)
        {
            partitioning[p.second] = p.first;
        }
        // for (int i = 0; i < nsharedfaces; ++i)
        // {
        //     const int face = this->GetSharedFace(i);
        //     // auto *vi = this->GetFace(face)->GetVertices();
        //     // this->AddBdrQuad(vi, this->bdr_attributes.Max() + 1 + this->GetMyRank());
        //     const int facegroup = squad_group->GetRow(i)[0];
        //     faces_info[face].Elem2No = partioningsize + facegroup;

        //     std::cout << "rank " << MyRank << " : face " << face << " : FaceNbrRank = " << GetFaceNbrRank(squad_group->GetRow(i)[0]) << std::endl;
        //     // std::cout << "rank " << MyRank << " : face " << face << " : FaceNbrRank = " << GetFaceNbrElementTransformation()(squad_group->GetRow(i)[0]) << std::endl;
        // }
        // this->GetElementToFaceTable();
    }
};

int main (int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Errhandler mpiErrorHandler;
    // Set this up so we always get an exception that will stop TV
    MPI_Comm_create_errhandler(MPIErrorHandler::convertToException,
                               &mpiErrorHandler);
    MPI_Comm_set_errhandler( comm, mpiErrorHandler );
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
    const bool unstructured = prob_list.Get("Unstructured coarsening", false);
    const bool geometric_coarsening = prob_list.Get("Use geometric coarsening", false);
    const bool multiple_copies = prob_list.Get("Distribute multiple copies", false);

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

    // precompute communicators
    std::vector<MPI_Comm> comms(1);
    Array<int> num_redist_procs;
    Array<int> par_partitioning;
    std::unique_ptr<mfem::Mesh> mesh;
    std::vector<SerialRefinementInfo> serial_refinements;

    if (geometric_coarsening && multiple_copies)
    {
        const int ser_ref_levels = prob_list.Get("Serial refinement levels", 0);
        const int par_ref_levels = prob_list.Get("Parallel refinement levels", 0);
        // const int hierachy_levels = prob_list.Get("Hierarchy levels", ser_ref_levels + par_ref_levels + 1);
        const int nLevels = ser_ref_levels + par_ref_levels + 1;
        const int processor_coarsening_factor = prob_list.Get("Processor coarsening factor", 8);
        comms[0] = MPI_COMM_WORLD;
        int worldsize;
        MPI_Comm_size(comms[0], &worldsize);
        Array<int> k_l, mycopies;
        k_l.SetSize(nLevels, 0);
        mycopies.SetSize(1,0);
        num_redist_procs.SetSize(nLevels, worldsize);
        int num_procs = worldsize;
        for (int level = par_ref_levels; level < nLevels - 1; ++level)
        {
            num_redist_procs[level] = num_procs;
            num_procs = std::max(1, num_procs / processor_coarsening_factor);
            if ((num_redist_procs[level] / processor_coarsening_factor) >= 1)
            {
                int num_copies = num_redist_procs[level] / num_procs;
                int myid;
                auto parent_comm = comms[k_l[level]];
                MPI_Comm_rank(parent_comm, &myid);
                int mycopy = myid / num_procs;
                MPI_Comm child_comm;

                MPI_Comm_split(parent_comm, mycopy, myid, &child_comm);
                comms.push_back(std::move(child_comm));
                mycopies.Append(mycopy);
                k_l[level + 1] = k_l[level] + 1;
            }
            else
            {
                k_l[level + 1] = k_l[level];
            }
            num_redist_procs[level + 1] = num_procs;
        }
        if (myid == 0)
        {
            std::cout << "num_redist_procs = ";
            num_redist_procs.Print(mfem::out, num_redist_procs.Size());
            std::cout << "k_l = ";
            k_l.Print(mfem::out, k_l.Size());
        }
        std::vector<std::vector<std::shared_ptr<mfem::ParMesh>>> pmeshes(k_l.Max()+1);
        std::ifstream imesh(meshfile.c_str());
        mesh = make_unique<mfem::Mesh>(imesh, 1, 1);
        Array<int> new_numbering;
        serial_refinements.resize(ser_ref_levels);

        for (int level = num_redist_procs.Size() - 1; level >= 0 ; --level)
        {
            int klp1 = k_l[std::min(level + 1, num_redist_procs.Size() - 1)];
            if (pmeshes[klp1].empty())
                pmeshes[klp1].resize(num_redist_procs.Size());
            if (level == num_redist_procs.Size() - 1)
            {
                int mysize;
                MPI_Comm_size(comms[klp1], &mysize);

                Array<int> partioning(mesh->GeneratePartitioning(mysize), mesh->GetNE());
                auto pmesh = make_shared<mfem::ParMesh>(comms[klp1], *mesh, partioning);

                pmeshes[klp1][level] = std::move(pmesh);
                serial_refinements[num_redist_procs.Size() - 2 - par_ref_levels].num_redist_proc = num_redist_procs[level];
                if (mycopies[klp1] == 0)
                {
                    char fname[256];
                    sprintf(fname, "mesh_L%d_K%d", level, klp1);
                    pmeshes[klp1][level]->Save(fname);
                }
                continue;
            }
            int kl = k_l[level];
            pmeshes[klp1][level] = make_shared<mfem::ParMesh>(*pmeshes[klp1][level+1]);
            pmeshes[klp1][level]->UniformRefinement();
            if (mycopies[klp1] == 0)
            {
                char fname[256];
                sprintf(fname, "mesh_L%d_K%d", level, klp1);
                pmeshes[klp1][level]->Save(fname);
            }
            if (kl != klp1)
            {
                serial_refinements[level - par_ref_levels - 1].num_redist_proc = num_redist_procs[level];
                if (new_numbering.Size())
                    mesh->ReorderElements(new_numbering);
                mesh->UniformRefinement();
                int groupsize;
                int mygroupid;
                MPI_Comm_rank(comms[klp1], &mygroupid);
                MPI_Comm_size(comms[klp1], &groupsize);
                if (pmeshes[kl].empty())
                    pmeshes[kl].resize(num_redist_procs.Size());
                Array<int> partioning, local_ordering;
                {
                    Array<int> coarser_partitioning(pmeshes[klp1][level+1]->GeneratePartitioning(processor_coarsening_factor), pmeshes[klp1][level+1]->GetNE());
                    partioning.SetSize(pmeshes[klp1][level]->GetNE());
                    local_ordering.SetSize(pmeshes[klp1][level]->GetNE());
                    int *partitioning_ptr = partioning.GetData();
                    // int *local_ordering_ptr = local_ordering.GetData();
                    PARELAG_ASSERT_DEBUG(partioning.Size() == 8 * coarser_partitioning.Size());
                    for (int i = 0; i < coarser_partitioning.Size(); i++)
                    {
                        std::fill_n(partitioning_ptr + i * 8, 8, coarser_partitioning[i]);
                        // std::fill_n(local_ordering_ptr + i * 8, 8, aggregates[coarser_partitioning[i]]++);
                    }
                }
                std::for_each(partioning.begin(),partioning.end(), [&groupsize,mygroupid](int &a) -> void { a = (a * groupsize) + mygroupid; });

                Array<int> global_partitioning(pmeshes[klp1][level]->GetGlobalNE());
                global_partitioning = -1;
                new_numbering.SetSize(global_partitioning.Size());
                new_numbering = -1;
                Array<int> redist_ordering(global_partitioning.Size());
                redist_ordering = -1;
                for (int i = 0; i < partioning.Size(); ++i)
                {
                    auto glob_i = static_cast<int>(pmeshes[klp1][level]->GetGlobalElementNum(i));
                    global_partitioning[glob_i] = partioning[i];
                    // redist_ordering[glob_i] = local_ordering[i];
                }
                MPI_Allreduce(MPI_IN_PLACE, global_partitioning.GetData(), global_partitioning.Size(), MPI_INT, MPI_MAX, comms[kl]);
                // MPI_Allreduce(MPI_IN_PLACE, redist_ordering.GetData(), redist_ordering.Size(), MPI_INT, MPI_MAX, comms[kl]);
                if (myid == 0)
                {
                    global_partitioning.Print(out, global_partitioning.Size());
                    std::cout << std::string(40, '=') << std::endl;
                }
                const int nPartitions = global_partitioning.Max() + 1;
                int o = 0;
                for (int p = 0; p < nPartitions; ++p)
                {
                    int m = 0;
                    for (int i = 0; i < global_partitioning.Size(); ++i)
                    {
                        if (global_partitioning[i] == p)
                        {
                            redist_ordering[o] = m++;
                            new_numbering[i] = o++;
                        }
                    }
                }
                if (myid == 0)
                {
                    // redist_ordering.Print(out, redist_ordering.Size());
                    new_numbering.Print(out, new_numbering.Size());
                }
                const int slice_size = new_numbering.Size() / groupsize;
                serial_refinements[level - par_ref_levels].reordering.SetSize(slice_size);
                serial_refinements[level - par_ref_levels].reordering.Assign(new_numbering.GetData() + mygroupid * slice_size);
                // redist_ordering.Print(out, redist_ordering.Size());
                global_partitioning.Copy(par_partitioning);
                pmeshes[kl][level] = make_shared<mfem::ParMesh>(comms[kl], *mesh, global_partitioning);

                serial_refinements[level - par_ref_levels].elem_redist_procs.resize(pmeshes[kl][level]->GetNE());
                serial_refinements[level - par_ref_levels].num_elems = pmeshes[kl][level]->GetNE();
                std::fill(serial_refinements[level - par_ref_levels].elem_redist_procs.begin(), serial_refinements[level - par_ref_levels].elem_redist_procs.end(), mygroupid);

                if (mycopies[kl] == 0)
                {
                    char fname2[256];
                    sprintf(fname2, "mesh_L%d_K%d", level, kl);
                    pmeshes[kl][level]->Save(fname2);
                }
            }
        }
#ifdef TESTING
        auto hdiv_fespace = std::make_unique<mfem::ParFiniteElementSpace>(pmeshes[0][0].get(), new RT_FECollection(0, 3));

        mfem::ParBilinearForm a(hdiv_fespace.get());
        a.AddDomainIntegrator(new DivDivIntegrator);
        a.AddDomainIntegrator(new VectorFEMassIntegrator);
        mfem::ParLinearForm b(hdiv_fespace.get());
        // Vector tmp(3);
        // tmp = 1.;
        // mfem::VectorConstantCoefficient one_vec(tmp);
        auto f_fun = [](const Vector &x, Vector &y) -> void
        {
            y(0) = (1. + M_PI * M_PI) * sin(x(0) * M_PI) * sin(x(1) * M_PI) * sin(x(2) * M_PI);
            y(1) = -M_PI * M_PI * cos(x(0) * M_PI) * cos(x(1) * M_PI) * sin(x(2) * M_PI);
            y(2) = -M_PI * M_PI * cos(x(0) * M_PI) * sin(x(1) * M_PI) * cos(x(2) * M_PI);
        };
        mfem::VectorFunctionCoefficient f_coeff(3, f_fun);
        b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_coeff));

        a.Assemble();
        b.Assemble();

        auto u_fun = [](const Vector &x, Vector &y) -> void
        {
            y(0) = sin(x(0) * M_PI) * sin(x(1) * M_PI) * sin(x(2) * M_PI);
            y(1) = y(2) = 0.;
        };
        mfem::VectorFunctionCoefficient u_coeff(3, u_fun);
        ParGridFunction x(hdiv_fespace.get());
        x = 0.;
        x.ProjectCoefficient(u_coeff);

        Array<int> ess_tdof_list, ess_bdr;
        ess_bdr.SetSize(6, 1);
        hdiv_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
        Vector B, X;
        OperatorHandle A;
        a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

        HypreADS ads(hdiv_fespace.get());
        HyprePCG pcg(hdiv_fespace->GetComm());
        pcg.SetTol(1e-9);
        pcg.SetPreconditioner(ads);
        pcg.SetOperator(*A);
        pcg.Mult(B,X);

        a.RecoverFEMSolution(X, b, x);

        auto err = x.ComputeL2Error(u_coeff);
        if (myid == 0)
        {
            std::cout << "L2-error = " << err << std::endl;
        }

        // x.Save("solution");
#endif
    }



    // return EXIT_SUCCESS;

    int ser_ref_levels;
    shared_ptr<ParMesh> pmesh;
    {
        if (print_progress_report)
            std::cout << "-- Building and refining serial mesh...\n";

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

            // mesh = make_unique<mfem::Mesh>(imesh, 1, 1);
            // {
            //     auto hilbtimer = TimeManager::AddTimer("Mesh : reorder");
            //     Array<int> hilb;
            //     mesh->GetHilbertElementOrdering(hilb);
            //     mesh->ReorderElements(hilb);
            // }
            // mesh->EnsureNCMesh();
            imesh.close();

            if (print_progress_report)
                std::cout << "-- Read mesh \"" << meshfile
                          << "\" successfully.\n";
        }

        ser_ref_levels =
            prob_list.Get("Serial refinement levels", -1);

        // This will do no refs if ser_ref_levels <= 0.
        // for (int l = 0; l < ser_ref_levels; l++)
        // {
        //     mesh->UniformRefinement();
        // }

        // Negative means refine until mesh is big enough to distribute!
        // if (ser_ref_levels < 0)
        // {
        //     ser_ref_levels = 0;
        //     for (; mesh->GetNE() < 6 * num_ranks; ++ser_ref_levels)
        //         mesh->UniformRefinement();
        // }

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

        if (myid == 0)
        {
            for (auto f : serial_refinements)
                std::cout << f.num_redist_proc << endl;
        }

        // FIXME (aschaf 08/22/23) : when using METIS to generate the parallel distribution there are problems with processor boundaries when redistributing back to a single processor
        // pmesh = BuildParallelMesh(comm, *mesh, serial_refinements, prob_list);
        pmesh = make_shared<ParMesh>(comm, *mesh, par_partitioning);

        if (pmesh && print_progress_report)
            std::cout << "-- Built parallel mesh successfully.\n"
                      << std::flush;
    }

    const int nDimensions = pmesh->Dimension();

    const int nGeometricLevels = par_ref_levels+1;
    std::vector<int> num_elements(nGeometricLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        num_elements[par_ref_levels-l] = pmesh->GetNE();
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

    prob_list.Set("Distribute multiple copies", false);

    SequenceHierarchy hierarchy(pmesh, prob_list, print_progress_report);
    hierarchy.SetCoefficient(pform, coeffL2, false);
    hierarchy.SetCoefficient(uform, coeffHdiv, true);
    hierarchy.SetSerialRefinementInfos(serial_refinements);
    hierarchy.Build(std::move(num_elements));
    auto& sequence = hierarchy.GetDeRhamSequences();

    if (myid == 0)
        std::cout << "-- Hierarchy built!" << std::endl;

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
        const std::string solver_type = prob_list.Get("Linear solver","Hybridization");
        auto prec_factory = lib->GetSolverFactory(solver_type);
        const int rescale_iter = prec_list.Sublist(solver_type).Sublist(
                "Solver Parameters").Get<int>("RescaleIteration", -20);

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
            auto only_one = std::make_unique<Vector>(hierarchy.GetDeRhamSequences(0)[0]->GetNumberOfTrueDofs(pform));
            *only_one = 0.;
            if (myid == 0)
            {
                (*only_one)[0] = 1;
                (*only_one)[only_one->Size() - 1] = 2;
            }
            std::unique_ptr<Vector> tmp;
            for (int ilevel = 0; ilevel < start_level; ++ilevel)
            {
                tmp = make_unique<Vector>(hierarchy.GetDeRhamSequences(0)[ilevel + 1]->GetNumTrueDofs(pform));
                hierarchy.ApplyTruePTranspose(ilevel, pform, *only_one, *tmp);
                only_one = std::move(tmp);
            }

            int k_l = hierarchy.GetRedistributionIndex(start_level);
            int level_numgroups = hierarchy.GetNumGlobalCopies(k_l);
            MultiVector u(psol.GetData(), 1, psol.BlockSize(0));
            MultiVector p(psol.GetBlock(1).GetData(), 1, psol.BlockSize(1));

            MultiVector ids(1, psol.BlockSize(1));
            ids = myid;
            MultiVector elemids(1, psol.BlockSize(1));
            int ne = psol.BlockSize(1);
            int start = 0;
            // MPI_Exscan(&ne, &start, 1, MPI_INT, MPI_SUM, hierarchy.GetComm(k_l));
            if (myid == 0)
                start = 0;
            for (int i = 0; i < elemids.Size(); ++i)
                elemids[i] = start++;

            MultiVector onlyone(only_one->GetData(), 1, only_one->Size());

            if (!prob_list.Get("Visualize multiple copies", false))
                level_numgroups = 1;
            for (int groupid(0); groupid < level_numgroups; groupid++)
            {
                // hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, uform, u);
                // hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, pform, p);
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, pform, elemids);
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, pform, ids);
                hierarchy.ShowTrueData(start_level, hierarchy.GetRedistributionIndex(start_level), groupid, pform, onlyone);
            }
        }

        if (print_progress_report)
            std::cout << "-- Solver has exited.\n";
    }

    if (print_progress_report)
        std::cout << "-- Good bye!\n\n";

    return EXIT_SUCCESS;
}
