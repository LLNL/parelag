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

/*

### Example runs:

### very basic
./test_GeometricBoxPartitioner.exe

### slightly more interesting, non-square elements and slightly non-square partitions
./test_GeometricBoxPartitioner.exe --x-elem 12 --y-elem 16 --partitions 4
./test_GeometricBoxPartitioner.exe --x-elem 12 --y-elem 16 --partitions 9

### if you want not a perfect square number of partitions, you're sol
./test_GeometricBoxPartitioner.exe --x-elem 12 --y-elem 12 --partitions 6

### star mesh
./test_GeometricBoxPartitioner.exe --mesh ${MESH_DIR}/star.mesh --partitions 4 --refine 3

### square-disc (triangles) - 9 partitions is a mistake here, but that's not the partitioner's fault
./test_GeometricBoxPartitioner.exe --mesh ${MESH_DIR}/square-disc.mesh --partitions 4 --refine 1
./test_GeometricBoxPartitioner.exe --mesh ${MESH_DIR}/square-disc.mesh --partitions 9 --refine 1

### 3D
./test_GeometricBoxPartitioner.exe --dimension 3 --x-elem 4 --y-elem 4 --z-elem 4 --partitions 8

### 3D, tets, some aspect ratio
./test_GeometricBoxPartitioner.exe --mesh ${MESH_DIR}/beam-tet.mesh --partitions 8 --refine 1
./test_GeometricBoxPartitioner.exe --mesh ${MESH_DIR}/beam-tet.mesh --partitions 64 --refine 2

### potentially empty partition (missing 0 and 12)
./test_GeometricBoxPartitioner.exe --mesh ${MESH_DIR}/star.mesh --partitions 16 --refine 3

### irregular tet mesh
./test_GeometricBoxPartitioner.exe --mesh ~/meshes/cube474.mesh3d --partitions 8 --refine 1

*/

#include "mfem.hpp"

#include "elag.hpp"
#include "partitioning/GeometricBoxPartitioner.hpp"

using namespace mfem;
using namespace parelag;

int main(int argc, char * argv[])
{
    mpi_session sess(argc, argv);

    // parse command line
    OptionsParser args(argc, argv);
    // todo elementtype, aspect ratio for generated mesh
    const char* meshfile_c = "GENERATE_MESH";
    args.AddOption(&meshfile_c, "-m", "--mesh",
                   "MFEM mesh file to load.");
    int dimension = 2;
    args.AddOption(&dimension, "--dimension", "--dimension",
                   "Spatial dimension of generated mesh.");
    int num_partitions = 4;
    args.AddOption(&num_partitions, "--partitions", "--partitions",
                   "Number of partitions to divide the mesh into.");
    int x_elements = 4;
    args.AddOption(&x_elements, "--x-elem", "--x-elem",
                   "Number of elements in x direction for generated mesh.");
    int y_elements = 4;
    args.AddOption(&y_elements, "--y-elem", "--y-elem",
                   "Number of elements in y direction for generated mesh.");
    int z_elements = 4;
    args.AddOption(&z_elements, "--z-elem", "--z-elem",
                   "Number of elements in z direction for generated mesh.");
    int refine = 0;
    args.AddOption(&refine, "--refine", "--refine",
                   "Number of times to refine mesh before partitioning.");
    bool visualization = true;
    args.AddOption(&visualization, "--visualization", "--visualization",
                   "--no-visualization", "--no-visualization",
                   "Visualize the partition.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(std::cout);
        MPI_Finalize();
        return 1;
    }
    PARELAG_ASSERT(args.Good());
    std::string meshfile(meshfile_c);

    // build mesh
    std::unique_ptr<Mesh> mesh;
    if (meshfile == "GENERATE_MESH")
    {
        if (dimension == 2)
            mesh = make_unique<Mesh>(x_elements, y_elements, Element::QUADRILATERAL, true);
        else
            mesh = make_unique<Mesh>(x_elements, y_elements, z_elements,
                                     Element::HEXAHEDRON, true);
    }
    else
    {
        std::ifstream imesh(meshfile.c_str());
        if (!imesh)
        {
            std::cerr << "\nCan not open mesh file: " << meshfile << "\n\n";
            return EXIT_FAILURE;
        }
        mesh = make_unique<Mesh>(imesh, 1, 1);
        imesh.close();
        dimension = mesh->SpaceDimension();
    }
    for (int i=0; i<refine; ++i)
    {
        mesh->UniformRefinement();
    }

    // do partitioning
    GeometricBoxPartitioner gbp;
    Array<int> partitioning;
    gbp.doPartition(*mesh, num_partitions, partitioning);
    {
        std::ofstream out("partitioning.array");
        partitioning.Print(out, 1);
    }

    auto& partition_sizes = const_cast<Array<int>&>(gbp.GetPartitionSizes());
    int pmax = partition_sizes.Max();
    int pmin = partition_sizes.Min();
    int psum = 0;
    for (int i : partition_sizes)
        psum += i;
    double pmean = ((double) psum) / ((double) partition_sizes.Size());
    std::cout << "Created " << partition_sizes.Size() << " partitions." << std::endl;
    std::cout << "  max size: " << pmax << std::endl;
    std::cout << "  min size: " << pmin << std::endl;
    std::cout << "  mean size: " << pmean << std::endl;
    // todo: empty partitions

    L2_FECollection fec(0, dimension);
    FiniteElementSpace fespace(mesh.get(), &fec);
    GridFunction gf(&fespace);
    for (int e=0; e<mesh->GetNE(); ++e)
    {
        gf(e) = partitioning[e];
    }

    // visualize
    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock.precision(8);
        if (dimension == 2)
            sol_sock << "fem2d_gf_data_keys\n";
        else
            sol_sock << "solution\n";
        sol_sock << *mesh << gf;
        if (dimension == 2)
            sol_sock << "Rjm";
    }

    return 0;
}
