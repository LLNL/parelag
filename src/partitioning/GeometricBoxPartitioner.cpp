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

#include "GeometricBoxPartitioner.hpp"

namespace parelag
{
using namespace mfem;

void GeometricBoxPartitioner::doPartition(mfem::Mesh & mesh,
                                          int & num_partitions,
                                          mfem::Array<int> & partitioning)
{
    const int dimension = mesh.SpaceDimension();
    Vector bmin, bmax;
    mesh.GetBoundingBox(bmin, bmax);
    double volume = 1.0;
    for (int i=0; i<dimension; ++i)
        volume *= bmax(i) - bmin(i);
    double target_volume = volume / ((double) num_partitions);
    double target_radius = std::pow(target_volume, 1.0/dimension);
    int num_partitions_dir[3];
    double part_radius_dir[3];
    int actual_num_partitions = 1;
    for (int i=0; i<dimension; ++i)
    {
        num_partitions_dir[i] =
            ((bmax(i) - bmin(i)) / target_radius) + 0.5; // implicit floor, 0.5 makes it a round
        actual_num_partitions *= num_partitions_dir[i];
        part_radius_dir[i] = (bmax(i) - bmin(i)) / ((double) num_partitions_dir[i]);
    }

    partition_sizes_.SetSize(actual_num_partitions);
    partition_sizes_ = 0;
    partitioning.SetSize(mesh.GetNE());
    for (int e=0; e<mesh.GetNE(); ++e)
    {
        int which_partition_dir[3];
        Array<int> vertices;
        mesh.GetElementVertices(e, vertices);
        double coords[3];
        int num_verts = vertices.Size();
        for (int i=0; i<dimension; ++i)
        {
            coords[i] = 0.0;
            for (int v=0; v<num_verts; ++v)
            {
                coords[i] += mesh.GetVertex(vertices[v])[i];
            }
            coords[i] /= ((double) num_verts);
        }
        for (int i=0; i<dimension; ++i)
        {
            which_partition_dir[i] = (coords[i] - bmin(i)) / part_radius_dir[i];
        }
        int partition = which_partition_dir[0];
        if (dimension > 1)
        {
            partition += num_partitions_dir[0]*which_partition_dir[1];
        }
        if (dimension > 2)
        {
            partition += num_partitions_dir[0]*num_partitions_dir[1]*which_partition_dir[2];
        }
        partitioning[e] = partition;
        partition_sizes_[partition]++;
    }

    // could check for disconnected partitions, but Topology::connectedComponents takes
    // care of them later on
}

}
