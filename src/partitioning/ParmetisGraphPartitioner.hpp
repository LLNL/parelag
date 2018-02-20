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

#ifndef PARMETISGRAPHPARTITIONER_HPP_
#define PARMETISGRAPHPARTITIONER_HPP_

#include "ParELAG_Config.h"

#ifdef ParELAG_ENABLE_PARMETIS
#include <mfem.hpp>
#include <parmetis.h>

#include "elag_typedefs.hpp"

#define PARMETIS_NUM_OPTIONS 3
#define PARMETIS_OPTION_DEFAULTS 0
#define PARMETIS_OPTION_OUTPUT_INFO 1
#define PARMETIS_OPTION_SEED 2
#endif

namespace parelag
{

class ParmetisGraphPartitioner
#ifdef ParELAG_ENABLE_PARMETIS
{
public:

    //! Constructor: initialize default options for the partitioner.
    ParmetisGraphPartitioner();
    //! Destructor
    virtual ~ParmetisGraphPartitioner();

    //! Set not default options for metis
    void setOption(const int i, const int val){options[i] = val;}
    //! Allow some imbalance in the size of the partitions
    void setUnbalanceTol(double utol){ unbalance_tol = utol; }
    //! Reset the level of imbalance suggested in Metis manual, Umberto likes 1.001
    void setDefaultUnbalanceTol(){ unbalance_tol = 1.05; }

    //! Partition a graph with num_vertexes in num_partitions
    /*!
     * @param table: a num_vertexes by num_vertexes table representing the connectivity of the graph.
     *               table has an entry (i,j) if there is an edge between vertex i and vertex j
     *
     * @param num_partitions: number of partitions in which we want to divide the graph
     *
     * @param partitioning: vector of size num_vertexes. partitioning[v] = p if
    *                      vertex v belongs to partition p. (OUT).
     *                      If the array has not the correct dimension it will be resized
     *
     * returns number of (LOCAL) partitions actually created (TODO: just returns requested)
     */
    int doPartitionEqualWeight(ParallelCSRMatrix & table,
                               int & num_partitions,
                               mfem::Array<int> & partitioning,
                               MPI_Comm comm = MPI_COMM_WORLD);

    //! Partition a graph with weighted edges in num_partitions
    /*!
     * @param wtable: a num_vertexes by num_vertexes wtable representing the connectivity of the graph.
     *               table has an entry (i,j) if there is an edge between vertex i and vertex j.
     *               The weight of the edge is the value of the matrix
     *
     * @param num_partitions: number of partitions in which we want to divide the graph
     *
     * @param partitioning: vector of size num_vertexes. partitioning[v] = p if
    *                      vertex v belongs to partition p. (OUT).
     *                      If the array has not the correct dimension it will be resized
     *
     * returns number of (LOCAL) partitions actually created (TODO: just returns requested)
     */
    int doPartition(ParallelCSRMatrix & wtable,
                    int & num_partitions,
                    mfem::Array<int> & partitioning,
                    MPI_Comm comm = MPI_COMM_WORLD);

    //! Partition a graph with weighted edges and weighted vertices in num_partitions
    //  this is the one that is normally called
    /*!
     * @param wtable: a num_vertexes by num_vertexes wtable representing the connectivity of the graph.
     *                table has an entry (i,j) if there is an edge between vertex i and vertex j.
     *                The weight of the edge is the value of the matrix
     *
     * @param vertex_weight: weights in the vertices
     *
     * @param num_partitions: number of partitions in which we want to divide the graph
     *
     * @param partitioning: vector of size num_vertexes. partitioning[v] = p if
    *                      vertex v belongs to partition p. (OUT).
     *                      If the array has not the correct dimension it will be resized
     *
     * returns number of (LOCAL) partitions actually created (TODO: just returns requested)
     */
    int doPartition(ParallelCSRMatrix & wtable,
                    const mfem::Array<int> & vertex_weight, 
                    int & num_partitions,
                    mfem::Array<int> & partitioning,
                    MPI_Comm comm = MPI_COMM_WORLD);

    int doPartition(ParallelCSRMatrix & table,
                    const mfem::Array<int> & edge_weight,
                    const mfem::Array<int> & vertex_weight, 
                    int & num_partitions,
                    mfem::Array<int> & partitioning,
                    MPI_Comm comm = MPI_COMM_WORLD);

private:

    int doPartition(int *row_starts,
                    const mfem::Array<int> & i,
                    const mfem::Array<int> & j, 
                    const mfem::Array<int> & edge_weight,
                    const mfem::Array<int> & vertex_weight,
                    int & num_partitions, 
                    mfem::Array<int> & partitioning,
                    MPI_Comm comm = MPI_COMM_WORLD);

    int * options;
    real_t unbalance_tol;
}
#endif
;
}//namespace parelag
#endif
