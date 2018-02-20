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

#include "ParmetisGraphPartitioner.hpp"

#ifdef ParELAG_ENABLE_PARMETIS

namespace parelag
{
using namespace mfem;

/*!
  because Parmetis apparently can't tolerate diagonal entries,
  we remove them from mat and put the resulting i, j, data arrays

  we assume all the diagonals are included to begin with

  need rowstarts...
*/
void removeDiagonals( hypre_CSRMatrix* mat,
                      int my_row_start,
                      Array<int> &iout,
                      Array<int> &jout,
                      Array<double> &dataout )
{
    int *i = mat->i;
    int *j = mat->j;
    double *data = mat->data;
    int n = mat->num_rows;
    int nnz = mat->num_nonzeros;

    iout.SetSize(n+1);
    jout.SetSize(nnz - n);
    dataout.SetSize(nnz - n);
    for (int row=0; row<n; ++row)
    {
        // std::cout << "row " << my_row_start + row << std::endl;
        iout[row] = i[row] - row;
        // std::cout << "  iout[" << row << "] = " << iout[row] << std::endl;
        int indexout = iout[row];
        for (int index=i[row]; index<i[row+1]; ++index)
        {
            if (j[index] != my_row_start + row)
            {
                jout[indexout] = j[index];
                dataout[indexout] = data[index];
                indexout++;
            }
        }
    }
    iout[n] = i[n] - n;
}


// According to ATB, the row_starts returned by the underlying
// hypre object is not the global row_starts. Rather,
//
//        my_first_global_row = row_starts[0]
//   one_past_last_global_row = row_starts[1].
//
// However, ParMETIS requires the old format; namely
//
//        my_first_global_row = row_starts[my_rank]
//   one_past_last_global_row = row_starts[my_rank+1]
//      total_num_global_rows = row_starts[comm_size+1]
void getRowStarts(int * row_starts_in,
                  Array<int> &row_starts_out,
                  MPI_Comm comm)
{
    // Set the first rowstart to be 0
    row_starts_out[0] = 0;

    // Because I know the row_start on proc 0 is 0, I actually need
    // the "row_end", which is the second entry in the "row_starts_in"
    // array.
    MPI_Allgather(row_starts_in+1, 1, MPI_INT,
                  row_starts_out+1, 1, MPI_INT,
                  comm);

    // TODO: This should take an MPI_Comm argument in case the matrix
    // is distributed over some other communicator. This doesn't seem
    // common in ParELAG...
}

ParmetisGraphPartitioner::ParmetisGraphPartitioner()
{
    options = new int[PARMETIS_NUM_OPTIONS];
    options[PARMETIS_OPTION_DEFAULTS] = 0;
    options[PARMETIS_OPTION_OUTPUT_INFO] = 0;
    options[PARMETIS_OPTION_SEED] = 0;
    setDefaultUnbalanceTol();
}

ParmetisGraphPartitioner::~ParmetisGraphPartitioner()
{
    delete[] options;
}

int ParmetisGraphPartitioner::doPartitionEqualWeight(ParallelCSRMatrix & table,
                                                     int & num_partitions,
                                                     Array<int> & partitioning,
                                                     MPI_Comm comm)
{
    int myid;
    MPI_Comm_rank(comm,&myid);

    if(table.M() != table.N())
        mfem_error("Table is not square!");

    hypre_ParCSRMatrix* hypretable = table;
    hypre_CSRMatrix* combined = hypre_MergeDiagAndOffd(hypretable);

    Array<int> i;
    Array<int> j;
    Array<double> data;

    removeDiagonals(combined,hypretable->row_starts[0],i,j,data);
    hypre_CSRMatrixDestroy(combined);

    Array<int> edge_weight( 0 );
    Array<int> vertex_weight( 0 );

    return doPartition( hypretable->row_starts,
                        i,
                        j,
                        edge_weight,
                        vertex_weight,
                        num_partitions,
                        partitioning,
                        comm);
}

int ParmetisGraphPartitioner::doPartition(ParallelCSRMatrix & wtable,
                                          int & num_partitions,
                                          Array<int> & partitioning,
                                          MPI_Comm comm)
{
    Array<int> vertex_weight( 0 );

    return doPartition( wtable,
                        vertex_weight,
                        num_partitions,
                        partitioning,
                        comm);
}

int ParmetisGraphPartitioner::doPartition(ParallelCSRMatrix & wtable,
                                          const Array<int> & vertex_weight,
                                          int & num_partitions,
                                          Array<int> & partitioning,
                                          MPI_Comm comm)
{
    int myid;
    MPI_Comm_rank(comm,&myid);

    if(wtable.M() != wtable.N())
        mfem_error("Table is not square!");

    hypre_ParCSRMatrix* hypretable = wtable;
    hypre_CSRMatrix* combined = hypre_MergeDiagAndOffd(hypretable);

    int num_local_vertices(combined->num_rows);

    // const Array<int> i(const_cast<int *>(combined->i),num_local_vertices+1);
    // const Array<int> j(const_cast<int *>(combined->j),num_local_edges);
    Array<int> i;
    Array<int> j;
    Array<double> data;

    removeDiagonals(combined,hypretable->row_starts[0],i,j,data);
    int num_local_edges = combined->num_nonzeros - combined->num_rows;
    hypre_CSRMatrixDestroy(combined);

    Array<int> edge_weight( num_local_edges );

    for(int irow(0); irow < num_local_vertices; ++irow)
        for( int jcol(i[irow]); jcol < i[irow+1]; ++jcol)
        {
            if(j[jcol] == irow)
                edge_weight[jcol] = 0;
            else
                edge_weight[jcol] = static_cast<int>(ceil(fabs(data[jcol])));
        }
    
    return doPartition(hypretable->row_starts,
                       i,
                       j,
                       edge_weight,
                       vertex_weight,
                       num_partitions,
                       partitioning,
                       comm);
}

int ParmetisGraphPartitioner::doPartition(ParallelCSRMatrix & table,
                                          const Array<int> & edge_weight,
                                          const Array<int> & vertex_weight,
                                          int & num_partitions,
                                          Array<int> & partitioning,
                                          MPI_Comm comm)
{
    int myid;
    MPI_Comm_rank(comm,&myid);

    if(table.M() != table.N())
        mfem_error("Table is not square!");

    hypre_ParCSRMatrix* hypretable = table;
    hypre_CSRMatrix* combined = hypre_MergeDiagAndOffd(hypretable);

    // const Array<int> i(const_cast<int *>(combined->i), num_local_vertices+1);
    // const Array<int> j(const_cast<int *>(combined->j), num_local_edges);
    Array<int> i;
    Array<int> j;
    Array<double> data;

    removeDiagonals(combined,hypretable->row_starts[0],i,j,data);
    hypre_CSRMatrixDestroy(combined);

    return doPartition(hypretable->row_starts,
                       i,
                       j,
                       edge_weight,
                       vertex_weight,
                       num_partitions,
                       partitioning,
                       comm);
}


// ---
// private version
// in the end everyone calls this
int ParmetisGraphPartitioner::doPartition(int *row_starts,
                                          const Array<int> & i,
                                          const Array<int> & j,
                                          const Array<int> & edge_weight,
                                          const Array<int> & vertex_weight,
                                          int & num_partitions,
                                          Array<int> & partitioning,
                                          MPI_Comm comm)
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // Need to get the correct row_starts array for ParMETIS
    Array<int> parmetis_row_starts(num_procs+1);
    getRowStarts(row_starts,
                 parmetis_row_starts, comm);

    int num_local_vertexes(i.Size()-1);
    int num_edges(j.Size());

    int target_partitions = num_partitions;
    partitioning.SetSize(num_local_vertexes);

    int edgecut = 0;
    if (target_partitions == 1)
    {
        partitioning = 0;
        edgecut = 0;
    }
    else
    {
        // std::cout << "[" << myid << "] i.Size() = " << i.Size() << std::endl;
        int *i_ptr, *j_ptr;
        i_ptr = const_cast<int *>(i.GetData());
        j_ptr = const_cast<int *>(j.GetData());

        int * edge_weight_ptr;
        int * vertex_weight_ptr;

        if(edge_weight.Size() == 0)
        {
            edge_weight_ptr = NULL;
            // std::cout << "[" << myid << "] edge_weight_ptr = NULL" << std::endl;
        }
        else
        {
            if(edge_weight.Size() == num_edges)
                edge_weight_ptr = const_cast<int *>(edge_weight.GetData() );
            else
            {
                std::cout << "edge_weights is of size " << edge_weight.Size()
                          << "It should be" << num_edges <<"\n";
                throw 1;
            }
        }

        if(vertex_weight.Size() == 0)
        {
            vertex_weight_ptr = NULL;
            // std::cout << "[" << myid << "] vertex_weight_ptr = NULL" << std::endl;
        }
        else
        {
            if(vertex_weight.Size() == num_local_vertexes)
                vertex_weight_ptr = const_cast<int *>(vertex_weight.GetData() );
            else
            {
                std::cerr << "vertex_weights is of size " << vertex_weight.Size()
                          << "It should be" << num_local_vertexes <<"\n";
                throw 2;
            }
        }

        int ncon = 1;
        int err = 0;
        int wgtflag = 0;
        if (vertex_weight.Size() == 0 && edge_weight.Size() == 0)
        {
            wgtflag = 0;
            // std::cout << "[" << myid << "] wgtflag = 0" << std::endl;
        }
        else if (vertex_weight.Size() == 0)
        {
            wgtflag = 1;
        }
        else if (edge_weight.Size() == 0)
        {
            wgtflag = 2;
        }
        else
        {
            wgtflag = 3;
        }
        int numflag = 0;
        real_t *tpwgts = new real_t[target_partitions];
        for (int l=0; l<target_partitions; ++l)
        {
            tpwgts[l] = 1.0/((real_t) target_partitions);
        }

        err = ParMETIS_V3_PartKway( parmetis_row_starts,
                                    i_ptr,
                                    j_ptr,
                                    vertex_weight_ptr,
                                    edge_weight_ptr,
                                    &wgtflag,
                                    &numflag,
                                    &ncon,
                                    &target_partitions,
                                    tpwgts,
                                    &unbalance_tol,
                                    options,
                                    &edgecut,
                                    partitioning,
                                    &comm );
        if (err != METIS_OK)
            mfem_error("error in ParMETIS_V3_PartKway!");
        delete[] tpwgts;
    }

    // TODO: this is a really, really naive way to assign agglomerates
    //       to processors should have some sort of AE_processor table
    //       or something
    return num_partitions / num_procs + (myid < (num_partitions % num_procs));
}
}//namespace parelag
#endif
