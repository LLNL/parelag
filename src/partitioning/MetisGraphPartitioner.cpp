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

#include "MetisGraphPartitioner.hpp"
#include "utilities/elagError.hpp"

namespace parelag
{
using mfem::Array;
using mfem::Vector;
using mfem::Table;
using mfem::SparseMatrix;

MetisGraphPartitioner::MetisGraphPartitioner()
{
    options = new int[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_CONTIG] = 1;
    setDefaultUnbalanceToll();
}

MetisGraphPartitioner::~MetisGraphPartitioner()
{
    delete[] options;
}

void MetisGraphPartitioner::doPartition(const Table & table,
                                        int & num_partitions,
                                        Array<int> & partitioning) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        table.Size() != table.Width(),
        std::invalid_argument,
        "MetisGraphPartitioner::doPartition(): Table is not square!");

    const Array<int> i(const_cast<int *>(table.GetI()), table.Size()+1 );
    const Array<int> j(const_cast<int *>(table.GetJ()), table.Size_of_connections() );

    Array<int> edge_weight(0);
    Array<int> vertex_weight(0);

    doPartition(i, j,
                edge_weight,
                vertex_weight,
                num_partitions,
                partitioning);
}

void MetisGraphPartitioner::doPartition(const SparseMatrix & wtable,
                                        int & num_partitions,
                                        Array<int> & partitioning) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        wtable.Size() != wtable.Width(),
        std::invalid_argument,
        "MetisGraphPartitioner::doPartition(): Table is not square!");

    const auto num_vertices = wtable.Size();
    const auto num_edges = wtable.NumNonZeroElems();

    Array<int> i(wtable.Size()+1);
    Array<int> j(num_edges);
    Array<int> edge_weight(num_edges);

    const int * i_t = wtable.GetI();
    const int * j_t = wtable.GetJ();
    const double * data = wtable.GetData();

    int nnz = 0;
    for(int irow(0); irow < num_vertices; ++irow)
    {
        i[irow] = nnz;
        for( int jcol(i_t[irow]); jcol < i_t[irow+1]; ++jcol)
        {
            if(j_t[jcol] != irow)
            {
                edge_weight[nnz] = static_cast<int>(ceil(fabs(data[jcol])));
                j[nnz] = j_t[jcol];
                nnz++;
            }
        }
    }
    i.Last() = nnz;

    Array<int> vertex_weight( 0 );

    doPartition(i, j, edge_weight, vertex_weight, num_partitions, partitioning);
}

void MetisGraphPartitioner::doPartition(const SparseMatrix & wtable,
                                        const Array<int> & vertex_weight,
                                        int & num_partitions,
                                        Array<int> & partitioning) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        wtable.Size() != wtable.Width(),
        std::invalid_argument,
        "MetisGraphPartitioner::doPartition(): Table is not square!");

    const auto num_vertices = wtable.Size();
    const auto num_edges = wtable.NumNonZeroElems();

    const Array<int> i(const_cast<int *>(wtable.GetI()), num_vertices+1);
    const Array<int> j(const_cast<int *>(wtable.GetJ()), num_edges);
    const double * data = wtable.GetData();
    Array<int> edge_weight(num_edges);

    for(int irow(0); irow < num_vertices; ++irow)
        for( int jcol(i[irow]); jcol < i[irow+1]; ++jcol)
        {
            if(j[jcol] == irow)
                edge_weight[jcol] = 0;
            else
                edge_weight[jcol] = static_cast<int>(ceil(fabs(data[jcol])));
        }

    doPartition(i, j,
                edge_weight,
                vertex_weight,
                num_partitions,
                partitioning);
}

void MetisGraphPartitioner::doPartition(const SparseMatrix & wtable,
                                        const Vector & vertex_weight,
                                        int & num_partitions,
                                        Array<int> & partitioning) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        wtable.Size() != wtable.Width(),
        std::invalid_argument,
        "MetisGraphPartitioner::doPartition(): Table is not square!");

    const auto num_vertices = wtable.Size();
    const auto num_edges = wtable.NumNonZeroElems();

    const Array<int> i(const_cast<int *>(wtable.GetI()), num_vertices+1 );
    const Array<int> j(const_cast<int *>(wtable.GetJ()), num_edges );
    const double * data = wtable.GetData();
    Array<int> edge_weight( num_edges );

    for(int irow(0); irow < num_vertices; ++irow)
        for( int jcol(i[irow]); jcol < i[irow+1]; ++jcol)
        {
            if(j[jcol] == irow)
                edge_weight[jcol] = 0;
            else
                edge_weight[jcol] = static_cast<int>(ceil(fabs(data[jcol])));
        }

    Array<int> vertex_weight_int( vertex_weight.Size() );
    for(int i = 0; i < vertex_weight.Size(); ++i )
        vertex_weight_int[i] = static_cast<int>(ceil(vertex_weight[i])  );

    doPartition(i, j,
                edge_weight,
                vertex_weight_int,
                num_partitions,
                partitioning);
}

void MetisGraphPartitioner::doRecursivePartition(
    const SparseMatrix & wtable,
    const Array<int> & vertex_weight,
    Array<int> & num_elements,
    Array<Array<int> *> & partitioning) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        wtable.Size() != wtable.Width(),
        std::invalid_argument,
        "MetisGraphPartitioner::doPartition(): Table is not square!");

    const auto num_vertices = wtable.Size();
    const auto num_edges = wtable.NumNonZeroElems();

    const Array<int> i(const_cast<int *>(wtable.GetI()), wtable.Size()+1 );
    const Array<int> j(const_cast<int *>(wtable.GetJ()), num_edges );
    Array<int> edge_weight( wtable.NumNonZeroElems() );
    const double * data = wtable.GetData();

    for(int irow(0); irow < num_vertices; ++irow)
        for( int jcol(i[irow]); jcol < i[irow+1]; ++jcol)
        {
            if(j[jcol] == irow)
                edge_weight[jcol] = 0;
            else
                edge_weight[jcol] = static_cast<int>(ceil(fabs(data[jcol])));
        }

    Array<int> * part = new Array<int>(num_vertices);

    int number_of_recursions = num_elements.Size()-1;
    int edgecut(0);
    int k(0);
    partitioning.SetSize(number_of_recursions);

    std::cout<< "Partition in " << num_elements[number_of_recursions-k]
             << std::endl;
    edgecut = doPartition(i, j, edge_weight, vertex_weight, num_elements[number_of_recursions-k], *part);
    partitioning[number_of_recursions-k-1] = part;
    for(k=1; k < number_of_recursions; ++k)
    {
        for(int ivert(0); ivert < num_vertices; ++ivert)
        {
            for(int kpos(i[ivert]); kpos < i[ivert+1]; ++kpos)
            {
                if((*part)[ivert] == (*part)[j[kpos]] && ivert != j[kpos] )
                    edge_weight[kpos] += edgecut;
            }
        }

        part = new Array<int>(num_vertices);
        std::cout << "Partition in " << num_elements[number_of_recursions-k]
                  << std::endl;
        edgecut = doPartition(i, j,
                              edge_weight,
                              vertex_weight,
                              num_elements[number_of_recursions-k],
                              *part);
        partitioning[number_of_recursions-k-1] = part;
        comparePartitioning(num_elements[number_of_recursions+1-k],
                            *(partitioning[number_of_recursions-k]),
                            num_elements[number_of_recursions-k],
                            *(partitioning[number_of_recursions-k-1]));
    }
}

void comparePartitioning(int nCoarsePart,
                         const Array<int> & coarsePart,
                         int & nFinePart,
                         Array<int> & finePart)
{

    Table AEC_elem, AEF_elem, elem_AEF;
    Transpose(coarsePart, AEC_elem);
    Transpose(finePart, AEF_elem);
    Transpose(AEF_elem, elem_AEF);

    Table AEC_AEF;
    Mult(AEC_elem, elem_AEF, AEC_AEF);

    int * iCF = AEC_AEF.GetI();
    int * jCF = AEC_AEF.GetJ();

    for(int ivert(0); ivert < coarsePart.Size(); ++ivert)
    {
        int AEC = coarsePart[ivert];
        int AEF = finePart[ivert];
        for(int kpos(iCF[AEC]); kpos < iCF[AEC+1]; ++kpos)
        {
            if(jCF[kpos] == AEF)
            {
                finePart[ivert] = kpos;
                break;
            }
        }
    }

    std::cout << "The number of fine partitions has increased from "
              << nFinePart << " to " << iCF[nCoarsePart] << std::endl;
    nFinePart = iCF[nCoarsePart];
}

//=======================================================================
// Private funtions
//=======================================================================
int MetisGraphPartitioner::doPartition(const Array<int> & i,
                                       const Array<int> & j,
                                       const Array<int> & edge_weight,
                                       const Array<int> & vertex_weight,
                                       int & num_partitions,
                                       Array<int> & partitioning) const
{
    int num_vertexes = i.Size()-1;
    int num_edges = j.Size();

    partitioning.SetSize(num_vertexes);

    if (num_partitions == 1)
    {
        partitioning = 0;
    }
    else
    {
        int *i_ptr, *j_ptr;
        i_ptr = const_cast<int *>(i.GetData());
        j_ptr = const_cast<int *>(j.GetData());

        int * edge_weight_ptr;
        int * vertex_weight_ptr;

        if(edge_weight.Size() == 0)
            edge_weight_ptr = nullptr;
        else
        {
            PARELAG_TEST_FOR_EXCEPTION(
                edge_weight.Size() != num_edges,
                std::runtime_error,
                "edge_weights is of size " << edge_weight.Size()
                << " It should be" << num_edges );
            edge_weight_ptr = const_cast<int *>(edge_weight.GetData());
        }

        if(vertex_weight.Size() == 0)
            vertex_weight_ptr = nullptr;
        else
        {
            PARELAG_TEST_FOR_EXCEPTION(
                vertex_weight.Size() != num_vertexes,
                std::runtime_error,
                "vertex_weights is of size " << vertex_weight.Size()
                << "It should be" << num_vertexes);
            vertex_weight_ptr = const_cast<int *>(vertex_weight.GetData() );
        }

        int ncon = 1;
        int err;
        int edgecut = 0;

#if 0
        if( flags & REORDER)
        {
            PARELAG_TEST_FOR_EXCEPTION(
                edge_weight_ptr,
                std::runtime_error,
                "we can't sort edges if edge_weight is provided!");

            for (int irow(0); irow < num_vertexes; ++irow)
                qsort(&j_ptr[i_ptr[irow]], i_ptr[irow+1]-i_ptr[irow], sizeof(int), &mfem_less);
        }
#endif

        // This function should be used to partition a graph into a small
        // number of partitions (less than 8).
        if (flags & RECURSIVE)
        {
            options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
            err = METIS_PartGraphRecursive(&num_vertexes,
                                           &ncon,
                                           i_ptr,
                                           j_ptr,
                                           vertex_weight_ptr,
                                           (idx_t *) NULL,
                                           edge_weight_ptr,
                                           &num_partitions,
                                           (real_t *) NULL,
                                           const_cast<float*>(&unbalance_toll),
                                           options,
                                           &edgecut,
                                           partitioning);
            PARELAG_TEST_FOR_EXCEPTION(
                err != METIS_OK,
                std::runtime_error,
                "MetisGraphPartitioner::doPartition(): "
                "error in METIS_PartGraphRecursive!");
        }
        else if(flags & KWAY)
        {
            options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
            err = METIS_PartGraphKway(&num_vertexes,
                                      &ncon,
                                      i_ptr,
                                      j_ptr,
                                      vertex_weight_ptr,
                                      (idx_t *) NULL,
                                      edge_weight_ptr,
                                      &num_partitions,
                                      (real_t *) NULL,
                                      const_cast<float*>(&unbalance_toll),
                                      options,
                                      &edgecut,
                                      partitioning);
            PARELAG_TEST_FOR_EXCEPTION(
                err != METIS_OK,
                std::runtime_error,
                "MetisGraphPartitioner::doPartition(): "
                "error in METIS_PartGraphKway!" );
        }
        else if(flags & TOTALCOMMUNICATION)
        {
            options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
            err = METIS_PartGraphKway(&num_vertexes,
                                      &ncon,
                                      i_ptr,
                                      j_ptr,
                                      vertex_weight_ptr,
                                      (idx_t *) NULL,
                                      edge_weight_ptr,
                                      &num_partitions,
                                      (real_t *) NULL,
                                      const_cast<float*>(&unbalance_toll),
                                      options,
                                      &edgecut,
                                      partitioning);
            PARELAG_TEST_FOR_EXCEPTION(
                err != METIS_OK,
                std::runtime_error,
                "MetisGraphPartitioner::doPartition(): "
                "error in METIS_PartGraphKway!" );
        }
        else
        {
            edgecut = -1;//silence the compiler warning about initialization
            mfem::mfem_error("invalid options");
        }

        if(flags & CHECKPARTITION)
        {
            Table table;
            table.SetIJ(i_ptr,j_ptr, num_vertexes);
            CheckPartitioning(table, num_vertexes, partitioning);
            table.LoseData();
        }

        return edgecut;
    }

    return 0;

}

//==============================================================//
// Helper functions                                             //
//==============================================================//

/* required:  partitioning[i] < num_part */
// If partitioning[i] < 0 skip vertex
void FindPartitioningComponents(const Table &table,
                                const int * const partitioning,
                                Array<int> &component,
                                Array<int> &num_comp)
{
    int i, j, k;
    int num_vertexes, *i_table, *j_table;

    num_vertexes    = table.Size();
    i_table = const_cast<int *>(table.GetI());
    j_table = const_cast<int *>(table.GetJ());

    component.SetSize(num_vertexes);

    Array<int> vertex_stack(num_vertexes);
    int stack_p, stack_top_p, vertex;
    int num_part(count_partitions(partitioning, num_vertexes));

    for (i = 0; i < num_vertexes; i++)
        component[i] = -1;

    num_comp.SetSize(num_part);
    for (i = 0; i < num_part; i++)
        num_comp[i] = 0;

    stack_p = 0;
    stack_top_p = 0;  // points to the first unused element in the stack
    for (vertex = 0; vertex < num_vertexes; vertex++)
    {

        if(partitioning[vertex] < 0)
            continue;

        if (component[vertex] >= 0)
            continue;

        component[vertex] = num_comp[partitioning[vertex]]++;

        vertex_stack[stack_top_p++] = vertex;

        for ( ; stack_p < stack_top_p; stack_p++)
        {
            i = vertex_stack[stack_p];
            if(partitioning[i] < 0)
                continue;

            for (j = i_table[i]; j < i_table[i+1]; j++)
            {
                k = j_table[j];
                if (partitioning[k] == partitioning[i] )
                {
                    if (component[k] < 0)
                    {
                        component[k] = component[i];
                        vertex_stack[stack_top_p++] = k;
                    }
                    PARELAG_TEST_FOR_EXCEPTION(
                        component[k] != component[i],
                        std::runtime_error,
                        "FindPartitioningComponents()" );
                }
            }
        }
    }
}

int CheckPartitioning(const Table & table,
                      const int,
                      const int * const partitioning)
{
    int i, n_empty, n_mcomp;
    Array<int> component, num_comp;

    FindPartitioningComponents(table, partitioning, component, num_comp);

    n_empty = n_mcomp = 0;
    for (i = 0; i < num_comp.Size(); i++)
        if (num_comp[i] == 0)
            n_empty++;
        else if (num_comp[i] > 1)
            n_mcomp++;

    if (n_empty > 0)
    {
        std::cout << "The following partitions are empty :" << std::endl;
        for (i = 0; i < num_comp.Size(); i++)
            if (num_comp[i] == 0)
                std::cout << ' ' << i;
        std::cout << std::endl;
    }
    if (n_mcomp > 0)
    {
        std::cout << "The following partitions are NOT connected :"
                  << std::endl;
        for (i = 0; i < num_comp.Size(); i++)
            if (num_comp[i] > 1)
                std::cout << ' ' << i;
        std::cout << std::endl;
    }

    if (n_empty == 0 && n_mcomp == 0)
        return 0;
    else
        return 1;
}

void build_table_from_partitioning(const int * const partitioning,
                                   int num_vertexes,
                                   int num_parts,
                                   Table & table)
{
    // Set the number of rows and the number of non-zeros
    table.SetDims(num_parts, num_vertexes);

    // Get the row size vector
    int * t_i(table.GetI());
    std::fill(t_i, t_i+num_parts+1, 0);
    int * sizes(t_i+1);

    for(int i(0); i < num_vertexes; ++i)
        ++(sizes[partitioning[i]]);

    // Accumulate the sizes so that we have t_j
    for(int i(0); i < num_parts; ++i)
        t_i[i+1] += t_i[i];

    int * t_j(table.GetJ());

    int * whereToAdd = new int[num_parts];

    std::copy(t_i, t_i+num_parts, whereToAdd);

    const int * it(partitioning);
    for(int i(0); i < num_vertexes; ++i, ++it)
    {
        t_j[whereToAdd[ *it ] ] = i;
        ++whereToAdd[ *it ];
    }

    delete[] whereToAdd;
}

int build_table_from_incomplete_partitioning(
    const int * const partitioning,
    int num_vertexes,
    int num_parts,
    Table & table )
{
    // Set the number of rows and the number of non-zeros
    table.SetDims(num_parts, num_vertexes);

    // Get the row size vector
    int * t_i(table.GetI());
    std::fill(t_i, t_i+num_parts+1, 0);
    int * sizes(t_i+1);

    int irow;

    for(int i(0); i < num_vertexes; ++i)
    {
        irow = partitioning[i];
        if(irow > -1)
            ++(sizes[irow]);
    }

    // Accumulate the sizes so that we have t_j
    for(int i(0); i < num_parts; ++i)
        t_i[i+1] += t_i[i];

    int * t_j(table.GetJ());

    int * whereToAdd = new int[num_parts];

    std::copy(t_i, t_i+num_parts, whereToAdd);

    const int * it(partitioning);
    for(int i(0); i < num_vertexes; ++i, ++it)
    {
        if(*it > -1)
        {
            t_j[whereToAdd[*it]] = i;
            ++whereToAdd[*it];
        }
    }
    delete[] whereToAdd;

    return num_vertexes - table.Size_of_connections();
}

int count_partitions(const int * const partitioning, int num_vertexes)
{
    int num_partitions(-1);
    for (int i = 0; i < num_vertexes; i++)
    {
        if (partitioning[i] > num_partitions)
            num_partitions = partitioning[i];
    }

    return ++num_partitions;
}

void build_partitioning_from_table(int nrows,
                                   int ncols,
                                   const int * table_i,
                                   const int * table_j,
                                   Array<int> & partitioning )
{
    partitioning.SetSize(ncols);
    partitioning = -1;

    for(int i(0); i < nrows; ++i)
    {
        const int * jcol = table_j+table_i[i];
        const int * end = table_j+table_i[i+1];
        for( ; jcol != end; ++jcol)
            partitioning[*jcol] = i;
    }
}

}//namespace parelag
