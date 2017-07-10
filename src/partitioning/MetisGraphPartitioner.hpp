/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#ifndef METISGRAPHPARTITIONER_HPP_
#define METISGRAPHPARTITIONER_HPP_



//! Given a connectivity table and a partitioning vector find the number of connected component in each partition
/*!
 * @param table: the connectivity table (num_vertexes by num_vertexes)
 * @param partitioning: the partitioning vector
 * @param component: (OUT) vector of size num_vertexes component[v] = j if vertex v is in the jth connected component of partition k
 * @param num_comp: (OUT) vector of size num_partitions; num_comp[p] = k if partition p has k connected component
 */
void FindPartitioningComponents(const Table &table, const int * const partitioning, Array<int> &component, Array<int> &num_comp);

//! Given a connectivity table and a partitioning vector check if there are not connected or empty partitions
int CheckPartitioning(const Table & table, const int num_vertexes, const int * const partitioning);

//! Given a partitioning vector, build the corresponding table (num_parts by num_vertexes)
/*!
 * @param partitioning: a partitioning vector of size num_vertexes. partitioning[v] = p if vertex v belongs to partition p; (v < num_vertexes, p < num_partitions)
 * @param num_vertexes: number of vertexes in the graph
 * @param num_parts: number of partitions
 * @param table: (OUT) a num_parts by num_vertexes table where (p,v) is an entry if vertex v belongs to partition p
 */
void build_table_from_partitioning(const int * const partitioning, int num_vertexes, int num_parts, Table & table);
int build_table_from_incomplete_partitioning(const int * const partitioning, int num_vertexes, int num_parts, Table & table);
void build_partitioning_from_table(int nrows, int ncols, const int * table_i, const int * table_j, Array<int> & partitiong );

void comparePartitioning(int nCoarsePart, const Array<int> & coarsePart, int & nFinePart, Array<int> & finePart);

int count_partitions(const int * const partitioning, int num_vertexes);

//! @class MetisGraphPartitioner
/**
	@brief Basically wraps Metis, with key routine doPartition(), given a graph and a number of partitions gives you the partitioning vector
*/
class MetisGraphPartitioner
{
public:

	//! Flags
	enum {REORDER = 0x01, KWAY = 0x02, RECURSIVE = 0x04, TOTALCOMMUNICATION = 0x08, CHECKPARTITION = 0x10};

	//! Constructor: initialize default options for the partitioner.
	MetisGraphPartitioner();
	//! Destructor
	virtual ~MetisGraphPartitioner();

	//! Set flags
	void setFlags(int  _flags){flags = _flags;}
	//! Set not default options for metis
	void setOption(const int i, const int val){options[i] = val;}
	//! Allow some imbalance in the size of the partitions
	void setUnbalanceToll(double utoll){ unbalance_toll = utoll; }
	//! Reset the level of imbalance suggested in Metis manual
	void setDefaultUnbalanceToll(){ unbalance_toll = 1.001; }

	//! Partition a graph with num_vertexes in num_partitions
	/*!
	 * @param table: a num_vertexes by num_vertexes table representing the connectivity of the graph.
	 *               table has an entry (i,j) if there is an edge between vertex i and vertex j
	 *
	 * @param num_partitions: number of partitions in which we want to divide the graph
	 *
	 * @param partitioning: vector of size num_vertexes. partitioning[v] = p if vertex v belongs to partition p. (OUT).
	 *                      If the array has not the correct dimension it will be resized
	 */
	void doPartition(const Table & table, int & num_partitions, Array<int> & partitioning);

	//! Partition a graph with weighted edges in num_partitions
	/*!
	 * @param wtable: a num_vertexes by num_vertexes wtable representing the connectivity of the graph.
	 *               table has an entry (i,j) if there is an edge between vertex i and vertex j.
	 *               The weight of the edge is the value of the matrix
	 *
	 * @param num_partitions: number of partitions in which we want to divide the graph
	 *
	 * @param partitioning: vector of size num_vertexes. partitioning[v] = p if vertex v belongs to partition p. (OUT).
	 *                      If the array has not the correct dimension it will be resized
	 */
	void doPartition(const SparseMatrix & wtable, int & num_partitions, Array<int> & partitioning);

	//! Partition a graph with weighted edges and weighted vertices in num_partitions
	/*!
	 * @param wtable: a num_vertexes by num_vertexes wtable representing the connectivity of the graph.
	 *                table has an entry (i,j) if there is an edge between vertex i and vertex j.
	 *                The weight of the edge is the value of the matrix
	 *
	 * @param vertex_weight: weights in the vertices
	 *
	 * @param num_partitions: number of partitions in which we want to divide the graph
	 *
	 * @param partitioning: vector of size num_vertexes. partitioning[v] = p if vertex v belongs to partition p. (OUT).
	 *                      If the array has not the correct dimension it will be resized
	 */
	void doPartition(const SparseMatrix & wtable, const Array<int> & vertex_weight, int & num_partitions, Array<int> & partitioning);
	//! Partition a graph with weighted edges and weighted vertices in num_partitions (double version)
	void doPartition(const SparseMatrix & wtable, const Vector & vertex_weight, int & num_partitions, Array<int> & partitioning);

	void doRecursivePartition(const SparseMatrix & wtable, const Array<int> & vertex_weight, Array< int > & num_partitions, Array < Array<int> *> & partitioning);

	//=====================================================================
	//! Partition a graph with num_vertexes in num_partitions (deprecated!)
	/*!
	 * @param table: a num_vertexes by num_vertexes table representing the connectivity of the graph.
	 *               table has an entry (i,j) if there is an edge between vertex i and vertex j
	 *
	 * @param num_vertexes: number of vertexes in the graph
	 *
	 * @param num_partitions: number of partitions in which we want to divide the graph
	 *
	 * @param partitioning: vector of size num_vertexes. partitioning[v] = p if vertex v belongs to partition p.
	 * (OUT) partitioning must be allocated outside.
	 */
	void doPartition(const Table & table, int num_vertexes, int & num_partitions, int * partitioning);

private:

	int doPartition(const Array<int> & i, const Array<int> & j, const Array<int> & edge_weight, const Array<int> & vertex_weight, int & num_partitions, Array<int> & partitioning);

	int * options;
	int flags;
	real_t unbalance_toll;

};

#endif /* METISGRAPHPARTITIONER_HPP_ */
