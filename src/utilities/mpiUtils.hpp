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

#ifndef PARELAG_MPIUTILS_HPP_
#define PARELAG_MPIUTILS_HPP_

#include <mfem.hpp>

namespace parelag
{

/// Just a quick-and dirty RAII struct for managing the MPI resource.
/// This will force MPI_Finalize() to be called in case of an uncaught
/// exception, which (a) is good practice and (b) might cause a
/// marginally less-ugly error message to print.
///
/// This object should only be created once and only in a driver.
///
/// This could be much expanded.
struct mpi_session
{
    mpi_session(int argc, char** argv);
    ~mpi_session();
};

/// Given one integer for each processors (myVal), each processor
/// computes the array globalPartialSums of size nproc + 1, such that:
///
/// globalPartialSums[i] = \sum_{iproc = 0}^{iproc < i} myVal_iproc.
///
/// This method contains a MPI_GatherAll operation.
///
/// It is useful to compute row_starts / column_starts in Hypre when
/// HYPRE_AssumedPartitionCheck() is false.
int ParPartialSums_Global(
    MPI_Comm comm, int& myVal, mfem::Array<int>& globalPartialSums);

/// Given one integer for each processors (myVal), each processor
/// computes the array distributedPartialSums of size 3, such that:
///
/// distributedPartialSums[0] = \sum_{iproc = 0}^{iproc < myprocid} myVal_iproc.
/// distributedPartialSums[1] = distributePartialSums[0] + myVal
/// distributedPartialSums[2] = \sum_{iproc = 0}^{iproc < nprocs} myVal_iproc
///
/// This method contains a MPI_Scan operation.
///
/// It is useful to compute row_starts / column_starts in Hypre when
/// HYPRE_AssumedPartitionCheck() is true.
int ParPartialSums_Distributed(
    MPI_Comm comm, int& myVal, mfem::Array<int>& distributedPartialSums);

/// When HYPRE_AssumedPartitionCheck() is true it will call
/// ParPartialSums_Distributed. partialSums will have size 3.
///
/// When HYPRE_AssumedPartitionCheck() is false it will call
/// ParPartialSums_Global. partialSums will have size np+1.
int ParPartialSums_AssumedPartitionCheck(
    MPI_Comm comm, int& myVal, mfem::Array<int>& partialSums);

/// Each process will write his own msg to out in a ordered fashion
/// (according to its own process id)
void SerializedOutput(
    MPI_Comm comm, std::ostream& out, const std::string& msg);

/// Processor root will write its own msg to out
void RootOutput(
    MPI_Comm comm, int root, std::ostream& out, const std::string& msg);


/// Generate the string $prefix.XXXXXX.$extension where XXXXXX is the process id.
std::string AppendProcessId(
    MPI_Comm comm, const std::string& prefix,
    const std::string& extension, int padding=6);


}// namespace parelag
#endif /* PARELAG_MPIUTILS_HPP_ */
