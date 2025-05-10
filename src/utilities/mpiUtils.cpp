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

#include "mpiUtils.hpp"

#include "utilities/elagError.hpp"

using namespace mfem;

namespace parelag
{

mpi_session::mpi_session(int argc, char** argv)
{
    MPI_Init(&argc,&argv);
#if MFEM_HYPRE_VERSION >= 22900
    HYPRE_Initialize();
    // ParELAG currently does not support GPU
    hypre_HandleMemoryLocation(hypre_handle()) = HYPRE_MEMORY_HOST;
#endif
}

mpi_session::~mpi_session()
{
    MPI_Finalize();
#if MFEM_HYPRE_VERSION >= 22900
    HYPRE_Finalize();
#endif
}

int ParPartialSums_Global(MPI_Comm comm, int& myVal, Array<int>& globalPartialSums)
{
    int ierr(0);

    int n_procs(0);

    MPI_Comm_size(comm, &n_procs);

    globalPartialSums.SetSize(n_procs+1);

    globalPartialSums[0] = 0;

    int * recbf = globalPartialSums.GetData();
    ++recbf;

    ierr = MPI_Allgather(&myVal, 1, MPI_INT, recbf, 1, MPI_INT, comm);
    PARELAG_ASSERT(ierr == MPI_SUCCESS);

    globalPartialSums.PartialSum();

    return globalPartialSums.Last();
}

int ParPartialSums_Distributed(
    MPI_Comm comm, int& myVal, Array<int>& distributedPartialSums)
{
    int n_procs;
    MPI_Comm_size(comm, &n_procs);

    int ierr(0);
    distributedPartialSums.SetSize(3);

    ierr = MPI_Scan(
        &myVal,distributedPartialSums.GetData()+1,1,MPI_INT,MPI_SUM,comm);
    distributedPartialSums[2] = distributedPartialSums[1];
    distributedPartialSums[0] = distributedPartialSums[1] - myVal;

    ierr |= MPI_Bcast(
        distributedPartialSums.GetData()+2,1,MPI_INT,n_procs-1,comm);

    PARELAG_ASSERT(ierr == MPI_SUCCESS );

    return distributedPartialSums.Last();
}

int ParPartialSums_AssumedPartitionCheck(
    MPI_Comm comm, int& myVal, Array<int>& partialSums)
{
    if( HYPRE_AssumedPartitionCheck() )
        return ParPartialSums_Distributed(comm, myVal, partialSums);
    else
        return ParPartialSums_Global(comm, myVal, partialSums);
}

void SerializedOutput(
    MPI_Comm comm, std::ostream& out, const std::string& my_msg)
{
    int n_procs, pid, ierr;
    MPI_Comm_size(comm, &n_procs);
    MPI_Comm_rank(comm, &pid);

    std::stringstream msg;

    if (my_msg.size())
        msg << "%Pid " << pid << "\n" << my_msg << "\n";

    std::string msgstr(msg.str());
    int size = msgstr.size();

    std::vector<int> recvcounts(n_procs);
    ierr = MPI_Gather(&size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, comm);
    PARELAG_ASSERT(ierr == 0);
    std::vector<int> displs(n_procs);
    int dspl = 0;
    for(int i = 0; i < n_procs; ++i)
    {
        displs[i] = dspl;
        dspl += recvcounts[i];
    }

    std::string allmsg;
    allmsg.resize(dspl);
    ierr = MPI_Gatherv(&msgstr[0], size, MPI_CHAR,
                       &allmsg[0], recvcounts.data(),
                       displs.data(), MPI_CHAR, 0, comm);
    PARELAG_ASSERT(ierr == MPI_SUCCESS);

    out << allmsg << std::flush;
}

void RootOutput(MPI_Comm comm,int root,std::ostream& out,const std::string& msg)
{
    int n_procs, pid;
    MPI_Comm_size(comm, &n_procs);
    MPI_Comm_rank(comm, &pid);

    if(pid == root)
    {
        out << "%Pid " << pid << "\n" << msg << std::flush;
    }

    MPI_Barrier(comm);

}

std::string AppendProcessId(
    MPI_Comm comm, const std::string& prefix,
    const std::string& extension, int padding)
{
    int n_procs, pid;
    MPI_Comm_size(comm, &n_procs);
    MPI_Comm_rank(comm, &pid);

    elag_assert(log10(static_cast<double>(n_procs)) < static_cast<double>(padding));

    std::stringstream buff;
    buff << prefix << "." << std::setfill('0') << std::setw(padding)
         << pid << "." << extension;

    return buff.str();
}

}// namespace parelag
