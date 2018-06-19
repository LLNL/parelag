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


#ifndef _SHAREDENTITYCOMMUNICATION_HPP
#define _SHAREDENTITYCOMMUNICATION_HPP

#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include <map>

#include <mpi.h>
#include <mfem.hpp>

namespace parelag
{

/**
   \class SharedEntityCommunication
   \brief A class to manage shared entity communication

   The limitation of SharingMap is that it is all based on Hypre matrix
   multiplies, so you cannot do reductions where the operation is something other
   than addition.

   In particular, for the spectral method we want to do reduction where the
   operation is some kind of SVD, which requires something more complicated.

   The complicated logic on the Reduce side is because we are leaving the actual
   reduce operation to the user, so you can think of it more as a "collect"
   operation onto the master, where the user is responsible to do what is
   necessary.

   This is "fairly" generic but not completely, if you want to use for a
   datatype other than mfem::DenseMatrix or mfem::Vector or parelag::MultiVector
   you need to implement: SetSizeSpecifer(), PackSizes(), SendData(),
   ReceiveData(), and CopyData() routines yourself.

   Significant improvements to handling the "tags" argument to honor the
   MPI_TAG_UB constraint are due to Alex Druinsky from Lawrence Berkeley
   (adruinksy@lbl.gov).

   Andrew T. Barker
   atb@llnl.gov
   17 July 2015
*/
template <class T>
class SharedEntityCommunication
{
public:
    /**
       entity_trueentity has more rows than columns, each row has
       exactly one nonzero. The number of nonzeros in a column tells
       you how many processors share the entity, and the partitions they
       are in tell you which processors it is on.

       If sec is built with this constructor, the destructor will
       NOT free entity_proc or entity_master, they belong to caller.
    */
    SharedEntityCommunication(MPI_Comm comm,
                              hypre_ParCSRMatrix * entity_trueentity,
                              mfem::Table * entity_proc,
                              int * entity_master);

    /**
       This constructor attempts to build entity_proc and entity_master
       from the hypre_ParCSRMatrix.

       It may be possible or desirable to have a third constructor that
       takes an mfem::FES, and uses some GroupCommunicator / GroupTopology
       to construct the needed data structures.
    */
    SharedEntityCommunication(MPI_Comm comm,
                              hypre_ParCSRMatrix * entity_trueentity);

    ~SharedEntityCommunication();

    SharedEntityCommunication(SharedEntityCommunication const&) = delete;
    SharedEntityCommunication(SharedEntityCommunication&&) = delete;

    SharedEntityCommunication& operator=(
        SharedEntityCommunication const&) = delete;
    SharedEntityCommunication& operator=(SharedEntityCommunication&&) = delete;

    /**
       Initializes some data structures and posts some Recvs from
       master processes so they can know data sizes
    */
    void ReducePrepare();

    /**
       Sends mat from entity to whichever processor owns the corresponding
       trueentity. Can (and in fact should) be called even if you are the
       owner. Does not do anything (from user perspective) until
       Collect() is called

       This routine copies mat into a buffer, so it can be
       modified or destroyed after you call it.
    */
    void ReduceSend(int entity, const T& mat);

    /**
       Returns a T[number of local entities]
                  [number of processors who share entity, including yourself]
       If processor does not own entity, out[entity] == NULL
       Contains all the T that you sent with ReduceSend()
       Caller is responsible for freeing.
    */
    T** Collect();

    /**
       Does everything to Broadcast DenseMatrix from master to slave.
       data[] should be size num_entities. The array entries where this
       processor is master should be filled with the appropriate matrix,
       all others will be overwritten.
    */
    void Broadcast(T ** data);

    /**
       Broadcast ints from master to slaves
       the idea here is if we know size in advance we can
       cut out a communication step

       XXX: This function requires that the entities in ete_diag and ete_offd be ordered consistently
            with the global true entity ordering on all CPUs. This is the case with ete_diag, so the
            requirement is reduced to ete_col_map being sorted in increasing order.
    */
    void BroadcastFixedSize(int * values, int num_per_entity);

    /**
       returns owner of entity
    */
    int Owner(int entity) {return entity_master[entity];}

    /**
       returns whether this entity is local
    */
    bool OwnedByMe(int entity) {return entity_master[entity] == comm_rank;}

    /**
       puts neighbor processors who share this entity in Array<int> neighbors
    */
    void Neighbors(int entity, mfem::Array<int> &neighbors)
    {entity_proc->GetRow(entity, neighbors);}

    /**
       returns number of processors who share entity, including yourself
    */
    int NumNeighbors(int entity) {return entity_proc->RowSize(entity);}

private:

    /**
       These are just some shared code for the constructors
    */
    void ETEPointers();
    void Initialize();

    /**
       Given entity in local numbering, return the global entity number
       (column number of entity_trueentity) that it corresponds to
    */
    int GetTrueEntity(int entity) const;

    void SetSizeSpecifier();
    void PackSendSizes(const T& mat, int * sizes);
    /**
       this should maybe rely on T's copy constructor?
    */
    void CopyData(T& copyto, const T& copyfrom);
    void SendData(const T& mat,
                  int recipient,
                  int tag,
                  MPI_Request * request);
    void ReceiveData(T &mat,
                     int * sizes,
                     int recipient,
                     int tag,
                     MPI_Request * request);

    void BroadcastSizes(T ** data);
    void BroadcastData(T ** data);

    enum { ENTITY_HEADER_TAG, ENTITY_MESSAGE_TAG, };

    MPI_Comm comm;
    hypre_ParCSRMatrix * entity_trueentity;
    /**
       Rows of entity_proc corresponding to entities OWNED on this
       processor must be filled in completely, ie with all the
       processors that share the entity. If the entity is not owned by
       the processor, there is no need to fill in all the entries (or
       to use the row at all).
    */
    mfem::Table * entity_proc;
    bool owns_entity_proc;
    int * entity_master;
    bool owns_entity_master;

    int comm_size;
    int comm_rank;

    bool preparing_to_reduce;

    hypre_CSRMatrix * ete_diag;
    hypre_CSRMatrix * ete_offd;
    int * ete_diag_I;
    int * ete_diag_J;
    int * ete_col_starts;
    int * ete_offd_I;
    int * ete_offd_J;
    int * ete_colmap;

    hypre_CSRMatrix * ete_diagT;
    hypre_CSRMatrix * ete_offdT;
    int * ete_diagT_I;
    int * ete_diagT_J;
    int * ete_offdT_I;
    int * ete_offdT_J;

    int * entity_slaveid;

    int num_entities;
    int send_counter;
    int num_slave_comms; // where this processor plays role of slave
    int num_master_comms; // where this processor plays role of master
    MPI_Request * header_requests;
    MPI_Request * data_requests;
    int * send_headers;
    int * receive_headers;

    int size_specifier;
    T * reduce_send_buffer;
    T ** reduce_receive_buffer;
};

template <class T>
void SharedEntityCommunication<T>::ETEPointers()
{
    ete_diag = entity_trueentity->diag;
    ete_offd = entity_trueentity->offd;
    ete_diag_I = ete_diag->i;
    ete_diag_J = ete_diag->j;
    ete_col_starts = entity_trueentity->col_starts;
    ete_offd_I = ete_offd->i;
    ete_offd_J = ete_offd->j;
    ete_colmap = entity_trueentity->col_map_offd;

    hypre_CSRMatrixTranspose(ete_diag, &ete_diagT, 0);
    hypre_CSRMatrixTranspose(ete_offd, &ete_offdT, 0);
    ete_diagT_I = ete_diagT->i;
    ete_diagT_J = ete_diagT->j;
    ete_offdT_I = ete_offdT->i;
    ete_offdT_J = ete_offdT->j;
}

template <class T>
void SharedEntityCommunication<T>::Initialize()
{
    preparing_to_reduce = false;

    num_entities = entity_proc->Size();

    entity_slaveid = new int[num_entities];
    num_master_comms = 0;
    num_slave_comms = 0;
    for (int i=0; i<num_entities; ++i)
    {
        if (entity_master[i] == comm_rank)
        {
            int neighbor_row_size = entity_proc->RowSize(i);
            num_master_comms += neighbor_row_size - 1; // -1 for myself
            entity_slaveid[i] = -1;
        }
        else
        {
            entity_slaveid[i] = num_slave_comms++;
        }
    }
    SetSizeSpecifier();
}

template <class T>
SharedEntityCommunication<T>::SharedEntityCommunication(
    MPI_Comm comm,
    hypre_ParCSRMatrix * entity_trueentity,
    mfem::Table * entity_proc,
    int * entity_master)
    : comm(comm),
      entity_trueentity(entity_trueentity),
      entity_proc(entity_proc),
      entity_master(entity_master)
{
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    owns_entity_proc = false;
    owns_entity_master = false;
    ETEPointers();
    Initialize();
}

// this construction might be simpler if we used SharingMap instead of
// hypre_ParCSRMatrix
template <class T>
SharedEntityCommunication<T>::SharedEntityCommunication(
    MPI_Comm comm,
    hypre_ParCSRMatrix * entity_trueentity)
    :
    comm(comm),
    entity_trueentity(entity_trueentity)
{
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    ETEPointers();
    hypre_ParCSRCommPkg * comm_pkg = entity_trueentity->comm_pkg;
    // we need to directly access the comm_pkg, so insure it has been built
    if (!comm_pkg)
    {
        hypre_MatvecCommPkgCreate(entity_trueentity);
        comm_pkg = entity_trueentity->comm_pkg;
    }

    // hypre_ParCSRMatrixPrintIJ(entity_trueentity, 0, 0, "entity_trueentity");

    entity_proc = new mfem::Table;
    int num_entities =
        entity_trueentity->row_starts[1] - entity_trueentity->row_starts[0];
    entity_master = new int[num_entities];
    entity_proc->MakeI(num_entities);
    std::vector<std::pair<int, int> > trueentity_proc;
    for (int send=0; send<comm_pkg->num_sends; ++send)
    {
        int proc = comm_pkg->send_procs[send];
        for (int j=comm_pkg->send_map_starts[send];
             j<comm_pkg->send_map_starts[send+1];
             ++j)
        {
            int trueentity = comm_pkg->send_map_elmts[j];
            trueentity_proc.push_back(std::make_pair(trueentity, proc));
        }
    }
    for (int entity=0; entity<num_entities; ++entity)
    {
        MFEM_ASSERT((ete_offd_I[entity+1] - ete_offd_I[entity] == 1) ||
                    (ete_offd_I[entity+1] - ete_offd_I[entity] == 0),
                    "entity_trueentity has more than one column per row!");
        entity_master[entity] = comm_rank;
        entity_proc->AddAColumnInRow(entity);
        if (ete_offd_I[entity+1] - ete_offd_I[entity] == 0) // local
        {
            int trueentity = ete_diag_J[ete_diag_I[entity]];
            for (unsigned int i=0; i<trueentity_proc.size(); ++i)
            {
                if (trueentity == trueentity_proc[i].first)
                {
                    entity_proc->AddAColumnInRow(entity);
                }
            }
        }
        if (ete_offd_I[entity+1] - ete_offd_I[entity] == 1)
        {
            int col = ete_offd_J[ete_offd_I[entity]];
            for (int recv=0; recv<comm_pkg->num_recvs; ++recv)
            {
                int proc = comm_pkg->recv_procs[recv];
                for (int k=comm_pkg->recv_vec_starts[recv];
                     k<comm_pkg->recv_vec_starts[recv+1];
                     ++k)
                {
                    if (k == col)
                    {
                        // next line is really unnecessary, we never touch the
                        // unowned rows of entity_proc
                        entity_proc->AddAColumnInRow(entity);
                        entity_master[entity] =
                            (proc < entity_master[entity]) ? proc : entity_master[entity];
                    }
                }
            }
        }
    }
    entity_proc->MakeJ();
    for (int entity=0; entity<num_entities; ++entity)
    {
        MFEM_ASSERT((ete_offd_I[entity+1] - ete_offd_I[entity] == 1) ||
                    (ete_offd_I[entity+1] - ete_offd_I[entity] == 0),
                    "entity_trueentity has more than one column per row!");
        entity_proc->AddConnection(entity,comm_rank);
        if (ete_offd_I[entity+1] - ete_offd_I[entity] == 0)
        {
            int trueentity = ete_diag_J[ete_diag_I[entity]];
            for (unsigned int i=0; i<trueentity_proc.size(); ++i)
            {
                if (trueentity == trueentity_proc[i].first)
                {
                    int proc = trueentity_proc[i].second;
                    entity_proc->AddConnection(entity,proc);
                }
            }
        }
        if (ete_offd_I[entity+1] - ete_offd_I[entity] == 1)
        {
            int col = ete_offd_J[ete_offd_I[entity]];
            for (int recv=0; recv<comm_pkg->num_recvs; ++recv)
            {
                int proc = comm_pkg->recv_procs[recv];
                for (int k=comm_pkg->recv_vec_starts[recv];
                     k<comm_pkg->recv_vec_starts[recv+1];
                     ++k)
                {
                    if (k == col)
                    {
                        entity_proc->AddConnection(entity,proc);
                    }
                }
            }
        }
    }
    entity_proc->ShiftUpI();
    entity_proc->Finalize();

    if (false)
    {
        std::stringstream filename;
        filename << "entity_proc." << comm_rank << ".table";
        std::ofstream out(filename.str().c_str());
        entity_proc->Print(out,1000);
    }

    owns_entity_proc = true;
    owns_entity_master = true;
    Initialize();
}

template <class T>
SharedEntityCommunication<T>::~SharedEntityCommunication()
{
    hypre_CSRMatrixDestroy(ete_diagT);
    hypre_CSRMatrixDestroy(ete_offdT);
    delete [] entity_slaveid;
    if (owns_entity_proc)
        delete entity_proc;
    if (owns_entity_master)
        delete [] entity_master;
}

/**
   note well that we assume Hypre is doing assumed_partition
   we also assume that the diag portion of ete has 1 entry per row (fair...)
*/
template <class T>
int SharedEntityCommunication<T>::GetTrueEntity(int entity) const
{
    if (entity_master[entity] == comm_rank)
        return ete_col_starts[0] + ete_diag_J[ete_diag_I[entity]];
    else
        return ete_colmap[ete_offd_J[ete_offd_I[entity]]];
}

/**
   Should these arrays have size num_entities or num_possible_comms?
   the "right" answer is num_possible_comms (which in general should
   be much smaller than num_entities)...  the real "right" answer
   might be some kind of map from entity number to comm number

   But this is complicated by the fact that we want to Send/Receive
   matrices we own, as well...
*/
template <class T>
void SharedEntityCommunication<T>::ReducePrepare()
{
    preparing_to_reduce = true;

    reduce_send_buffer = new T[num_slave_comms];
    reduce_receive_buffer = new T*[num_entities];
    std::memset(reduce_receive_buffer, 0, sizeof(T*) * num_entities);

    send_counter = 0;
    // A header consists of the dimensions of the entity, followed by its
    // true-entity id.
    const int header_length = size_specifier + 1;
    send_headers = new int[header_length * num_slave_comms];
    receive_headers = new int[header_length * num_master_comms];
    header_requests = new MPI_Request[num_master_comms + num_slave_comms]; // receives come first
    data_requests = new MPI_Request[num_slave_comms + num_master_comms]; // sends come first

    int header_receive_counter = 0;
    for (int i=0; i<num_entities; ++i)
    {
        if (entity_master[i] == comm_rank) // I own this entity
        {
            // loop over other processors that have this entity, see how big the object each will give you is
            int neighbor_row_size = entity_proc->RowSize(i);
            reduce_receive_buffer[i] = new T[neighbor_row_size];
            int * neighbor_row = entity_proc->GetRow(i);
            for (int neighbor=0; neighbor<neighbor_row_size; ++neighbor)
            {
                if (neighbor_row[neighbor] != comm_rank)
                {
                    MPI_Irecv(
                        &receive_headers[header_receive_counter*header_length],
                        header_length, MPI_INT, neighbor_row[neighbor],
                        ENTITY_HEADER_TAG, comm,
                        &header_requests[header_receive_counter]);
                    header_receive_counter++;
                }
            }
        }
        else
        {
            MFEM_ASSERT(reduce_receive_buffer[i] == NULL,
                        "reduce_receive_buffer not NULL!");
        }
    }
    MFEM_ASSERT(header_receive_counter == num_master_comms,
                "Not performing the right number of header receives!");
}

template <class T>
void SharedEntityCommunication<T>::BroadcastFixedSize(int * values,
                                                      int n_per_entity)
{
    data_requests = new MPI_Request[num_master_comms + num_slave_comms];
    MPI_Status * data_statuses =
        new MPI_Status[num_master_comms + num_slave_comms];

    // Receive slave entities in true-entity order. This code assumes that
    // ete_col_map is sorted in increasing order.
    // Currently, Hypre's par_csr_matrix.c:GenerateDiagAndOffd() routine seems
    // to generate the colmap in sorted form. Not sure if this is actually
    // guaranteed by the ParCSRMatrix interface
    int receive_counter = 0;
    for (int j = 0; j < ete_offd->num_cols; ++j)
    {
        int entity = ete_offdT_J[j];
        int owner = entity_master[entity];
        MPI_Irecv(
            &(values[entity * n_per_entity]), n_per_entity,
            MPI_INT, owner, ENTITY_MESSAGE_TAG, comm,
            &(data_requests[j]));
        receive_counter++;
    }

    // Send master entities in true-entity order.
    int count_master_entities = ete_col_starts[1] - ete_col_starts[0];
    int send_counter = 0;
    for (int j = 0; j < count_master_entities; ++j)
    {
        int entity = ete_diagT_J[j];
        int neighbor_row_size = entity_proc->RowSize(entity);
        int * neighbor_row = entity_proc->GetRow(entity);
        for (int neighbor=0; neighbor<neighbor_row_size; ++neighbor)
        {
            if (neighbor_row[neighbor] != comm_rank)
            {
                MPI_Isend(
                    &(values[entity * n_per_entity]), n_per_entity,
                    MPI_INT, neighbor_row[neighbor], ENTITY_MESSAGE_TAG, comm,
                    &data_requests[num_slave_comms + send_counter]);
                send_counter++;
            }
        }
    }

    MFEM_ASSERT(send_counter == num_master_comms,
                "Not sending the right amount of data!");
    MFEM_ASSERT(receive_counter == num_slave_comms,
                "Not receiving the right amount of data!");

    MPI_Waitall(num_master_comms + num_slave_comms, data_requests, data_statuses);
    delete [] data_requests;
    delete [] data_statuses;
}

template <class T>
void SharedEntityCommunication<T>::BroadcastSizes(T ** mats)
{
    const int header_length = size_specifier + 1;
    header_requests = new MPI_Request[num_master_comms + num_slave_comms];
    MPI_Status * header_statuses =
        new MPI_Status[num_master_comms + num_slave_comms];
    send_headers = new int[header_length*num_master_comms];
    receive_headers = new int[header_length*num_slave_comms];
    int send_counter = 0;
    int receive_counter = 0;

    for (int j = 0; j < ete_offd->num_cols; ++j)
    {
        int entity = ete_offdT_J[j];
        int owner = entity_master[entity];
        MPI_Irecv(
            &(receive_headers[j * header_length]), header_length,
            MPI_INT, owner, ENTITY_HEADER_TAG, comm,
            &(header_requests[j]));
        receive_counter++;
    }

    int count_master_entities = ete_col_starts[1] - ete_col_starts[0];
    for (int j = 0; j < count_master_entities; ++j)
    {
        int entity = ete_diagT_J[j];
        int neighbor_row_size = entity_proc->RowSize(entity);
        int * neighbor_row = entity_proc->GetRow(entity);
        for (int neighbor=0; neighbor<neighbor_row_size; ++neighbor)
        {
            if (neighbor_row[neighbor] != comm_rank)
            {
                PackSendSizes(
                    *mats[entity],
                    &(send_headers[send_counter * header_length]));
                send_headers[(send_counter + 1) * header_length - 1] = GetTrueEntity(entity);
                MPI_Isend(
                    &(send_headers[send_counter * header_length]),
                    header_length, MPI_INT, neighbor_row[neighbor],
                    ENTITY_HEADER_TAG, comm,
                    &header_requests[num_slave_comms + send_counter]);
                send_counter++;
            }
        }
    }

    MFEM_ASSERT(send_counter == num_master_comms,
                "Not sending the right amount of data!");
    MFEM_ASSERT(receive_counter == num_slave_comms,
                "Not receiving the right amount of data!");
    MPI_Waitall(num_master_comms + num_slave_comms, header_requests,
                header_statuses);
    delete [] header_requests;
    delete [] header_statuses;
    delete [] send_headers;
}

template <class T>
void SharedEntityCommunication<T>::BroadcastData(T ** mats)
{
    const int header_length = size_specifier + 1;
    int send_counter = 0;
    int receive_counter = 0;
    data_requests = new MPI_Request[num_master_comms + num_slave_comms];
    MPI_Status * data_statuses =
       new MPI_Status[num_master_comms + num_slave_comms];

    // Invert the entity to true entity relation to obtain true entity to entity.
    // This is the simplest thing to do and costs O(n log n), where n is the number
    // of entities on the processor that are not owned by the processor. This is the same
    // asymptotic cost as sorting the column map.
    std::map<int, int> te_to_e;
    std::map<int, int>::iterator it;
    for (int j = 0; j < ete_offd->num_cols; ++j)
    {
        const int e = ete_offdT_J[j];
        const int te = GetTrueEntity(e);
        te_to_e.insert(std::pair<int, int>(te, e));
    }

    for (int j = 0; j < ete_offd->num_cols; ++j)
    {
        const int owner = entity_master[ete_offdT_J[j]];
        MFEM_ASSERT(owner != comm_rank, "Ownership mismatch!")
        const int trueentity = receive_headers[(j+1) * header_length - 1];
        it = te_to_e.find(trueentity);
        MFEM_ASSERT(te_to_e.end() != it, "Cannot find entity associated with true entity!");
        const int entity = it->second;
        MFEM_ASSERT(owner == entity_master[entity], "Ownership mismatch!");
        ReceiveData(
            *mats[entity], &(receive_headers[j * header_length]),
            owner, ENTITY_MESSAGE_TAG,
            &data_requests[j]);
        receive_counter++;
    }

    int count_master_entities = ete_col_starts[1] - ete_col_starts[0];
    for (int j = 0; j < count_master_entities; ++j)
    {
        int entity = ete_diagT_J[j];
        int neighbor_row_size = entity_proc->RowSize(entity);
        int * neighbor_row = entity_proc->GetRow(entity);
        for (int neighbor=0; neighbor<neighbor_row_size; ++neighbor)
        {
            if (neighbor_row[neighbor] != comm_rank)
            {
                SendData(
                    *mats[entity],
                    neighbor_row[neighbor], ENTITY_MESSAGE_TAG,
                    &data_requests[num_slave_comms + send_counter]);
                send_counter++;
            }
        }
    }

    delete [] receive_headers;
    MFEM_ASSERT(send_counter == num_master_comms,
                "Not sending the right amount of data!");
    MFEM_ASSERT(receive_counter == num_slave_comms,
                "Not receiving the right amount of data!");
    MPI_Waitall(num_master_comms + num_slave_comms, data_requests, data_statuses);
    delete [] data_requests;
    delete [] data_statuses;
}

template <class T>
void SharedEntityCommunication<T>::ReduceSend(int entity, const T& mat)
{
    MFEM_ASSERT(preparing_to_reduce,"Must call ReducePrepare() first!");

    int owner = entity_master[entity];
    if (owner == comm_rank)
    {
        CopyData(reduce_receive_buffer[entity][0], mat);
    }
    else
    {
        int trueentity = GetTrueEntity(entity);
        int sendid = entity_slaveid[entity];
        MFEM_ASSERT(sendid >= 0, "Master/slave is confused for this entity!");

        const int header_length = size_specifier + 1;
        int *header = &(send_headers[sendid*header_length]);
        int *size = header;
        int *true_id = size + size_specifier;
        PackSendSizes(mat, size);
        *true_id = trueentity;

        MPI_Isend(header, header_length,
                  MPI_INT, owner, ENTITY_HEADER_TAG, comm,
                  &header_requests[num_master_comms + sendid]);

        CopyData(reduce_send_buffer[sendid], mat);
        SendData(reduce_send_buffer[sendid], owner, ENTITY_MESSAGE_TAG,
                 &data_requests[sendid]);
        send_counter++;
    }
}

template <class T>
T ** SharedEntityCommunication<T>::Collect()
{
    MFEM_ASSERT(send_counter == num_slave_comms,
                "Have not called ReduceSend() for every entity!");

    MPI_Status * header_statuses =
        new MPI_Status[num_slave_comms + num_master_comms];
    MPI_Waitall(num_slave_comms + num_master_comms, header_requests,
                header_statuses);
    delete [] header_requests;
    delete [] header_statuses;

    int data_receive_counter = 0;
    std::vector<int> received_entities(num_entities);
    for (int i=0; i<num_entities; ++i)
    {
        int owner = entity_master[i];
        if (owner == comm_rank)
        {
            int neighbor_row_size = entity_proc->RowSize(i);
            int * neighbor_row = entity_proc->GetRow(i);
            for (int neighbor=0; neighbor<neighbor_row_size; ++neighbor)
            {
                if (neighbor_row[neighbor] != comm_rank)
                {
                    const int header_length = size_specifier + 1;
                    int *header =
                        &(receive_headers[header_length*data_receive_counter]);
                    int *size = header;
                    int trueentity = header[size_specifier];
                    int entity = ete_diagT_J[trueentity - ete_col_starts[0]];
                    int row = entity;
                    int column = 1 + received_entities[entity];
                    ReceiveData(
                        reduce_receive_buffer[row][column], size,
                        neighbor_row[neighbor], ENTITY_MESSAGE_TAG,
                        &data_requests[num_slave_comms + data_receive_counter]);
                    received_entities[entity]++;
                    data_receive_counter++;
                }
            }
        }
        else
        {
            MFEM_ASSERT(reduce_receive_buffer[i] == NULL, "reduce_receive_buffer not null!");
        }
    }

    delete [] send_headers;
    delete [] receive_headers;

    MPI_Status * data_statuses =
        new MPI_Status[num_slave_comms + num_master_comms];
    MPI_Waitall(num_slave_comms + num_master_comms, data_requests,
                data_statuses);
    delete [] data_requests;
    delete [] data_statuses;

    delete [] reduce_send_buffer;

    preparing_to_reduce = false;

    return reduce_receive_buffer;
}

template <class T>
void SharedEntityCommunication<T>::Broadcast(T ** data)
{
    MFEM_ASSERT(!preparing_to_reduce, "Cannot interleave Reduce and Broadcast!");

    BroadcastSizes(data);
    BroadcastData(data);
}
}//namespace parelag

#endif
