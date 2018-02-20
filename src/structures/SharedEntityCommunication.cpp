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

/**
   \file

   \brief A class to manage shared entity communication

   This particular file contains specific instantiations for data types.
   You need to reimplement each of these routines for each datatype
   you want to communicate.

   Andrew T. Barker
   atb@llnl.gov
   17 July 2015
*/

// for MultiVector (and only for MultiVector)
#include "linalg/dense/ParELAG_MultiVector.hpp"

#include "SharedEntityCommunication.hpp"

namespace parelag
{

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::SetSizeSpecifier()
{
    size_specifier = 2;
}

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::PackSendSizes(
    const mfem::DenseMatrix& mat, int * sizes)
{
    sizes[0] = mat.Height();
    sizes[1] = mat.Width();
}

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::CopyData(
    mfem::DenseMatrix& copyto, const mfem::DenseMatrix& copyfrom)
{
    // todo: should just use operator=?
    copyto.SetSize(copyfrom.Height(), copyfrom.Width());
    memcpy(copyto.Data(), copyfrom.Data(),
           copyfrom.Height() * copyfrom.Width() * sizeof(double));
}

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::SendData(
    const mfem::DenseMatrix& mat, int recipient, int tag, MPI_Request * request)
{
    MPI_Isend(mat.Data(), mat.Height() * mat.Width(), MPI_DOUBLE,
              recipient, tag, comm, request);
}

template <>
void SharedEntityCommunication<mfem::DenseMatrix>::ReceiveData(
    mfem::DenseMatrix& mat, int * sizes, int sender, int tag,
    MPI_Request *request)
{
    const int rows = sizes[0];
    const int columns = sizes[1];
    mat.SetSize(rows,columns);
    MPI_Irecv(mat.Data(),
              rows * columns,
              MPI_DOUBLE,
              sender,
              tag,
              comm,
              request);
}

template class SharedEntityCommunication<mfem::DenseMatrix>;

template <>
void SharedEntityCommunication<mfem::Vector>::SetSizeSpecifier()
{
    size_specifier = 1;
}

template <>
void SharedEntityCommunication<mfem::Vector>::PackSendSizes(
    const mfem::Vector& vec, int * sizes)
{
    sizes[0] = vec.Size();
}

template <>
void SharedEntityCommunication<mfem::Vector>::CopyData(
    mfem::Vector& copyto, const mfem::Vector& copyfrom)
{
    copyto.SetSize(copyfrom.Size());
    copyto = copyfrom;
}

template <>
void SharedEntityCommunication<mfem::Vector>::SendData(
    const mfem::Vector& vec, int recipient, int tag, MPI_Request * request)
{
    MPI_Isend(vec.GetData(), vec.Size(), MPI_DOUBLE,
              recipient, tag, comm, request);
}

template <>
void SharedEntityCommunication<mfem::Vector>::ReceiveData(
    mfem::Vector& vec, int * sizes, int sender, int tag, MPI_Request *request)
{
    const int size = sizes[0];
    vec.SetSize(size);
    MPI_Irecv(vec.GetData(),
              size,
              MPI_DOUBLE,
              sender,
              tag,
              comm,
              request);
}

template class SharedEntityCommunication<mfem::Vector>;

/**
   I do not really like this MultiVector implementation, probably because I do
   not really like the MultiVector in general. Also I think of this class as
   "parelag-independent" (and in fact use nearly the same code in SAAMGe), and
   implementing for MultiVector makes that less true.
*/
template <>
void SharedEntityCommunication<parelag::MultiVector>::SetSizeSpecifier()
{
    size_specifier = 2;
}

template <>
void SharedEntityCommunication<parelag::MultiVector>::PackSendSizes(
    const parelag::MultiVector& vec, int * sizes)
{
    sizes[0] = vec.NumberOfVectors();
    sizes[1] = vec.Size();
}

template <>
void SharedEntityCommunication<parelag::MultiVector>::CopyData(
    parelag::MultiVector& copyto, const parelag::MultiVector& copyfrom)
{
    copyto.SetSizeAndNumberOfVectors(copyfrom.Size(),
                                     copyfrom.NumberOfVectors());
    copyto = copyfrom;
}

template <>
void SharedEntityCommunication<parelag::MultiVector>::SendData(
    const parelag::MultiVector& vec, int recipient, int tag,
    MPI_Request * request)
{
    MPI_Isend(vec.GetData(), vec.Size()*vec.NumberOfVectors(), MPI_DOUBLE,
              recipient, tag, comm, request);
}

template <>
void SharedEntityCommunication<parelag::MultiVector>::ReceiveData(
    parelag::MultiVector& vec, int * sizes, int sender, int tag,
    MPI_Request *request)
{
    const int numberofvectors = sizes[0];
    const int vectorlength = sizes[1];
    vec.SetSizeAndNumberOfVectors(vectorlength, numberofvectors);
    MPI_Irecv(vec.GetData(),
              vectorlength*numberofvectors,
              MPI_DOUBLE,
              sender,
              tag,
              comm,
              request);
}

template class SharedEntityCommunication<parelag::MultiVector>;

}//namespace parelag
