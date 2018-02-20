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

#ifndef PARELAG_MPIDATATYPES_HPP_
#define PARELAG_MPIDATATYPES_HPP_

#include <mpi.h>

namespace parelag
{
// A function to get the MPI type of a variable
template <typename T> MPI_Datatype GetMPIType(const T& = T{}) noexcept;

#define __PARELAG_ADD_MPI_TYPE(CPPType,MPIType)                         \
    template <>                                                         \
    inline MPI_Datatype GetMPIType<CPPType>(CPPType const&) noexcept    \
    { return MPIType; }


__PARELAG_ADD_MPI_TYPE(char          , MPI_CHAR          )
__PARELAG_ADD_MPI_TYPE(unsigned char , MPI_UNSIGNED_CHAR )
__PARELAG_ADD_MPI_TYPE(int           , MPI_INT           )
__PARELAG_ADD_MPI_TYPE(unsigned int  , MPI_UNSIGNED      )
__PARELAG_ADD_MPI_TYPE(long          , MPI_LONG          )
__PARELAG_ADD_MPI_TYPE(unsigned long , MPI_UNSIGNED_LONG )
__PARELAG_ADD_MPI_TYPE(short         , MPI_SHORT         )
__PARELAG_ADD_MPI_TYPE(unsigned short, MPI_UNSIGNED_SHORT)
__PARELAG_ADD_MPI_TYPE(float         , MPI_FLOAT         )
__PARELAG_ADD_MPI_TYPE(double        , MPI_DOUBLE        )


#ifndef MPI_C_BOOL
inline MPI_Datatype Make_Mpi_Bool_Type()
{
    MPI_Datatype type;
    MPI_Type_contiguous(sizeof(bool),MPI_BYTE,&type);
    MPI_Type_commit(&type);
    return type;
}
#endif /* ifndef MPI_C_BOOL */

template <>
inline MPI_Datatype GetMPIType(const bool&) noexcept
{
#ifdef MPI_C_BOOL
    return MPI_C_BOOL;
#else
    static MPI_Datatype type = Make_Mpi_Bool_Type();
    return type;
#endif /* ifdef MPI_C_BOOL */
}

}// namespace parelag

#endif /* PARELAG_MPIDATATYPES_HPP_ */
