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

#ifndef CARTESIANPARTITIONER_HPP_
#define CARTESIANPARTITIONER_HPP_

#include <mfem.hpp>

#include "utilities/elagError.hpp"

namespace parelag
{
struct LogicalCartesian
{
    int i;
    int j;
    int k;
};

struct LogicalCartesianMaterialId
{
    int i;
    int j;
    int k;
    int materialId;
};

//! @class
/*!
 * @brief Computes IJK indices (and sets e.g. attributes) for a fine grid mesh
 *
 */
class CartesianIJK
{
public:

    //! Ordering in which the elements in a cartesian topology are stored
    enum Ordering {XYZ};

    static void SetupCartesianIJK(const mfem::Mesh & mesh, const mfem::Array<int> & N,
                                  const Ordering & ordering, mfem::Array<LogicalCartesian> & ijk);
    static void SetupCartesianIJKMaterialId(
        const mfem::Mesh & mesh, const mfem::Array<int> & N,
        const Ordering & ordering, mfem::Array<LogicalCartesianMaterialId> & ijk);

private:

    template<class T>
    static void setupCartesianIJK(const mfem::Mesh & mesh, const mfem::Array<int> & N,
                                  const Ordering & ordering, mfem::Array<T> & ijk)
    {
        if(ordering != CartesianIJK::XYZ)
        {
            elag_error_msg(1,"CartesianIJK only works for XYZ ordering");
        }

        int dim = mesh.Dimension();
        elag_assert(N.Size()==dim);
        ijk.SetSize(mesh.GetNE());
        if(dim==1)
            for(int i(0); i < N[0]; ++i)
                ijk[i].i = i;
        else if(dim==2)
        {
            for(int j(0); j < N[1]; ++j)
            {
                for(int i(0); i < N[0]; ++i)
                {
                    ijk[j*N[0]+i].i = i;
                    ijk[j*N[0]+i].j = j;
                }
            }
        }
        else
        {
            for(int k(0); k < N[2]; ++k)
            {
                for(int j(0); j < N[1]; ++j)
                {
                    for(int i(0); i < N[0]; ++i)
                    {
                        ijk[k*N[0]*N[1]+j*N[0]+i].i = i;
                        ijk[k*N[0]*N[1]+j*N[0]+i].j = j;
                        ijk[k*N[0]*N[1]+j*N[0]+i].k = k;
                    }
                }
            }
        }
    }

    CartesianIJK() {}
};

inline bool operator==(const LogicalCartesian & lhs, const LogicalCartesian & rhs)
{
    return lhs.i == rhs.i && lhs.j == rhs.j && lhs.k == rhs.k;
}

inline bool operator==(const LogicalCartesianMaterialId & lhs,
                       const LogicalCartesianMaterialId & rhs)
{
    return lhs.i == rhs.i && lhs.j == rhs.j && lhs.k == rhs.k && lhs.materialId == rhs.materialId;
}

class CoarsenLogicalCartesianOperator
{
public:
    CoarsenLogicalCartesianOperator(mfem::Array<int> & coarseningratio_)
    {
        coarseningratio.SetSize(coarseningratio_.Size());
        coarseningratio_.Copy(coarseningratio);
    }
    inline void operator()(const LogicalCartesian & fine, LogicalCartesian & coarse) const
    {
        coarse.i = fine.i/coarseningratio[0];
        coarse.j = fine.j/coarseningratio[1];
        coarse.k = fine.k/coarseningratio[2];
    }
private:
    mfem::Array<int> coarseningratio;
};

class CoarsenLogicalCartesianOperatorMaterialId
{
public:
    CoarsenLogicalCartesianOperatorMaterialId(mfem::Array<int> & coarseningratio_)
    {
        coarseningratio.SetSize(coarseningratio_.Size());
        coarseningratio_.Copy(coarseningratio);
    }
    inline void operator()(const LogicalCartesianMaterialId & fine,
                           LogicalCartesianMaterialId & coarse) const
    {
        coarse.i = fine.i/coarseningratio[0];
        coarse.j = fine.j/coarseningratio[1];
        coarse.k = fine.k/coarseningratio[2];
        coarse.materialId = fine.materialId;
    }
private:
    mfem::Array<int> coarseningratio;
};
}//namespace parelag
#endif /* CARTESIANPARTITIONER_HPP_ */
