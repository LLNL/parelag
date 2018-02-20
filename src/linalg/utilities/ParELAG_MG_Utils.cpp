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


#include "ParELAG_MG_Utils.hpp"
#include "ParELAG_MG_Utils_impl.hpp"

#include <random>

namespace parelag
{

namespace mg_utils
{


MPI_Comm GetComm(mfem::Operator const& op)
{
    if (auto tmp = dynamic_cast<mfem::HypreParMatrix const*>(&op))
        return GetComm(*tmp);
    else if (auto tmp = dynamic_cast<MfemBlockOperator const*>(&op))
        return GetComm(*tmp);
    else if (auto tmp = dynamic_cast<mfem::SparseMatrix const*>(&op))
        return GetComm(*tmp);
    else
        return MPI_COMM_NULL;
}


MPI_Comm GetComm(mfem::HypreParMatrix const& op)
{
    return op.GetComm();
}


MPI_Comm GetComm(mfem::SparseMatrix const&)
{
    return MPI_COMM_SELF;
}


MPI_Comm GetComm(MfemBlockOperator const& op)
{
    for (auto row = 0u; row < op.GetNumBlockRows(); ++row)
        for (auto col = 0u; col < op.GetNumBlockCols(); ++row)
            if (not op.IsZeroBlock(row,col))
                return GetComm(op.GetBlock(row,col));
    return MPI_COMM_NULL;
}


bool CheckSymmetric(const Solver& solver, double tol)
{
    mfem::Vector v(solver.Height()), w(solver.Height()),
        x(solver.Height()), y(solver.Height());

    std::mt19937_64 gen(13);
    std::uniform_real_distribution<double> dist(
        0.0,std::nextafter(1.0, std::numeric_limits<double>::max()));

    for (int ii = 0; ii < v.Size(); ++ii)
    {
        v[ii] = dist(gen);
        w[ii] = dist(gen);
    }

    x = 0.;
    y = 0.;

    // x = M^{-1}v
    solver.Mult(v,x);

    // y = M^{-1}w
    solver.Mult(w,y);

    auto wT_Minv_v = x*w;
    auto vT_Minv_w = y*v;

    // Test that w^{T}*M^{-1}*v == v^{T}*M^{-1}*w
    return (std::abs(wT_Minv_v - vT_Minv_w) < tol);
}


/** \brief ETI of BlockedMatrixMult for SparseMatrix with
 *  MfemBlockOperator. */
template std::unique_ptr<MfemBlockOperator>
BlockedMatrixMult<mfem::SparseMatrix>(
    const MfemBlockOperator& A, const MfemBlockOperator& B);

/** \brief ETI of BlockedMatrixMult for HypreParMatrix with
 *  MfemBlockOperator. */
template std::unique_ptr<MfemBlockOperator>
BlockedMatrixMult<mfem::HypreParMatrix>(
    const MfemBlockOperator& A, const MfemBlockOperator& B);

/** \brief ETI of BlockedMatrixMult for SparseMatrix with
 *  mfem::BlockOperator and MfemBlockOperator.
 */
template std::unique_ptr<MfemBlockOperator>
BlockedMatrixMult<mfem::SparseMatrix>(
    const mfem::BlockOperator& A, const MfemBlockOperator& B);

/** \brief ETI of BlockedMatrixMult for HypreParMatrix. */
template std::unique_ptr<MfemBlockOperator>
BlockedMatrixMult<mfem::HypreParMatrix>(
    const mfem::BlockOperator& A, const MfemBlockOperator& B);

/** \brief ETI of BlockedMatrixMult for SparseMatrix. */
template std::unique_ptr<MfemBlockOperator>
BlockedMatrixMult<mfem::SparseMatrix>(
    const MfemBlockOperator& A, const mfem::BlockOperator& B);

/** \brief ETI of BlockedMatrixMult for HypreParMatrix. */
template std::unique_ptr<MfemBlockOperator>
BlockedMatrixMult<mfem::HypreParMatrix>(
    const MfemBlockOperator& A, const mfem::BlockOperator& B);

/** \brief ETI of BlockedMatrixMult for SparseMatrix with
 *  mfem::BlockOperator. */
template std::unique_ptr<MfemBlockOperator>
BlockedMatrixMult<mfem::SparseMatrix>(
    const mfem::BlockOperator& A, const mfem::BlockOperator& B);

/** \brief ETI of BlockedMatrixMult for HypreParMatrix with
 *  mfem::BlockOperator. */
template std::unique_ptr<MfemBlockOperator>
BlockedMatrixMult<mfem::HypreParMatrix>(
    const mfem::BlockOperator& A, const mfem::BlockOperator& B);

}// namespace mg_utils
}// namespace parelag
