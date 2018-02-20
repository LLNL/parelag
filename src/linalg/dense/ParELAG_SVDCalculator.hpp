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

#ifndef SVDCALCULATOR_HPP_
#define SVDCALCULATOR_HPP_

#include <mfem.hpp>

#include "ParELAG_MultiVector.hpp"

namespace parelag
{
class SVD_Calculator
{
public:

    enum : int { COMPUTE_U = 0x01, COMPUTE_VT = 0x02, SKINNY = 0x04 };

    SVD_Calculator();
    void setFlag(int flag);
    void setFlagOA();
    void setFlagON();
    void AllocateOptimalSize(int maxNRows_, int maxNCols_);

    // A = U S V^T
    void Compute(MultiVector & A, mfem::Vector & SingularValues, MultiVector & U, MultiVector * VT, int flag_);
    // A = U S V^T; U overrides A.
    void ComputeOA(MultiVector & A, mfem::Vector & singularValues, MultiVector & VT);
    // A = U S V^T; U overrides A. Now U is w orthogonal, i.e. U^T diag(w) U = I
    void ComputeOA(mfem::Vector & sqrt_w, MultiVector & A, mfem::Vector & singularValues, MultiVector & VT);
    // A = U S V^T; U overrides A.
    void ComputeON(MultiVector & A, mfem::Vector & singularValues);
    // A = U S V^T; U overrides A. Now U is w orthogonal, i.e. U^T diag(w) U = I
    void ComputeON(mfem::Vector & sqrt_w, MultiVector & A, mfem::Vector & singularValues);
    // A = U S V^T; U overrides A. Now U is W-orthogonal, i.e. U^T W U = I
    void ComputeON(mfem::DenseMatrix & W, MultiVector & A, mfem::Vector & singularValues);

    void ComputeON(mfem::DenseMatrix & A, mfem::Vector & singularValues);

    virtual ~SVD_Calculator() = default;

private:

    int Flag_;

    char   jobu_;
    char   jobvt_;
    int    lwork_;
    int    info_;

    int maxNRows_;
    int maxNCols_;

    std::vector<double> work_;

};
}//namespace parelag
#endif /* SVDCALCULATOR_HPP_ */
