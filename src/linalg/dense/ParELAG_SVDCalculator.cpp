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

#include "ParELAG_SVDCalculator.hpp"

#include "ParELAG_Eigensolver.hpp"
#include "utilities/elagError.hpp"

using namespace mfem;

extern "C"
{
    void dgesvd_(const char* jobu, const char* jobvt, const int* m,
                 const int* n, double* a, const int* lda, double* s,
                 double* u, const int* ldu, double* vt, const int* ldvt,
                 double* work, const int* lwork, int* info);
}

namespace parelag
{

SVD_Calculator::SVD_Calculator():
    Flag_(0),
    jobu_('N'),
    jobvt_('N'),
    lwork_(-1),
    info_(0),
    maxNRows_(-1),
    maxNCols_(-1)
{
}


void SVD_Calculator::setFlag(int flag)
{
    Flag_ = flag;

    char job = 'A';
    if(Flag_ & SKINNY)
        job = 'S';
    else
        job = 'A';

    if(Flag_ & COMPUTE_U)
        jobu_ = job;
    else
        jobu_ = 'N';

    if(Flag_ & COMPUTE_VT)
        jobvt_ = 'S';
    else
        jobvt_ = 'N';

}

void SVD_Calculator::setFlagOA()
{
    Flag_ = COMPUTE_U | COMPUTE_VT | SKINNY;
    jobu_ = 'O';
    jobvt_ = 'A';
}

void SVD_Calculator::setFlagON()
{
    Flag_ = COMPUTE_U |  SKINNY;
    jobu_ = 'O';
    jobvt_ = 'N';
}

void SVD_Calculator::AllocateOptimalSize(int maxNRows, int maxNCols)
{
    maxNRows_ = std::max(maxNRows, 1);
    maxNCols_ = maxNCols;

    std::vector<double> tmp(maxNRows_*maxNCols_,0.);
    double * s = tmp.data();
    double * u = tmp.data();
    double * vt = tmp.data();

    lwork_ = -1;
    double qwork = 0.0;
    dgesvd_(&jobu_, &jobvt_, &maxNRows_, &maxNCols_, tmp.data(), &maxNRows_,
            s, u, &maxNRows_, vt, &maxNCols_, &qwork, &lwork_, &info_);

    lwork_ = (int) qwork;
    std::vector<double>(lwork_).swap(work_);
}

void SVD_Calculator::Compute(
    MultiVector & A, Vector & singularValues, MultiVector & U,
    MultiVector * VT, int flag)
{
    if(VT == nullptr)
        flag &= ~COMPUTE_VT;

    int nrows = A.Size();
    int ncols = A.NumberOfVectors();

    if(Flag_ != flag || nrows > maxNRows_ || ncols > maxNCols_)
    {
        setFlag(flag);
        AllocateOptimalSize(maxNRows_,maxNCols_);
    }

    singularValues.SetSize(ncols);

    if(Flag_ & SKINNY)
        U.SetSizeAndNumberOfVectors(nrows, ncols);
    else
        U.SetSizeAndNumberOfVectors(nrows, nrows);

    int ldA = A.LeadingDimension();
    int ldU = U.LeadingDimension();
    // int ldV = 0;

    double * vt = nullptr;
    if (Flag_ & COMPUTE_VT)
    {
        VT->SetSizeAndNumberOfVectors(ncols, ncols);
        vt = VT->GetData();
        // ldV = VT->LeadingDimension();
    }

    dgesvd_(&jobu_, &jobvt_, &nrows, &ncols, A.GetData(), &ldA,
            singularValues.GetData(), U.GetData(), &ldU, vt, &ncols,
            work_.data(), &lwork_, &info_);

    PARELAG_TEST_FOR_EXCEPTION(
        info_,
        std::runtime_error,
        "SVD_Calculator::Compute(...): "
        "DenseMatrix::SingularValues : info_ = " << info_);
}

void SVD_Calculator::ComputeOA(
    MultiVector & A, Vector & singularValues, MultiVector & VT)
{

    const int flag = COMPUTE_U | COMPUTE_VT | SKINNY;

    const int nrows = A.Size();
    const int ncols = A.NumberOfVectors();

    if(flag != Flag_ || nrows > maxNRows_ || ncols > maxNCols_)
    {
        setFlagOA();
        AllocateOptimalSize(nrows,ncols);
    }

    singularValues.SetSize(ncols);

    const int ldA = A.LeadingDimension();
    const int ldV = VT.LeadingDimension();

    double * u = nullptr;
    double * vt = VT.GetData();

    PARELAG_TEST_FOR_EXCEPTION(
        VT.Size() != ncols || VT.NumberOfVectors() != ncols,
        std::runtime_error,
        "SVD_Calculator::ComputeOA(...): Dimensions of VT are not correct!");

    dgesvd_(&jobu_, &jobvt_, &nrows, &ncols, A.GetData(), &ldA,
            singularValues.GetData(), u, &ldA, vt, &ldV, work_.data(),
            &lwork_, &info_);

    PARELAG_TEST_FOR_EXCEPTION(
        info_,
        std::runtime_error,
        "SVD_Calculator::ComputeOA(...): "
        "DenseMatrix::SingularValues : info_ = " << info_ << ".");
}

void SVD_Calculator::ComputeOA(
    Vector & sqrt_w, MultiVector & A, Vector & singularValues, MultiVector & VT)
{
    A.Scale(sqrt_w);
    ComputeOA(A, singularValues, VT);
    A.InverseScale(sqrt_w);
}

void SVD_Calculator::ComputeON(MultiVector & A, Vector & singularValues)
{

    const int flag = COMPUTE_U | SKINNY;

    const int nrows = A.Size();
    const int ncols = A.NumberOfVectors();

    if(flag != Flag_ || nrows > maxNRows_ || ncols > maxNCols_)
    {
        setFlagON();
        AllocateOptimalSize(nrows,ncols);
    }

    const int nSingValues = (nrows < ncols) ? nrows : ncols;
    singularValues.SetSize(nSingValues);

    const int ldA = std::max(A.LeadingDimension(), 1);

    double * u = nullptr;
    double * vt = nullptr;

    dgesvd_(&jobu_, &jobvt_, &nrows, &ncols, A.GetData(), &ldA,
            singularValues.GetData(), u, &ldA, vt, &ldA, work_.data(),
            &lwork_, &info_);

    PARELAG_TEST_FOR_EXCEPTION(
        info_,
        std::runtime_error,
        "SVD_Calculator::ComputeON(...): "
        "DenseMatrix::SingularValues : info_ = " << info_ << ".");

#ifdef ELAG_DEBUG
    {
        double val = singularValues(0)+1;
        for(double * it = singularValues.GetData();
            it != singularValues.GetData()+ nSingValues; ++it )
        {
            PARELAG_TEST_FOR_EXCEPTION(
                *it > val,
                std::runtime_error,
                "SVD_Calculator::ComputeON(...): "
                "Singular Values are not sorted :(");
            val = *it;
        }
    }
#endif

}

void SVD_Calculator::ComputeON(DenseMatrix & A, Vector & singularValues)
{
    MultiVector tmp(A.Data(), A.Width(), A.Height());
    ComputeON(tmp, singularValues);
}

void SVD_Calculator::ComputeON(
    Vector & sqrt_w, MultiVector & A, Vector & singularValues)
{
    // sqrt_w.CheckFinite(); A.CheckFinite();
    A.Scale(sqrt_w);
    ComputeON(A, singularValues);
    A.InverseScale(sqrt_w);
    // A.CheckFinite();
}

void SVD_Calculator::ComputeON(
    DenseMatrix & W, MultiVector & A, Vector & singularValues)
{
    elag_assert(W.Height() == W.Width() );
    SymEigensolver eigs;
    eigs.SetOverwrite(false);

    const int n = W.Height();
    std::vector<double> evals(n);
    DenseMatrix evects(n);
    eigs.ComputeAll(W, evals, evects);

    std::for_each(evals.begin(),evals.end(),[](double& a){a = std::sqrt(a);});

    DenseMatrix X(n);
    mfem::Vector D_tmp(evals.data(),evals.size());
    MultADAt(evects, D_tmp, X);

    MultiVector XA( A.NumberOfVectors(), A.Size() );
    Mult(X,A, XA);

    ComputeON(XA, singularValues);

    std::for_each(evals.begin(), evals.end(), [](double& a){a = 1.0/a;});
    MultADAt(evects, D_tmp, X);
    Mult(X,XA,A);
}
}//namespace parelag
