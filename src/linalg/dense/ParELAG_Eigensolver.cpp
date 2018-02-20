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

#include "ParELAG_Eigensolver.hpp"

#include "ParELAG_LAPACK.hpp"
#include "linalg/utilities/ParELAG_MatrixUtils.hpp"

namespace parelag
{
using namespace mfem;

SymEigensolver::SymEigensolver():
    itype_(1),
    jobz_('V'),
    range_('A'),
    uplo_('U'),
    abstol_(2*std::numeric_limits<double>::min()),
    NumFoundEigs_(0),
    lwork_(0),
    info_(0),
    n_max_(-1),
    doNotOverwrite_(false)
{
}


void SymEigensolver::CheckSPDEigenvalues(
    const std::vector<double>& w, double tol)
{
    PARELAG_TEST_FOR_EXCEPTION(
        w.size() > 0 && !std::is_sorted(w.begin(),w.end()),
        std::runtime_error,
        "SymEigensolver::CheckSPDEigenvalues(...): "
        "evals are not sorted :( ");

    PARELAG_TEST_FOR_EXCEPTION(
        w.size() > 0 && w[0] < -tol,
        std::runtime_error,
        "SymEigensolver::CheckSPDEigenvalues(...): "
        "First eigenvalue = " << w[0] << "; abstol_ = " << abstol_ << "\n"
        "A is not SPD!");
}

//! Compute all eigenvectors and eigenvalues A x = lambda x
// called in SVDCalculator.cpp
int SymEigensolver::ComputeAll(DenseMatrix & A_mat, std::vector<double> & evals,
                               DenseMatrix & evects)
{
    int n = A_mat.Size();

    if(n_max_ < n)
        AllocateWorkspace(n);

    jobz_  = 'V';
    range_ = 'A';
    uplo_  = 'U';

    double * a = nullptr;

    if(doNotOverwrite_)
    {
        std::copy(A_mat.Data(), A_mat.Data()+n*n, A_.begin());
        a = A_.data();
    }
    else
    {
        a = A_mat.Data();
    }

    evals.resize(n);
    evects.SetSize(n,n);

    PARELAG_ASSERT(evals.data());

    auto ret = Lapack<double>::SYEVX(
        jobz_, range_, uplo_, n, a, n, 0.0, 0.0, 0, 0,
        abstol_, evals.data(), evects.Data(), n,
        work_.data(), lwork_, iwork_.data(), ifail_.data());

    info_ = ret.first;
    NumFoundEigs_ = ret.second;
    return info_;
}

//! Compute all eigenvalues of the generalized eigenvalue problem
//!  A x = lambda B x
// CALLED through SmallerMagnitude
int SymEigensolver::ComputeGeneralizedAll(
    DenseMatrix & A_mat, DenseMatrix & B_mat, std::vector<double>& evals)
{
    int n = A_mat.Size();

    if(n_max_ < n)
        AllocateWorkspace(n);

    jobz_  = 'N';
    range_ = 'A';
    uplo_  = 'U';

    double * a = nullptr, * b = nullptr;

    if(doNotOverwrite_)
    {
        std::copy(A_mat.Data(),A_mat.Data()+n*n,A_.begin());
        std::copy(B_mat.Data(),B_mat.Data()+n*n,B_.begin());
        a = A_.data();
        b = B_.data();
    }
    else
    {
        a = A_mat.Data();
        b = B_mat.Data();
    }

    evals.resize(n);

    PARELAG_ASSERT(evals.data());
    auto ret = Lapack<double>::SYGVX(
        itype_, jobz_, range_, uplo_, n, a, n, b, n, 0.0, 0.0, 0, 0,
        abstol_, evals.data(), nullptr, n, work_.data(),
        lwork_, iwork_.data(), ifail_.data());

    info_ = ret.first;
    NumFoundEigs_ = ret.second;
    return info_;
}

int SymEigensolver::ComputeFixedNumber(
    DenseMatrix & A_mat, std::vector<double> & evals,
    int il, int iu)
{
    if(il > iu)
    {
        evals.resize(0);
        return 0;
    }

    int n = A_mat.Size();

    if(n_max_ < n)
        AllocateWorkspace(n);

    jobz_  = 'N';
    range_ = 'I';
    uplo_  = 'U';

    double * a = nullptr;

    if(doNotOverwrite_)
    {
        std::copy(A_mat.Data(), A_mat.Data()+n*n, A_.begin());
        a = A_.data();
    }
    else
    {
        a = A_mat.Data();
    }

    evals.resize(n);
    auto ret = Lapack<double>::SYEVX(
        jobz_, range_, uplo_, n, a, n, 0.0, 0.0, il, iu,
        abstol_, evals.data(), nullptr, n, work_.data(), lwork_,
        iwork_.data(), ifail_.data());

    info_ = ret.first;
    NumFoundEigs_ = ret.second;

    PARELAG_TEST_FOR_EXCEPTION(
        NumFoundEigs_ != iu- il + 1,
        std::runtime_error,
        "SymEigensolver::Compute(DenseMatrix & A_mat, Vector & evals,"
        "int il, int iu)");
    evals.resize(iu-il+1);

    return info_;
}

// Compute the il-th to up-th eigenvectors and eigenvalues A x = lambda x
// CALLED
int SymEigensolver::ComputeFixedNumber(
    DenseMatrix & A_mat, std::vector<double> & evals, DenseMatrix & evects,
    int il, int iu)
{
    if(il > iu)
    {
        evals.resize(0);
        return 0;
    }

    int n = A_mat.Size();

    if(n_max_ < n)
        AllocateWorkspace(n);

    jobz_  = 'V';
    range_ = 'I';
    uplo_  = 'U';

    double * a = nullptr;

    if(doNotOverwrite_)
    {
        std::copy(A_mat.Data(), A_mat.Data()+n*n, A_.begin());
        a = A_.data();
    }
    else
    {
        a = A_mat.Data();
    }

    evects.SetSize(n, iu-il+1);
    evals.resize(n);
    auto ret = Lapack<double>::SYEVX(
        jobz_, range_, uplo_, n, a, n, 0.0, 0.0, il, iu,
        abstol_, evals.data(), evects.Data(),
        n, work_.data(), lwork_, iwork_.data(), ifail_.data());

    info_ = ret.first;
    NumFoundEigs_ = ret.second;

    PARELAG_TEST_FOR_EXCEPTION(
        NumFoundEigs_ != iu- il + 1,
        std::runtime_error,
        "SymEigensolver::Compute(DenseMatrix & A_mat, Vector & evals,"
        "int il, int iu)");
    evals.resize(iu-il+1);

    return info_;
}

// Compute the il-th to up-th eigenvectors and eigenvalues of the generalized
// eigenvalue problem A x = lambda B x
// CALLED through SmallerMagnitude
int SymEigensolver::ComputeGeneralizedFixedNumber(
    DenseMatrix & A_mat, DenseMatrix & B_mat, std::vector<double> & evals,
    DenseMatrix & evects, int il, int iu)
{
    if(il > iu)
    {
        evals.resize(0);
        return 0;
    }

    int n = A_mat.Size();

    if (n_max_ < n)
        AllocateWorkspace(n);

    jobz_  = 'V';
    range_ = 'I';
    uplo_  = 'U';

    double * a = nullptr, * b = nullptr;

    if (doNotOverwrite_)
    {
        std::copy(A_mat.Data(), A_mat.Data()+n*n, A_.begin());
        std::copy(B_mat.Data(), B_mat.Data()+n*n, B_.end());
        a = A_.data();
        b = B_.data();
    }
    else
    {
        a = A_mat.Data();
        b = B_mat.Data();
    }

    evects.SetSize(n, iu-il+1);
    evals.resize(n);
    auto ret = Lapack<double>::SYGVX(
        itype_, jobz_, range_, uplo_, n, a, n, b, n,
        0.0, 0.0, il, iu, abstol_, evals.data(),
        evects.Data(), n, work_.data(), lwork_, iwork_.data(), ifail_.data());

    info_ = ret.first;
    NumFoundEigs_ = ret.second;

    PARELAG_TEST_FOR_EXCEPTION(
        NumFoundEigs_ != iu- il + 1,
        std::runtime_error,
        "SymEigensolver::Compute("
        "DenseMatrix & A_mat, std::vector<double> & evals, int il, int iu)");
    evals.resize(iu-il+1);

    return info_;
}

// Compute the il-th to up-th eigenvectors and eigenvalues of the generalized eigenvalue problem
// A x = lambda D x, D = diag(d)
// CALLED through SmallerMagnitude
int SymEigensolver::ComputeDiagonalFixedNumber(
    DenseMatrix & A_mat, Vector & d, std::vector<double> & evals,
    DenseMatrix & evects, int il, int iu)
{
    PARELAG_TEST_FOR_EXCEPTION(
        itype_ != 1,
        std::runtime_error,
        "SymEigensolver::ComputeDiagonalFixedNumber(...): "
        "Only A x = lambda D x is supported.");

    const int n = A_mat.Size();
    double * a = nullptr;

    if(n_max_ < n)
        AllocateWorkspace(n);

    bool oldOpt = doNotOverwrite_;

    if(doNotOverwrite_)
    {
        std::copy(A_mat.Data(), A_mat.Data()+n*n, A_.begin());
        a = A_.data();
        doNotOverwrite_ = false;
    }
    else
    {
        a = A_mat.Data();
    }

    DenseMatrix sA(a,n,n);
    sA.InvSymmetricScaling(d);

    int r;
    r = ComputeFixedNumber(sA, evals, evects, il, iu);

    Vector sqrtd(n);
    for (double * out = sqrtd.GetData(), *in = d.GetData();
         in != d.GetData()+n;
         ++out, ++in)
    {
        *out = sqrt(*in);
    }

    evects.InvLeftScaling(sqrtd);

    sA.ClearExternalData();
    doNotOverwrite_ = oldOpt;

    return r;
}

// CALLED
int SymEigensolver::ComputeInterval(
    DenseMatrix & A_mat, std::vector<double> & evals, double vl, double vu)
{
    int n = A_mat.Size();

    if(n_max_ < n)
        AllocateWorkspace(n);

    jobz_  = 'N';
    range_ = 'V';
    uplo_  = 'U';

    double * a = nullptr;

    if(doNotOverwrite_)
    {
        std::copy(A_mat.Data(), A_mat.Data()+n*n, A_.begin());
        a = A_.data();
    }
    else
    {
        a = A_mat.Data();
    }

    evals.resize(n);
    auto ret = Lapack<double>::SYEVX(
        'N','V','U', n, a, n, vl, vu, 0, 0, abstol_,
        evals.data(), nullptr, n, work_.data(), lwork_, iwork_.data(),
        ifail_.data());

    info_ = ret.first;
    NumFoundEigs_ = ret.second;

    evals.resize(NumFoundEigs_);

    return info_;
}

// CALLED through SmallerMagnitude
int SymEigensolver::ComputeDiagonalInterval(
    DenseMatrix & A_mat, Vector & d, std::vector<double> & evals,
    double vl, double vu)
{
    PARELAG_TEST_FOR_EXCEPTION(
        itype_ != 1,
        std::runtime_error,
        "SymEigensolver::ComputeDiagonalInterval(...): "
        "Only A x = lambda D x is supported.");

    const int n = A_mat.Size();
    double * a = nullptr;

    if(n_max_ < n)
        AllocateWorkspace(n);

    bool oldOpt = doNotOverwrite_;

    if(doNotOverwrite_)
    {
        std::copy(A_mat.Data(), A_mat.Data()+n*n, A_.begin());
        a = A_.data();
        doNotOverwrite_ = false;
    }
    else
    {
        a = A_mat.Data();
    }

    DenseMatrix sA(a,n,n);
    sA.InvSymmetricScaling(d);

    int r = ComputeInterval(sA, evals, vl, vu);

    sA.ClearExternalData();
    doNotOverwrite_ = oldOpt;

    return r;
}

//! Compute no more than max(\a max_evects, 1) eigenvectors and eigenvalues of
//! the generalized eigenvalue problem A x = lambda D x, D = diag(d)
//!  such that | lambda_i | \leq rel_tol max_eval
//
// ATB: under the assumption that finding eigenvalues (without vectors) is much
//      faster than finding both, this implementation might make a little sense.
//      I am quite skeptical of that assumption.
int SymEigensolver::ComputeDiagonalSmallerMagnitude(
    SparseMatrix & As, Vector & d, std::vector<double> & evals,
    DenseMatrix & evects, double rel_tol, double max_eval, size_t max_evects)
{
    PARELAG_TEST_FOR_EXCEPTION(
        itype_ != 1,
        std::runtime_error,
        "SymEigensolver::ComputeDiagonalSmallerMagnitude(...): "
        "Only A x = lambda D x is supported.");

    DenseMatrix A_mat;
    Full(As, A_mat);

    bool oldOpt = doNotOverwrite_;

    doNotOverwrite_ = true;
    double tol = max_eval*rel_tol;
    std::vector<double> tmp;
    int r = ComputeDiagonalInterval(A_mat, d, tmp, -1., tol);
    if (r)
        return r;

#ifdef ELAG_DEBUG
    CheckSPDEigenvalues(tmp);
#endif

    int m = tmp.size() < max_evects ? tmp.size() : max_evects;
    if (m < 1) m = 1;

    doNotOverwrite_ = false;

    r = ComputeDiagonalFixedNumber(A_mat, d, evals, evects, 1, m);

    doNotOverwrite_ = oldOpt;

    return r;
}

//! Compute no more than max(\a max_evects, 1) eigenvectors and eigenvalues
//! of the generalized eigenvalue problem A x = lambda B x such that
//!  | lambda_i | \leq rel_tol |\lambda_\max|
//
// ATB: under the assumption that finding eigenvalues (without vectors) is much
//      faster than finding both, this implementation might make a little sense.
//      I am quite skeptical of that assumption.
int SymEigensolver::ComputeGeneralizedSmallerMagnitude(
    DenseMatrix & A_mat, DenseMatrix & B_mat, std::vector<double>& evals,
    DenseMatrix & evects, double rel_tol, int maxEvals)
{
    bool oldOpt = doNotOverwrite_;

    doNotOverwrite_ = true;
    const int n = A_mat.Size();
    std::vector<double> tmp;
    int r = ComputeGeneralizedAll(A_mat,B_mat,tmp);

    if (r)
        return r;

#ifdef ELAG_DEBUG
    CheckSPDEigenvalues(tmp);
#endif

    double tol = tmp.back()*rel_tol;
    int m;
    if (maxEvals == -1) maxEvals = n;
    for ( m = 0; m < maxEvals && fabs(tmp[m]) < tol; ++m);

    doNotOverwrite_ = oldOpt;

    m = std::max(1,m);
    r = ComputeGeneralizedFixedNumber(A_mat, B_mat, evals, evects, 1, m);

    return r;
}

void SymEigensolver::AllocateWorkspace(int n)
{
    double qwork = 0.0;
    double qworkg = 0.0;
    n_max_ = n;
    lwork_ = -1;

    A_.resize(n*n,0.0);
    B_.resize(n*n,0.0);
    //Z_.resize(n*n,0.0);
    iwork_.resize(5*n,0);
    ifail_.resize(n,0);

    double vl(-1.), vu(1.);
    int il(1), iu(n);

    auto ret = Lapack<double>::SYEVX(
        jobz_, range_, uplo_, n, A_.data(), n, vl, vu, il, iu, abstol_,
        nullptr, nullptr, n, &qwork, lwork_, iwork_.data(), ifail_.data());

    ret = Lapack<double>::SYGVX(
        itype_, jobz_, range_, uplo_, n, A_.data(), n, B_.data(), n,
        vl, vu, il, iu, abstol_, nullptr, nullptr, n,
        &qworkg, lwork_, iwork_.data(), ifail_.data());

    lwork_ = std::max((int)qwork,(int)qworkg) + 1;
    work_.resize(lwork_,0.0);
}
}//namespace parelag
