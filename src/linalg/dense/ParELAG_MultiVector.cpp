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

#include <memory>

#include "ParELAG_MultiVector.hpp"

#include "utilities/elagError.hpp"
#include "utilities/MemoryUtils.hpp"

using namespace mfem;
using std::unique_ptr;

namespace parelag
{

MultiVector::MultiVector():
    Vector(),
    NumVectors_(0),
    LDA_(0),
    LocalSize_(0)
{
}

/// Creates vector of size s.
MultiVector::MultiVector (int _number_of_vectors, int s):
    Vector(_number_of_vectors * s),
    NumVectors_(_number_of_vectors),
    LDA_(s),
    LocalSize_(s)
{
}

MultiVector::MultiVector(double *_data,
                         int _number_of_vectors,
                         int _size,
                         int _lda):
    Vector(_data, _size*_number_of_vectors),
    NumVectors_(_number_of_vectors),
    LDA_(0),
    LocalSize_(_size)
{
    if(_lda < 0)
        _lda = _size;

    PARELAG_TEST_FOR_EXCEPTION(
        _lda < _size,
        std::runtime_error,
        "MultiVector::MultiVector(...): LDA too short!");

    LDA_ = _lda;
}

/// Copy constructor
MultiVector::MultiVector(const MultiVector & v):
    Vector(v),
    NumVectors_(v.NumVectors_),
    LDA_(v.LDA_),
    LocalSize_(v.LocalSize_)
{
}

void MultiVector::SetSizeAndNumberOfVectors(int size, int _number_of_vectors)
{
    SetSize(size*_number_of_vectors);
    NumVectors_ = _number_of_vectors;
    LDA_ = size;
    LocalSize_ = size;
}

void MultiVector::GetSubMultiVector(
    const Array<int> &dofs, MultiVector &elemvect) const
{
    elemvect.SetSizeAndNumberOfVectors(dofs.Size(), NumVectors_);
    GetSubMultiVector(dofs, elemvect.GetData() );
}

void MultiVector::GetSubMultiVector(
    const Array<int> &dofs, double *elem_data) const
{
    const double * this_start = GetData();
    for(int ivect(0); ivect < NumVectors_; ++ivect)
    {
        for(const int * it(dofs.GetData()); it != dofs.GetData()+dofs.Size(); ++it, ++elem_data)
            *elem_data = this_start[*it];

        this_start+=LDA_;
    }
}

void MultiVector::SetSubMultiVector(const Array<int> &dofs, const MultiVector &elemvect)
{

    PARELAG_TEST_FOR_EXCEPTION(
        elemvect.LocalSize_ != dofs.Size(),
        std::logic_error,
        "MultiVector::SetSubMultiVector(): "
        "Sizes of elemvect and dofs don't match.");

    PARELAG_TEST_FOR_EXCEPTION(
        elemvect.NumVectors_ != NumVectors_,
        std::logic_error,
        "MultiVector::SetSubMultiVector(): "
        "Number of vectors in elemvect and this don't match.");

    PARELAG_TEST_FOR_EXCEPTION(
        elemvect.LeadingDimension() != elemvect.Size(),
        std::logic_error,
        "MultiVector::SetSubMultiVector(): "
        "LeadingDimension and Size should be equal!");

    SetSubMultiVector(dofs, elemvect.GetData());
}

void MultiVector::SetSubMultiVector(const Array<int> &dofs, double *elem_data)
{

    double * this_start = GetData();
    for(int ivect(0); ivect < NumVectors_; ++ivect)
    {
        for(const int * it(dofs.GetData()); it != dofs.GetData()+dofs.Size(); ++it, ++elem_data)
            this_start[*it] = *elem_data;

        this_start+=LDA_;
    }
}

void MultiVector::AddSubMultiVector(const Array<int> &dofs, const MultiVector &elemvect)
{

    PARELAG_TEST_FOR_EXCEPTION(
        elemvect.LocalSize_ != dofs.Size(),
        std::logic_error,
        "MultiVector::AddSubMultiVector(): "
        "Sizes of elemvect and dofs don't match.");

    PARELAG_TEST_FOR_EXCEPTION(
        elemvect.NumVectors_ != NumVectors_,
        std::logic_error,
        "MultiVector::AddSubMultiVector(): "
        "Number of vectors in elemvect and this don't match.");

    PARELAG_TEST_FOR_EXCEPTION(
        elemvect.LeadingDimension() != elemvect.Size(),
        std::logic_error,
        "MultiVector::AddSubMultiVector(): "
        "LeadingDimension and Size should be equal!");

    SetSubMultiVector(dofs, elemvect.GetData());
}

void MultiVector::AddSubMultiVector(const Array<int> &dofs, double *elem_data)
{

    double * this_start = GetData();
    for(int ivect(0); ivect < NumVectors_; ++ivect)
    {
        for(const int * it(dofs.GetData()); it != dofs.GetData()+dofs.Size(); ++it, ++elem_data)
            this_start[*it] += *elem_data;

        this_start+=LDA_;
    }
}

void MultiVector::CopyToDenseMatrix(int ivectStart,
                                    int ivectEnd,
                                    int entryStart,
                                    int entryEnd,
                                    DenseMatrix & m)
{

#ifdef ELAG_DEBUG
    PARELAG_TEST_FOR_EXCEPTION(
        ivectEnd < ivectStart || entryEnd < entryStart,
        std::logic_error,
        "MultiVector::CopyToDenseMatrix(): "
        "ivectEnd < ivectStart or entryEnd < entryStart.");

    PARELAG_TEST_FOR_EXCEPTION(
        ivectStart < 0 || ivectEnd > NumVectors_,
        std::logic_error,
        "MultiVector::CopyToDenseMatrix(): "
        "ivectStart < 0 or ivectEnd > NumVectors_");

    PARELAG_TEST_FOR_EXCEPTION(
        entryStart < 0 || entryEnd > LocalSize_,
        std::logic_error,
        "MultiVector::CopyToDenseMatrix(): "
        "entryStart < 0 or entryEnd > LocalSize_");

#endif

    PARELAG_TEST_FOR_EXCEPTION(
        entryEnd-entryStart != m.Height() || ivectEnd - ivectStart != m.Width(),
        std::logic_error,
        "MultiVector::CopyToDenseMatrix(): m has incompatible sizes.");

    double * mdata = m.Data();
    double *d = GetData();

    int offset = ivectStart*LDA_;
    for(int ivect(ivectStart); ivect < ivectEnd; ++ivect)
    {
        for (int i = entryStart; i < entryEnd; i++)
            *(mdata++) = d[offset+i];
        offset += LDA_;
    }

}

void MultiVector::CopyToDenseMatrixT(int ivectStart,
                                     int ivectEnd,
                                     int entryStart,
                                     int entryEnd,
                                     DenseMatrix & m)
{
#ifdef ELAG_DEBUG
    PARELAG_TEST_FOR_EXCEPTION(
        ivectEnd < ivectStart || entryEnd < entryStart,
        std::logic_error,
        "MultiVector::CopyToDenseMatrix(): "
        "ivectEnd < ivectStart or entryEnd < entryStart.");

    PARELAG_TEST_FOR_EXCEPTION(
        ivectStart < 0 || ivectEnd > NumVectors_,
        std::logic_error,
        "MultiVector::CopyToDenseMatrix(): "
        "ivectStart < 0 or ivectEnd > NumVectors_");

    PARELAG_TEST_FOR_EXCEPTION(
        entryStart < 0 || entryEnd > LocalSize_,
        std::logic_error,
        "MultiVector::CopyToDenseMatrix(): "
        "entryStart < 0 or entryEnd > LocalSize_");

#endif

    PARELAG_TEST_FOR_EXCEPTION(
        entryEnd-entryStart != m.Width() || ivectEnd - ivectStart != m.Height(),
        std::logic_error,
        "MultiVector::CopyToDenseMatrix(): m has incompatible sizes.");

    double *d = GetData();
    int offset = ivectStart*LDA_;

    int mrow, mcol;
    for(int ivect(ivectStart); ivect < ivectEnd; ++ivect)
    {
        mrow = ivect-ivectStart;
        for (int i = entryStart; i < entryEnd; i++)
        {
            mcol = i - entryStart;
            m(mrow, mcol) = d[offset+i];
        }
        offset += LDA_;
    }

}

void MultiVector::GetVectorView(int ivect, Vector & view)
{
    view.SetDataAndSize(GetData()+ivect*LDA_, LocalSize_);
}

void MultiVector::GetRangeView(int start, int end, MultiVector & view)
{
    PARELAG_TEST_FOR_EXCEPTION(
        end < start,
        std::logic_error,
        "MultiVector::GetRangeView(): Invalid range: start > end!!");

    view.Destroy();

    view.SetDataAndSize(GetData() + start, size - start);
    view.LDA_  = LDA_;
    view.LocalSize_ = end-start;
    view.NumVectors_ = NumVectors_;
}

unique_ptr<MultiVector> MultiVector::Permutation(const Array<int> & p) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        p.Size() != LocalSize_,
        std::logic_error,
        "MultiVector::Permutation(): "
        "Sizes of p and this do not match.");

    return Permutation(p.GetData(), p.Size());
}

unique_ptr<MultiVector>
MultiVector::Permutation(const int * p, int p_size) const
{
    if(p_size == -1)
        p_size = LocalSize_;

    PARELAG_TEST_FOR_EXCEPTION(
        p_size != LocalSize_,
        std::logic_error,
        "MultiVector::Permutation(): "
        "Sizes of p and this do not match.");

    auto out = make_unique<MultiVector>(NumVectors_, LocalSize_);
    double * it_out = out->GetData();
    int offset = 0;
    double *d = GetData();

    for(int ivect(0); ivect < NumVectors_; ++ivect)
    {
        for(const int * it = p ; it != p + LDA_; ++it, ++it_out )
            *it_out = d[*it+offset];
        offset += LDA_;
    }

    return out;
}

unique_ptr<MultiVector>
MultiVector::InversePermutation(const Array<int> & p) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        p.Size() != LocalSize_,
        std::logic_error,
        "MultiVector::InversePermutation(): "
        "Sizes of p and this do not match.");

    return InversePermutation(p.GetData(), p.Size(), p.Max());
}

unique_ptr<MultiVector>
MultiVector::InversePermutation(const int * p, int p_size, int out_size) const
{
    if(p_size == -1)
        p_size = LocalSize_;

    if(out_size == -1)
        for(const int * it = p; it != p+p_size; ++it)
            out_size = (*it > out_size) ? *it : out_size;

    PARELAG_TEST_FOR_EXCEPTION(
        out_size < p_size,
        std::logic_error,
        "MultiVector::InversePermutation(): "
        "Incompatible out_size and p_size.");

    auto out = make_unique<MultiVector>(NumVectors_, out_size);
    double * o = out->GetData();
    const double * it_this = GetData();
    int offset_out = 0;
    int offset_in = 0;

    if(out_size > p_size)
    {
        double *d = out->GetData();
        for (int i=0; i < out->Size(); ++i)
            *(d++) = 0.0;
    }

    const double *d = GetData();
    for(int ivect(0); ivect < NumVectors_; ++ivect)
    {
        for(const int * it = p ; it != p + p_size; ++it, ++it_this )
            o[*it+offset_out] = *it_this;

        offset_out += out_size;
        offset_in += LDA_;
        it_this = d+offset_in;
    }

    return out;
}

MultiVector &MultiVector::operator=(const double *v)
{
    double *d = GetData();
    int offset = 0;
    for(int ivect(0); ivect < NumVectors_; ++ivect)
    {
        for (int i = 0; i < LocalSize_; i++)
            d[offset+i] = *(v++);
        offset += LDA_;
    }
    return *this;
}

MultiVector &MultiVector::operator=(const MultiVector &v)
{
    PARELAG_TEST_FOR_EXCEPTION(
        LocalSize_ != v.LocalSize_ || NumVectors_ != v.NumVectors_,
        std::logic_error,
        "MultiVector::operator=(): Sizes don't match.");

    int offset_this = 0;
    int offset_v = 0;
    const int lda_v = v.LeadingDimension();
    double * data_v = v.GetData();
    double *d = GetData();

    for(int ivect(0); ivect < NumVectors_; ++ivect)
    {
        for (int i = 0; i < LocalSize_; i++)
            d[offset_this+i] = data_v[offset_v+i];

        offset_this += LDA_;
        offset_v += lda_v;
    }

    return *this;
}

MultiVector &MultiVector::operator=(double value)
{
    double *d = GetData();
    int offset = 0;
    for(int ivect(0); ivect < NumVectors_; ++ivect)
    {
        for (int i = 0; i < LocalSize_; i++)
            d[offset+i] = value;
        offset += LDA_;
    }
    return *this;
}

void MultiVector::Copy(MultiVector & copy) const
{
    copy.SetSizeAndNumberOfVectors(LocalSize_, NumVectors_);
    copy = *this;
}

void MultiVector::Scale(const Vector & s)
{

    PARELAG_TEST_FOR_EXCEPTION(
        s.Size() != LocalSize_,
        std::logic_error,
        "MultiVector::Scale(): Sizes don't match.");

    double * it_this = GetData();
    int offset = 0;
    for(int ivect(0); ivect < NumVectors_; ++ivect)
    {
        for(const double * it = s.GetData(); it != s.GetData()+s.Size(); ++it, ++it_this)
            *it_this *= *it;

        offset += LDA_;
        it_this = GetData() + offset;
    }
}

void MultiVector::InverseScale(const Vector & s)
{

    PARELAG_TEST_FOR_EXCEPTION(
        s.Size() != LocalSize_,
        std::logic_error,
        "MultiVector::InverseScale(): Sizes don't match.");

    const double * it = s.GetData();
    const int s_size = s.Size();
    for (int ii = 0; ii < s_size; ++ii)
        PARELAG_TEST_FOR_EXCEPTION(
            it[ii] == 0,
            std::logic_error,
            "MultiVector::InverseScale(): s[i] == 0 for i = " << ii);

    double * it_this = GetData();
    int offset = 0;

    for(int ivect(0); ivect < NumVectors_; ++ivect)
    {

        for (it = s.GetData(); it != s.GetData()+s_size; ++it, ++it_this)
            *it_this /= *it;

        offset += LDA_;
        it_this = GetData() + offset;
    }
}

void MultiVector::RoundSmallEntriesToZero(double smallValue)
{
    double *d = GetData();
    int offset = 0;
    for(int ivect(0); ivect < NumVectors_; ++ivect)
    {
        for (int i = 0; i < LocalSize_; i++)
            if( fabs(d[offset+i]) < smallValue )
                d[offset+i] = 0.0;
        offset += LDA_;
    }
}

unique_ptr<MultiVector> MatrixTimesMultiVector(
    const SparseMatrix & M, const MultiVector & x)
{
    auto out = make_unique<MultiVector>(x.NumberOfVectors(),M.Size());
    MatrixTimesMultiVector(M,x,*out);
    return out;
}

void MatrixTimesMultiVector(
    const SparseMatrix & M, const MultiVector & x, MultiVector & y)
{
    PARELAG_TEST_FOR_EXCEPTION(
        M.Width() != x.Size() || M.Size() != y.Size() || x.NumberOfVectors() != y.NumberOfVectors(),
        std::logic_error,
        "MatrixTimesMultiVector: Dimensions don't match.");

    const int size = M.Size();
    const int * I = M.GetI();
    const int * J = M.GetJ();
    const double * val = M.GetData();

    int i,j,end;
    const double * xi_data(x.GetData());
    double * yi_data(y.GetData());

    for (int ivect(0); ivect < x.NumVectors_; ++ivect)
    {
        for (i = j = 0; i < size; i++)
        {
            double d = 0.0;
            for (end = I[i+1]; j < end; j++)
            {
                d += val[j] * xi_data[J[j]];
            }
            yi_data[i] = d;
        }
        xi_data += x.LDA_;
        yi_data += y.LDA_;
    }
}

void MatrixTimesMultiVector(double scaling, const SparseMatrix & M,
                            const MultiVector & x, MultiVector & y)
{
    PARELAG_TEST_FOR_EXCEPTION(
        M.Width() != x.Size() || M.Size() != y.Size() || x.NumberOfVectors() != y.NumberOfVectors(),
        std::runtime_error,
        "MatrixTimesMultiVector(...): "
        "size(M) = [ " << M.Size() << ", " << M.Width() << "]; \n"
        "size(x) = " << x.Size() << "# vector " << x.NumberOfVectors() << "\n"
        "size(y) = " << y.Size() << "# vector " << y.NumberOfVectors() << "\n"
        "Dimensions don't match y += s*M*x");

    const int size = M.Size();
    const int * I = M.GetI();
    const int * J = M.GetJ();
    const double * val = M.GetData();

    int i,j,end;
    const double * xi_data(x.GetData());
    double * yi_data(y.GetData());

    for( int ivect(0); ivect < x.NumVectors_; ++ivect)
    {
        for (i = j = 0; i < size; i++)
        {
            double d = 0.0;
            for (end = I[i+1]; j < end; j++)
            {
                d += val[j] * xi_data[J[j]];
            }
            yi_data[i] += scaling*d;
        }
        xi_data += x.LDA_;
        yi_data += y.LDA_;
    }
}

std::unique_ptr<MultiVector>
MatrixTTimesMultiVector(const SparseMatrix & M, const MultiVector & x)
{
    auto out = make_unique<MultiVector>(x.NumberOfVectors(),M.Width());
    MatrixTTimesMultiVector(M,x,*out);
    return out;
}

void MatrixTTimesMultiVector(
    const SparseMatrix & M, const MultiVector & x, MultiVector & y)
{
    y = 0.0;
    MatrixTTimesMultiVector(1.0, M, x, y);
}

void MatrixTTimesMultiVector(double scaling, const SparseMatrix & M,
                             const MultiVector & x, MultiVector & y)
{
    PARELAG_TEST_FOR_EXCEPTION(
        M.Size() != x.Size() || M.Width() != y.Size() || x.NumberOfVectors() != y.NumberOfVectors(),
        std::runtime_error,
        "MatrixTTimesMultiVector(...): Dimensions don't match!");

    const int size = M.Size();
    const int * I = M.GetI();
    const int * J = M.GetJ();
    const double * val = M.GetData();

    int i,j,end;
    const double * xi_data(x.GetData());
    double * yi_data(y.GetData());

    double sxi;
    for( int ivect(0); ivect < x.NumVectors_; ++ivect)
    {
        for (i = j = 0; i < size; i++)
        {
            sxi = scaling*xi_data[i];
            for (end = I[i+1]; j < end; j++)
                yi_data[ J[j] ] += val[j] * sxi;
        }

        xi_data += x.LDA_;
        yi_data += y.LDA_;
    }
}

extern "C"
{
    void dgemm_(char *, char *, int *, int *, int *, double *, double *,
                int *, double *, int *, double *, double *, int *);
}

void Mult(const DenseMatrix & M, const MultiVector & x, MultiVector & y)
{
    int nrows = M.Height();
    int ncols = M.Width();
    int nv = x.NumberOfVectors();

    PARELAG_TEST_FOR_EXCEPTION(
        nrows != y.Size(),
        std::runtime_error,
        "void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #1");

    PARELAG_TEST_FOR_EXCEPTION(
        ncols != x.Size(),
        std::runtime_error,
        "void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #2");

    PARELAG_TEST_FOR_EXCEPTION(
        x.NumberOfVectors() != y.NumberOfVectors(),
        std::runtime_error,
        "void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #3");

    if(ncols == 0)
        return;

    if(nv == 0 || nrows == 0 )
        return;

    double * xx(x.GetData()), *yy(y.GetData());
    int ldx = x.LeadingDimension();
    int ldy = y.LeadingDimension();

    static char transa = 'N', transb = 'N';
    static double alpha = 1.0, beta = 0.0;

    dgemm_(&transa, &transb, &nrows, &nv, &ncols,
           &alpha, M.Data(), &nrows, xx, &ldx,
           &beta, yy, &ldy);

}

void MultTranspose(const DenseMatrix & M, const MultiVector & x, MultiVector & y)
{
    int nrows = M.Height();
    int ncols = M.Width();
    int nv = x.NumberOfVectors();

    PARELAG_TEST_FOR_EXCEPTION(
        ncols != y.Size(),
        std::runtime_error,
        "void MultTranspose(DenseMatrix & M, const MultiVector & x, MultiVector & y) #1");

    PARELAG_TEST_FOR_EXCEPTION(
        nrows != x.Size(),
        std::runtime_error,
        "void MultTranspose(DenseMatrix & M, const MultiVector & x, MultiVector & y) #2");

    PARELAG_TEST_FOR_EXCEPTION(
        x.NumberOfVectors() != y.NumberOfVectors(),
        std::runtime_error,
        "void MultTranspose(DenseMatrix & M, const MultiVector & x, MultiVector & y) #3");

    if(nrows == 0)
        return;

    if(nv == 0 || ncols == 0 )
        return;

    double * xx(x.GetData()), *yy(y.GetData());
    int ldx = x.LeadingDimension();
    int ldy = y.LeadingDimension();

    static char transa = 'T', transb = 'N';
    static double alpha = 1.0, beta = 0.0;

    dgemm_(&transa, &transb, &ncols, &nv, &nrows,
           &alpha, M.Data(), &nrows, xx, &ldx,
           &beta, yy, &ldy);

}

void Mult(double scaling, DenseMatrix & M, const MultiVector & x, MultiVector & y)
{
    int nrows = M.Height();
    int ncols = M.Width();
    int nv = x.NumberOfVectors();

    PARELAG_TEST_FOR_EXCEPTION(
        nrows != y.Size(),
        std::runtime_error,
        "void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #1");

    PARELAG_TEST_FOR_EXCEPTION(
        ncols != x.Size(),
        std::runtime_error,
        "void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #2");

    PARELAG_TEST_FOR_EXCEPTION(
        x.NumberOfVectors() != y.NumberOfVectors(),
        std::runtime_error,
        "void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #3");

    PARELAG_TEST_FOR_EXCEPTION(
        ncols == 0,
        std::runtime_error,
        "void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #4");

    if(nv == 0 || nrows == 0 )
        return;

    double * xx(x.GetData()), *yy(y.GetData());
    int ldx = x.LeadingDimension();
    int ldy = y.LeadingDimension();

    static char transa = 'N', transb = 'N';
    static double  beta = 0.0;

    dgemm_(&transa, &transb, &nrows, &nv, &ncols,
           &scaling, M.Data(), &nrows, xx, &ldx,
           &beta, yy, &ldy);

}

void add(double a, const MultiVector &x, double b, const MultiVector &y,
         MultiVector &z)
{
#ifdef ELAG_DEBUG
    PARELAG_TEST_FOR_EXCEPTION(
        x.Size() != y.Size() || x.Size() != z.Size(),
        std::runtime_error,
        "subtract(const MultiVector &, const MultiVector &, MultiVector &) #1");

    PARELAG_TEST_FOR_EXCEPTION(
        x.NumberOfVectors() != y.NumberOfVectors() || x.NumberOfVectors() != z.NumberOfVectors(),
        std::runtime_error,
        "subtract(const MultiVector &, const MultiVector &, MultiVector &) #2");
#endif
    const double *xp = x.GetData();
    const double *yp = y.GetData();
    double       *zp = z.GetData();
    const int      s = x.Size();
    const int     nv = x.NumberOfVectors();

    const int ldx = x.LeadingDimension();
    const int ldy = y.LeadingDimension();
    const int ldz = z.LeadingDimension();

    for(int ivect(0); ivect < nv; ++ivect)
    {
        for (int i = 0; i < s; i++)
            zp[i] = a*xp[i] + b*yp[i];

        zp += ldz;
        xp += ldx;
        yp += ldy;
    }
}

void Mult(BlockMatrix & A, const MultiVector & x, MultiVector & y)
{
    elag_assert(x.NumberOfVectors() == y.NumberOfVectors());
    elag_assert(A.Height() == y.Size());
    elag_assert(A.Width() == x.Size());

    const int nv = x.NumberOfVectors();

    Vector xview, yview;
    for(int i(0); i < nv; ++i)
    {
        const_cast<MultiVector &>(x).GetVectorView(i,xview);
        y.GetVectorView(i,yview);
        A.Mult(xview, yview);
    }
}

void Mult(const mfem::HypreParMatrix & A, const MultiVector & x, MultiVector & y)
{
   elag_assert(x.NumberOfVectors() == y.NumberOfVectors());
   elag_assert(A.Height() == y.Size());
   elag_assert(A.Width() == x.Size());

   const int nv = x.NumberOfVectors();

   Vector xview, yview;
   for(int i(0); i < nv; ++i)
   {
       const_cast<MultiVector &>(x).GetVectorView(i,xview);
       y.GetVectorView(i,yview);
       A.Mult(xview, yview);
   }
}
}//namespace parelag
