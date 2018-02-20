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

#include "ParELAG_InnerProduct.hpp"

#include "linalg/utilities/ParELAG_MatrixUtils.hpp"

namespace parelag
{
using namespace mfem;

void InnerProductElemAgg::operator()(const MultiVector & x, const Vector & y, Vector & out) const
{
    int nv = x.NumberOfVectors();
    out.SetSize(nv);
    Vector xx;

    for(int ivect(0); ivect < nv; ++ivect)
    {
        const_cast<MultiVector &>(x).GetVectorView(ivect, xx);
        out(ivect) = operator()(xx, y);
    }
}

void InnerProductElemAgg:: operator()(const Vector & x, const MultiVector & y, Vector & out) const
{
    return operator()(y,x,out);
}


void InnerProductElemAgg::operator()(const MultiVector & x, const MultiVector & y, DenseMatrix & out) const
{
    int nvx = x.NumberOfVectors();
    int nvy = y.NumberOfVectors();
    out.SetSize(nvx,nvy);
    Vector xx, yy;

    for(int ivect(0); ivect < nvx; ++ivect)
    {
        const_cast<MultiVector &>(x).GetVectorView(ivect, xx);
        for(int jvect(0); jvect < nvy; ++jvect)
        {
            const_cast<MultiVector &>(y).GetVectorView(jvect, yy);
            out(ivect, jvect) = operator()(xx, yy);
        }
    }
}

void InnerProductElemAgg::operator()(DenseMatrix & x, DenseMatrix & y, DenseMatrix & out) const
{
    int nvx = x.Width();
    int nvy = y.Width();
    out.SetSize(nvx,nvy);
    Vector xx, yy;

    for(int ivect(0); ivect < nvx; ++ivect)
    {
        x.GetColumnReference(ivect, xx);
        for(int jvect(0); jvect < nvy; ++jvect)
        {
            y.GetColumnReference(jvect, yy);
            out(ivect, jvect) = operator()(xx, yy);
        }
    }
}

StdInnerProduct::StdInnerProduct(int size):
    InnerProductElemAgg(size)
{

}

double StdInnerProduct::operator()(const Vector & x, const Vector & y) const
{
    if(x.Size() != y.Size() || x.Size() != size)
        mfem_error("DWeightedInnerProduct::operator()");

    return x*y;
}

DWeightedInnerProduct::DWeightedInnerProduct(const Vector & w):
    InnerProductElemAgg(w.Size()),
    weight(w.GetData())
{

}

double DWeightedInnerProduct::operator()(const Vector & x, const Vector & y) const
{
    if(x.Size() != y.Size() || x.Size() != size)
        mfem_error("DWeightedInnerProduct::operator()");

    const double * xi = x.GetData();
    const double * yi = y.GetData();
    const double * wi = weight;
    const double * end = wi+size;

    double ip(0);

    while(wi != end)
        ip += *(xi++) * *(yi++) * *(wi++);

    return ip;
}


WeightedInnerProduct::WeightedInnerProduct(const SparseMatrix & A_):
    InnerProductElemAgg(A_.Size()),
    A(A_),
    dot(A.Size())
{

}

double WeightedInnerProduct::operator()(const Vector & x, const Vector & y) const
{
    return A.InnerProduct(x,y);
}

void WeightedInnerProduct::operator()(const MultiVector & x, const Vector & y, Vector & out) const
{
    MultiVector Ax(x.NumberOfVectors(), x.Size());
    MatrixTimesMultiVector(A, x, Ax);
    dot(Ax, y, out);
}

void WeightedInnerProduct::operator()(const MultiVector & x, const MultiVector & y, DenseMatrix & out) const
{
    MultiVector Ax(x.NumberOfVectors(), x.Size());
    MatrixTimesMultiVector(A, x, Ax);
    dot(Ax, y, out);
}

void WeightedInnerProduct::operator()(DenseMatrix & x, DenseMatrix & y, DenseMatrix & out) const
{
    DenseMatrix Ax(x.Height(), x.Width());
    Mult(A, x, Ax);
    dot(Ax, y, out);
}

void Deflate(MultiVector &, MultiVector const&, InnerProductElemAgg const&)
{
    //int nv = v.NumberOfVectors();
    mfem_error("Not Implemented!!");
}

void Deflate(MultiVector & a, const Vector & v, const InnerProductElemAgg & inner_product)
{
    int nv = a.NumberOfVectors();
    Vector a_view;

    double s = -1./inner_product(v,v);

    for(int ivect(0); ivect < nv; ++ivect)
    {
        a.GetVectorView(ivect, a_view);
        add(a_view, inner_product(v,a_view)*s, v, a_view);
    }
}
}//namespace parelag
