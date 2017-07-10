/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#ifndef INNERPRODUCT_HPP_
#define INNERPRODUCT_HPP_

class InnerProductElemAgg
{
public:
	InnerProductElemAgg(int size_):size(size_){};
	virtual ~InnerProductElemAgg(){};
	virtual double operator()(const Vector & x, const Vector & y) const = 0;
	virtual void operator()(const MultiVector & x, const Vector & y, Vector & out) const;
	virtual void operator()(const Vector & x, const MultiVector & y, Vector & out) const;
	virtual void operator()(const MultiVector & x, const MultiVector & y, DenseMatrix & out) const;
	virtual void operator()(DenseMatrix & x, DenseMatrix & y, DenseMatrix & out) const;

protected:
	int size;
};

class StdInnerProduct : public InnerProductElemAgg
{
public:
	StdInnerProduct(int size);
	virtual ~StdInnerProduct(){};
	virtual double operator()(const Vector & x, const Vector & y) const;
	using InnerProductElemAgg::operator();
};

class DWeightedInnerProduct : public InnerProductElemAgg
{
public:
	DWeightedInnerProduct(const Vector & w);
	virtual ~DWeightedInnerProduct(){};
	virtual double operator()(const Vector & x, const Vector & y) const;
	using InnerProductElemAgg::operator();

private:
	const double * weight;

};

class WeightedInnerProduct : public InnerProductElemAgg
{
public:
	WeightedInnerProduct(const SparseMatrix & A);
	virtual ~WeightedInnerProduct(){};
	virtual double operator()(const Vector & x, const Vector & y) const;
	virtual void operator()(const MultiVector & x, const Vector & y, Vector & out) const;
	virtual void operator()(const MultiVector & x, const MultiVector & y, DenseMatrix & out) const;
	virtual void operator()(DenseMatrix & x, DenseMatrix & y, DenseMatrix & out) const;
	using InnerProductElemAgg::operator();

private:
	const SparseMatrix &  A;
	StdInnerProduct dot;

};

// a_0 = a;  a_{k+1} = a_k - (v_k, a_k)/(v_k, v_k) v_k
void Deflate(MultiVector & a, const MultiVector & v, const InnerProductElemAgg & inner_product);

// a = a - (v,a)/(v,v) v
void Deflate(MultiVector & a, const Vector & v, const InnerProductElemAgg & inner_product);

#endif /* INNERPRODUCT_HPP_ */
