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

#include "elag_linalg.hpp"
#include <cassert>


MultiVector::MultiVector():
	Vector(),
	number_of_vectors(0),
	lda(0),
	local_size(0)
{

}

	   /// Copy constructor
MultiVector::MultiVector(const MultiVector & v):
		Vector(v),
		number_of_vectors(v.number_of_vectors),
        lda(v.lda),
		local_size(v.local_size)
{

}

/// Creates vector of size s.
MultiVector::MultiVector (int _number_of_vectors, int s):
		Vector(_number_of_vectors * s),
		number_of_vectors(_number_of_vectors),
		lda(s),
		local_size(s)
{

}

MultiVector::MultiVector (double *_data, int _number_of_vectors, int _size, int _lda):
		Vector(_data, _size*_number_of_vectors),
		number_of_vectors(_number_of_vectors),
		lda(0),
		local_size(_size)
{
	if(_lda < 0)
		_lda = _size;

	if(_lda < _size)
		mfem_error("Too short lda");

	lda = _lda;
}

void MultiVector::SetSizeAndNumberOfVectors(int size, int _number_of_vectors)
{
	SetSize(size*_number_of_vectors);
	number_of_vectors = _number_of_vectors;
	lda = size;
	local_size = size;
}

void MultiVector::GetSubMultiVector(const Array<int> &dofs, MultiVector &elemvect) const
{
	elemvect.SetSizeAndNumberOfVectors(dofs.Size(), number_of_vectors);
	GetSubMultiVector(dofs, elemvect.GetData() );
}

void MultiVector::GetSubMultiVector(const Array<int> &dofs, double *elem_data) const
{
	double * this_start = data;
	for(int ivect(0); ivect < number_of_vectors; ++ivect)
	{
		for(const int * it(dofs.GetData()); it != dofs.GetData()+dofs.Size(); ++it, ++elem_data)
			*elem_data = this_start[*it];

		this_start+=lda;
	}
}

void MultiVector::SetSubMultiVector(const Array<int> &dofs, const MultiVector &elemvect)
{

	if(elemvect.local_size != dofs.Size())
		mfem_error("Size of elemvect and dofs don't match \n");

	if(elemvect.number_of_vectors != number_of_vectors)
		mfem_error("number of vectors in elemvect and this don't match");

	if(elemvect.LeadingDimension() != elemvect.Size())
		mfem_error("LeadingDimension and Size should be equal! ");

	SetSubMultiVector(dofs, elemvect.GetData());
}

void MultiVector::SetSubMultiVector(const Array<int> &dofs, double *elem_data)
{

	double * this_start = data;
	for(int ivect(0); ivect < number_of_vectors; ++ivect)
	{
		for(const int * it(dofs.GetData()); it != dofs.GetData()+dofs.Size(); ++it, ++elem_data)
			this_start[*it] = *elem_data;

		this_start+=lda;
	}
}

void MultiVector::AddSubMultiVector(const Array<int> &dofs, const MultiVector &elemvect)
{

	if(elemvect.local_size != dofs.Size())
		mfem_error("Size of elemvect and dofs don't match \n");

	if(elemvect.number_of_vectors != number_of_vectors)
		mfem_error("number of vectors in elemvect and this don't match");

	if(elemvect.LeadingDimension() != elemvect.Size())
		mfem_error("LeadingDimension and Size should be equal! ");

	SetSubMultiVector(dofs, elemvect.GetData());
}

void MultiVector::AddSubMultiVector(const Array<int> &dofs, double *elem_data)
{

	double * this_start = data;
	for(int ivect(0); ivect < number_of_vectors; ++ivect)
	{
		for(const int * it(dofs.GetData()); it != dofs.GetData()+dofs.Size(); ++it, ++elem_data)
			this_start[*it] += *elem_data;

		this_start+=lda;
	}
}

void MultiVector::CopyToDenseMatrix(int ivectStart, int ivectEnd, int entryStart, int entryEnd, DenseMatrix & m)
{

#if elemAGG_Debug
	if(ivectEnd < ivectStart || entryEnd < entryStart)
		mfem_error("MultiVector::CopyToDenseMatrix #1");

	if( ivectStart < 0 || ivectEnd > number_of_vectors)
		mfem_error("MultiVector::CopyToDenseMatrix #2");

	if( entryStart < 0 || entryEnd > local_size)
			mfem_error("MultiVector::CopyToDenseMatrix #2");

#endif

	if(entryEnd-entryStart != m.Height() || ivectEnd - ivectStart != m.Width() )
		mfem_error("MultiVector::CopyToDenseMatrix: m has incompatible sizes");

	double * mdata = m.Data();

	int offset = ivectStart*lda;
	for(int ivect(ivectStart); ivect < ivectEnd; ++ivect)
	{
		for (int i = entryStart; i < entryEnd; i++)
			*(mdata++) = data[offset+i];
		offset += lda;
	}

}

void MultiVector::CopyToDenseMatrixT(int ivectStart, int ivectEnd, int entryStart, int entryEnd, DenseMatrix & m)
{
#if elemAGG_Debug
	if(ivectEnd < ivectStart || entryEnd < entryStart)
		mfem_error("MultiVector::CopyToDenseMatrixT #1");

	if( ivectStart < 0 || ivectEnd > number_of_vectors)
		mfem_error("MultiVector::CopyToDenseMatrixT #2");

	if( entryStart < 0 || entryEnd > local_size)
			mfem_error("MultiVector::CopyToDenseMatrixT #2");

#endif

	if(entryEnd-entryStart != m.Width() || ivectEnd - ivectStart != m.Height() )
		mfem_error("MultiVector::CopyToDenseMatrixT: m has incompatible sizes");

	int offset = ivectStart*lda;

	int mrow, mcol;
	for(int ivect(ivectStart); ivect < ivectEnd; ++ivect)
	{
		mrow = ivect-ivectStart;
		for (int i = entryStart; i < entryEnd; i++)
		{
			mcol = i - entryStart;
			m(mrow, mcol) = data[offset+i];
		}
		offset += lda;
	}

}

void MultiVector::GetVectorView(int ivect, Vector & view)
{
	view.SetDataAndSize(data+ivect*lda, local_size);
}

void MultiVector::GetRangeView(int start, int end, MultiVector & view)
{

	if(end < start)
		mfem_error("MultiVector::GetRangeView invalid range: start > end!!");

	if(view.allocsize > 0)
		delete[] view.data;

	view.data = data+start;
	view.size = size - start;
	view.allocsize = -view.size;
	view.lda  = lda;
	view.local_size = end-start;
	view.number_of_vectors = number_of_vectors;

}

MultiVector * MultiVector::Permutation(const Array<int> & p) const
{
	if(p.Size() != local_size)
		mfem_error("p and this does not match \n");

	return Permutation( p.GetData(), p.Size() );
}

MultiVector * MultiVector::Permutation(const int * p, int p_size) const
{
	if(p_size == -1)
		p_size = local_size;

	if(p_size != local_size)
		mfem_error("p and this does not match \n");

	MultiVector * out = new MultiVector(number_of_vectors, local_size);
	double * it_out = out->GetData();
	int offset = 0;

	for(int ivect(0); ivect < number_of_vectors; ++ivect)
	{
		for(const int * it = p ; it != p + lda; ++it, ++it_out )
			*it_out = data[*it+offset];
		offset += lda;
	}

	return out;
}


MultiVector * MultiVector::InversePermutation(const Array<int> & p) const
{
	if(p.Size() != local_size)
		mfem_error("p and this does not match \n");

	return InversePermutation(p.GetData(), p.Size(), p.Max() );
}

MultiVector * MultiVector::InversePermutation(const int * p, int p_size, int out_size) const
{
	if(p_size == -1)
		p_size = local_size;

	if(out_size == -1)
		for(const int * it = p; it != p+p_size; ++it)
			out_size = (*it > out_size) ? *it : out_size;

	if(out_size < p_size)
		mfem_error("Incompatible out_size and p_size");

	MultiVector * out = new MultiVector(number_of_vectors, out_size);
	double * o = out->GetData();
	const double * it_this = data;
	int offset_out = 0;
	int offset_in = 0;

	if(out_size > p_size)
		std::fill(out->data, out->data+ out->size, 0.0);

	for(int ivect(0); ivect < number_of_vectors; ++ivect)
	{
		for(const int * it = p ; it != p + p_size; ++it, ++it_this )
			o[*it+offset_out] = *it_this;

		offset_out += out_size;
		offset_in += lda;
		it_this = data+offset_in;
	}

	return out;
}

MultiVector &MultiVector::operator=(const double *v)
{
	int offset = 0;
	for(int ivect(0); ivect < number_of_vectors; ++ivect)
	{
		for (int i = 0; i < local_size; i++)
			data[offset+i] = *(v++);
		offset += lda;
	}
   return *this;
}

MultiVector &MultiVector::operator=(const MultiVector &v)
{
	if(local_size != v.local_size || number_of_vectors != v.number_of_vectors)
		mfem_error("Size don't match");

	int offset_this = 0;
	int offset_v = 0;
	int lda_v = v.LeadingDimension();
	double * data_v = v.GetData();

	for(int ivect(0); ivect < number_of_vectors; ++ivect)
	{
		for (int i = 0; i < local_size; i++)
			data[offset_this+i] = data_v[offset_v+i];

		offset_this += lda;
		offset_v += lda_v;
	}

   return *this;
}

MultiVector &MultiVector::operator=(double value)
{
	int offset = 0;
	for(int ivect(0); ivect < number_of_vectors; ++ivect)
	{
		for (int i = 0; i < local_size; i++)
			data[offset+i] = value;
		offset += lda;
	}
   return *this;
}

void MultiVector::Copy(MultiVector & copy) const
{
	copy.SetSizeAndNumberOfVectors(local_size, number_of_vectors);
	copy = *this;
}

void MultiVector::Scale(const Vector & s)
{

	if(s.Size() != local_size)
		mfem_error("Size don't match \n");

	double * it_this = data;
	int offset = 0;
	for(int ivect(0); ivect < number_of_vectors; ++ivect)
	{
		for(const double * it = s.GetData(); it != s.GetData()+s.Size(); ++it, ++it_this)
			*it_this *= *it;

		offset += lda;
		it_this = data + offset;
	}
}

void MultiVector::InverseScale(const Vector & s)
{

	if(s.Size() != local_size)
		mfem_error("Size don't match \n");

	for(const double * it = s.GetData(); it != s.GetData()+s.Size(); ++it)
		if(*it == 0)
			mfem_error("s[i] == 0 for some i \n");

	double * it_this = data;
	int offset = 0;

	for(int ivect(0); ivect < number_of_vectors; ++ivect)
	{

		for(const double * it = s.GetData(); it != s.GetData()+s.Size(); ++it, ++it_this)
			*it_this /= *it;

		offset += lda;
		it_this = data + offset;

	}
}

void MultiVector::RoundSmallEntriesToZero(double smallValue)
{
	int offset = 0;
	for(int ivect(0); ivect < number_of_vectors; ++ivect)
	{
		for (int i = 0; i < local_size; i++)
			if( fabs(data[offset+i]) < smallValue )
				data[offset+i] = 0.0;
		offset += lda;
	}
}

MultiVector::~MultiVector()
{

}

void MatrixTimesMultiVector(const SparseMatrix & M, const MultiVector & x, MultiVector & y)
{
	if(M.Width() != x.Size() || M.Size() != y.Size() || x.NumberOfVectors() != y.NumberOfVectors() )
		mfem_error("Dimensions don't match");

	int size = M.Size();
	const int * I = M.GetI();
	const int * J = M.GetJ();
	const double * val = M.GetData();

	int i,j,end;
	double * xi_data(x.data), * yi_data(y.data);

	for( int ivect(0); ivect < x.number_of_vectors; ++ivect)
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
		xi_data += x.lda;
		yi_data += y.lda;
	}
}

void MatrixTimesMultiVector(double scaling, const SparseMatrix & M, const MultiVector & x, MultiVector & y)
{
	if(M.Width() != x.Size() || M.Size() != y.Size() || x.NumberOfVectors() != y.NumberOfVectors() )
	{
		std::cout << "size(M) = [ " << M.Size() << ", " << M.Width() << "]; \n";
		std::cout << "size(x) = " << x.Size() << "# vector " << x.NumberOfVectors() << "\n";
		std::cout << "size(y) = " << y.Size() << "# vector " << y.NumberOfVectors() << "\n";
		mfem_error("Dimensions don't match y += s*M*x");
	}

	int size = M.Size();
	const int * I = M.GetI();
	const int * J = M.GetJ();
	const double * val = M.GetData();

	int i,j,end;
	double * xi_data(x.data), * yi_data(y.data);

	for( int ivect(0); ivect < x.number_of_vectors; ++ivect)
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
		xi_data += x.lda;
		yi_data += y.lda;
	}
}

void MatrixTTimesMultiVector(const SparseMatrix & M, const MultiVector & x, MultiVector & y)
{
	y = 0.0;
	MatrixTTimesMultiVector(1.0, M, x, y);
}



void MatrixTTimesMultiVector(double scaling, const SparseMatrix & M, const MultiVector & x, MultiVector & y)
{
	if(M.Size() != x.Size() || M.Width() != y.Size() || x.NumberOfVectors() != y.NumberOfVectors() )
		mfem_error("Dimensions don't match");

	int size = M.Size();
	const int * I = M.GetI();
	const int * J = M.GetJ();
	const double * val = M.GetData();

	int i,j,end;
	double * xi_data(x.data), * yi_data(y.data);

	double sxi;
	for( int ivect(0); ivect < x.number_of_vectors; ++ivect)
	{
		for (i = j = 0; i < size; i++)
		{
			sxi = scaling*xi_data[i];
			for (end = I[i+1]; j < end; j++)
				yi_data[ J[j] ] += val[j] * sxi;
		}

		xi_data += x.lda;
		yi_data += y.lda;
	}
}

extern "C"
{
void dgemm_(char *, char *, int *, int *, int *, double *, double *,
       int *, double *, int *, double *, double *, int *);
}

void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y)
{
	int nrows = M.Height();
	int ncols = M.Width();
	int nv = x.NumberOfVectors();

	if( nrows != y.Size())
		mfem_error("void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #1");

	if( ncols != x.Size())
		mfem_error("void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #2");

	if( x.NumberOfVectors() != y.NumberOfVectors() )
		mfem_error("void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #3");

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

void Mult(double scaling, DenseMatrix & M, const MultiVector & x, MultiVector & y)
{
	int nrows = M.Height();
	int ncols = M.Width();
	int nv = x.NumberOfVectors();

	if( nrows != y.Size())
		mfem_error("void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #1");

	if( ncols != x.Size())
		mfem_error("void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #2");

	if( x.NumberOfVectors() != y.NumberOfVectors() )
		mfem_error("void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #3");

	if(ncols == 0)
		mfem_error("void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y) #4");

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

void add(double a, const MultiVector &x, double b, const MultiVector &y, MultiVector &z)
{
#ifdef ELAG_DEBUG
   if (x.Size() != y.Size() || x.Size() != z.Size() )
      mfem_error ("subtract(const MultiVector &, const MultiVector &, MultiVector &) #1");

   if (x.NumberOfVectors() != y.NumberOfVectors() || x.NumberOfVectors() != z.NumberOfVectors() )
       mfem_error ("subtract(const MultiVector &, const MultiVector &, MultiVector &) #2");
#endif
   const double *xp = x.GetData();
   const double *yp = y.GetData();
   double       *zp = z.GetData();
   int            s = x.Size();
   int           nv = x.NumberOfVectors();

   int ldx = x.LeadingDimension();
   int ldy = y.LeadingDimension();
   int ldz = z.LeadingDimension();

   for(int ivect(0); ivect < nv; ++ivect)
   {
	   for (int i = 0; i < s; i++)
		   zp[i] = a*xp[i] + b*yp[i];

	   zp += ldz;
	   xp += ldx;
	   yp += ldy;
   }
}

void Mult( BlockMatrix & A, const MultiVector & x, MultiVector & y)
{
	elag_assert(x.NumberOfVectors() == y.NumberOfVectors() );
	elag_assert(A.Height() == y.Size() );
	elag_assert(A.Width() == x.Size() );

	int nv = x.NumberOfVectors();

	Vector xview, yview;
	for(int i(0); i < nv; ++i)
	{
		const_cast<MultiVector &>(x).GetVectorView(i,xview);
		y.GetVectorView(i,yview);
		A.Mult(xview, yview);
	}
}
