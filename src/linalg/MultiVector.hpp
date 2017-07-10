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

#ifndef ELEMAGG_MULTIVECTOR_HPP_
#define ELEMAGG_MULTIVECTOR_HPP_

class MultiVector : public Vector
{
public:

	   /// Default constructor for Vector. Sets size = 0 and data = NULL
	   MultiVector ();

	   /// Copy constructor
	   MultiVector(const MultiVector &);

	   /// Creates vector of size s. (lda == s)
	   MultiVector (int _number_of_vector, int s);

	   // if lda < 0 then lda = size
	   MultiVector (double *_data, int _number_of_vector, int _size, int _lda = -1);

	   /// lda = s
	   void SetSizeAndNumberOfVectors(int size, int _number_of_vectors);

	   void GetSubMultiVector(const Array<int> &dofs, MultiVector &elemvect) const;
	   void GetSubMultiVector(const Array<int> &dofs, double *elem_data) const;

	   void SetSubMultiVector(const Array<int> &dofs, const MultiVector &elemvect);
	   void SetSubMultiVector(const Array<int> &dofs, double *elem_data);

	   void AddSubMultiVector(const Array<int> &dofs, const MultiVector &elemvect);
	   void AddSubMultiVector(const Array<int> &dofs, double *elem_data);

	   // m = this
	   void CopyToDenseMatrix(int ivectStart, int ivectEnd, int entryStart, int entryEnd, DenseMatrix & m);
	   // m = this^T
	   void CopyToDenseMatrixT(int ivectStart, int ivectEnd, int entryStart, int entryEnd, DenseMatrix & m);

	   void GetVectorView(int ivect, Vector & view);
	   void GetRangeView(int start, int end, MultiVector & view);
	   double * GetDataFromVector(int ivect){return this->data+ivect*lda;}
	   const double * GetDataFromVector(int ivect) const {return this->data+ivect*lda;}

	   int Size() const { return local_size;}
	   int LeadingDimension() const { return lda; }
	   int NumberOfVectors() const {return number_of_vectors;}

	   // out[i] = this[p[i]];
	   MultiVector * Permutation(const Array<int> & p) const;
	   MultiVector * Permutation(const int * p, int size_p = -1) const;

	   // out[p] = this;
	   MultiVector * InversePermutation(const Array<int> & p) const;
	   MultiVector * InversePermutation(const int * p, int size_p = -1, int out_size = -1) const;

	   void Scale(const Vector & s);
	   void InverseScale(const Vector & s);

	   void RoundSmallEntriesToZero(double smallValue);

	   MultiVector & operator=(const double *v);

	   /// Redefine '=' for vector = vector.
	   MultiVector & operator=(const MultiVector &v);

	   /// Redefine '=' for vector = constant.
	   MultiVector & operator=(double value);

	   void Copy(MultiVector & copy) const;

	   virtual ~MultiVector();



	   friend void MatrixTimesMultiVector(const SparseMatrix & M, const MultiVector & x, MultiVector & y);
	   friend void MatrixTTimesMultiVector(const SparseMatrix & M, const MultiVector & x, MultiVector & y);
	   friend void MatrixTimesMultiVector(double scaling, const SparseMatrix & M, const MultiVector & x, MultiVector & y);
	   friend void MatrixTTimesMultiVector(double scaling, const SparseMatrix & M, const MultiVector & x, MultiVector & y);

private:
	int number_of_vectors;
	int lda;
	int local_size;
};

//y = M * x;
void Mult(DenseMatrix & M, const MultiVector & x, MultiVector & y);
// y = y + scaling M*x
void Mult(double scaling, DenseMatrix & M, const MultiVector & x, MultiVector & y);

void add(double a, const MultiVector &x, double b, const MultiVector &y, MultiVector &z);

//y = A * x;
void Mult( BlockMatrix & A, const MultiVector & x, MultiVector & y);

#endif /* MULTIVECTOR_HPP_ */
