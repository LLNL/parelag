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

#ifndef TOPOLOGYTABLE_HPP_
#define TOPOLOGYTABLE_HPP_

/**
 * @class TopologyTable.
 * \brief A simple wrapper for SerialCSRMatrix that adds a few extra functionality specific for Topological Tables.
 *
 * TopologicalTables are sparse matrices whose entries are plus or minus ones.
 *
 * Note: if A and B are TopologyTable objects then Mult(A,B) will be a SerialCSRMatrix object.
 * Use MultOrientation to obtain a TopologyTable
 *
 */
class TopologyTable : public SerialCSRMatrix
{
public:
	//! Convert a SerialCSRMatrix A into a TopologyTable. Note: A will be invalid. This routine does not check if some data are different from +-1
	explicit TopologyTable(SerialCSRMatrix & A);
	//! Convert a SerialCSRMatrix *A into a TopologyTable. Note: A will be deleted. This routine does not check if some data are different from +-1
	explicit TopologyTable(SerialCSRMatrix * A);
	//! Build a topology table from the i,j,val arrays. This routine does not check if some data are different from +-1
	TopologyTable(int *i, int *j, double *data, int m, int n);
	//! Build a topology table from an mfem table. MFEM tables use a XOR transformation to encode orientation in the j array.
	TopologyTable(const Table & T);
	//! Make sure that all entries in the Table are +-1 or 0
	void OrientationTransform();
	//! Eliminate from the sparsity pattern the entries that are less than tol.
	/**
	 * Note this elimination is done in-place, so no new memory is allocated, but also no memory is freed.
	 */
	void DropSmallEntries(double tol);
	//! y = Graph(A)*x, where Graph(A) is the boolean matrix representing the sparsity pattern of A. (Vector version).
	void WedgeMult(const Vector & x, Vector & y);
	//! y = Graph(A)*x, where Graph(A) is the boolean matrix representing the sparsity pattern of A. (Array<int> version).
	void WedgeMult(const Array<int> & x, Array<int> & y);
	//! y = Graph(A)^T*x, where Graph(A) is the boolean matrix representing the sparsity pattern of A. (Vector version).
	void WedgeMultTranspose(const Vector & x, Vector & y);
	//! y = Graph(A)^T*x, where Graph(A) is the boolean matrix representing the sparsity pattern of A. (Array<int> version).
	void WedgeMultTranspose(const Array<int> & x, Array<int> & y);

	TopologyTable * GetSubTable(const Array<int> & rows, const Array<int> & cols, Array<int> & marker);

	TopologyTable * Transpose();

	virtual ~TopologyTable();

private:
	template<class T>
	void wedgeMult(const T * x, T * y);

	template<class T>
	void wedgeMultTranspose(const T * x, T * y);
};

//! Multiply two topology tables by looking at orientation only.
TopologyTable * MultOrientation(const TopologyTable & A, const TopologyTable & B);

//! Multiply the sparsity pattern of two TopologyTable and ignore orientation
TopologyTable * MultBoolean(const TopologyTable & A, const TopologyTable & B);

//! Transpose a 1D array in a TopologyTable object. A(i,j) = 1 if partioning[j] = i.
TopologyTable * TransposeOrientation(const Array<int> & j, int nrowsOut);

template<class T>
void TopologyTable::wedgeMult(const T * x, T * y)
{
	const int * i_this = GetI();
	const int * j_this = GetJ();

	const int * jcol = j_this;
	for(int irow(0); irow < height; ++irow, ++y)
	{
		*y = static_cast<T>(0);
		for(const int * end(j_this+i_this[irow+1]); jcol != end; ++jcol)
			*y += x[*jcol];
	}
}

template<class T>
void TopologyTable::wedgeMultTranspose(const T * x, T * y)
{
	const int * i_this = GetI();
	const int * j_this = GetJ();

	const int * jcol = j_this;

	for(int i = 0; i < Width(); ++i)
		y[i] = static_cast<T>(0);

	for(int irow(0); irow < height; ++irow, ++x)
	{
		for(const int * end(j_this+i_this[irow+1]); jcol != end; ++jcol)
			y[*jcol] += *x;
	}

}

#endif /* TOPOLOGYTABLE_HPP_ */
