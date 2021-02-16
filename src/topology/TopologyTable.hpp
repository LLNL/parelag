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

#ifndef TOPOLOGYTABLE_HPP_
#define TOPOLOGYTABLE_HPP_

#include <memory>

#include <mfem.hpp>

#include "elag_typedefs.hpp"
#include "utilities/ParELAG_DataTraits.hpp"
#include "utilities/ParELAG_VectorTraits.hpp"

namespace parelag
{
/// \class TopologyTable.
/// \brief A simple wrapper for mfem::SparseMatrix that adds a few
/// extra functions specific to Topological Tables.
///
/// TopologicalTables are sparse matrices whose entries are plus or
/// minus ones.
///
/// Note: if A and B are TopologyTable objects then Mult(A,B) will be a
/// SerialCSRMatrix object.  Use MultOrientation to obtain a
/// TopologyTable
class TopologyTable : public SerialCSRMatrix
{
public:
    /// Convert a SerialCSRMatrix A into a TopologyTable. Note: ALL
    /// data will be stolen from A. This routine does not check if
    /// some data are different from +-1
    explicit TopologyTable(SerialCSRMatrix & A);

    /// Convert a SerialCSRMatrix *A into a TopologyTable. This
    /// routine does not check if some data are different from +-1
    explicit TopologyTable(std::unique_ptr<SerialCSRMatrix> A);

    /// Build a topology table from the i,j,val arrays. This routine
    /// does not check if some data are different from +-1
    TopologyTable(int *i,int *j,double *data,int m,int n);

    /// Build a topology table from an mfem table. MFEM tables use a
    /// XOR transformation to encode orientation in the j array.
    TopologyTable(const mfem::Table & T);

    /// Destructor
    virtual ~TopologyTable() = default;

    /// Make sure that all entries in the Table are +-1 or 0
    void OrientationTransform();

    /// Eliminate from the sparsity pattern the entries that are less
    /// than tol.
    ///
    /// Note this elimination is done in-place, so no new memory is
    /// allocated, but also no memory is freed.
    void DropSmallEntries(double tol);

    /// y = Graph(A)*x, where Graph(A) is the boolean matrix
    /// representing the sparsity pattern of A.
    template <class VectorT>
    void WedgeMult(const VectorT & x,VectorT & y);

    /// y = Graph(A)^T*x, where Graph(A) is the boolean matrix
    /// representing the sparsity pattern of A. (Vector version).
    template <class VectorT>
    void WedgeMultTranspose(const VectorT & x,VectorT & y);

    /// Compute some sub-table
    std::unique_ptr<TopologyTable> GetSubTable(
        const mfem::Array<int> & rows,const mfem::Array<int> & cols,
        mfem::Array<int> & marker);

    /// Compute the transpose of this table
    std::unique_ptr<TopologyTable> Transpose() const;
};

/// Multiply two topology tables by looking at orientation only.
std::unique_ptr<TopologyTable> MultOrientation(
    const TopologyTable & A,
    const TopologyTable & B );

/// Multiply the sparsity pattern of two TopologyTable and ignore
/// orientation
std::unique_ptr<TopologyTable> MultBoolean(
    const TopologyTable & A,
    const TopologyTable & B );

/// Transpose a 1D array in a TopologyTable object. A(i,j) = 1 if
/// partioning[j] = i.
std::unique_ptr<TopologyTable> TransposeOrientation(
    const mfem::Array<int> & j,
    int nrowsOut );

template<class VectorT>
void TopologyTable::WedgeMult(const VectorT& x, VectorT& y)
{
    using DataT = typename VectorTraits<VectorT>::value_type;
    using IndexT = typename VectorTraits<VectorT>::size_type;

    const DataT * xd = x.GetData();
    DataT * yd = y.GetData();

    const IndexT * i_this = GetI(), * j_this = GetJ();

    for (IndexT irow = 0; irow < height; ++irow)
    {
        DataT val = DataTraits<DataT>::zero();
        for (IndexT jj = i_this[irow]; jj < i_this[irow+1]; ++jj)
            val += xd[j_this[jj]];
        *yd = val;
        ++yd;
    }
}

template<class VectorT>
void TopologyTable::WedgeMultTranspose(const VectorT& x, VectorT& y)
{
    using DataT = typename VectorTraits<VectorT>::value_type;
    using IndexT = typename VectorTraits<VectorT>::size_type;

    const DataT * xd = x.GetData();
    DataT * yd = y.GetData();

    const IndexT * i_this = GetI(), * j_this = GetJ();

    std::fill(yd,yd+width,DataTraits<DataT>::zero());
    for (IndexT irow = 0; irow < height; ++irow)
    {
        for (IndexT jj = i_this[irow]; jj < i_this[irow+1]; ++jj)
            yd[j_this[jj]] += *xd;
        ++xd;
    }

}
}//namespace parelag
#endif /* TOPOLOGYTABLE_HPP_ */
