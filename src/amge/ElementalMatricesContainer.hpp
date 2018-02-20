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

#ifndef ELEMENTALMATRICESCONTAINER_HPP_
#define ELEMENTALMATRICESCONTAINER_HPP_

#include <memory>
#include <vector>

#include <mfem.hpp>

namespace parelag
{
//! @class
/*!
 * @brief a way of storing a matrix in un-assembled form, so we can assemble over subdomains at will
 *
 */
class ElementalMatricesContainer
{
    using iterator = std::vector<std::unique_ptr<mfem::DenseMatrix>>::iterator;
    using const_iterator =
        std::vector<std::unique_ptr<mfem::DenseMatrix>>::const_iterator;
public:
    ElementalMatricesContainer(int nEntities);
    // Deep copy
    ElementalMatricesContainer(const ElementalMatricesContainer & orig);
    virtual ~ElementalMatricesContainer();
    bool Finalized();

    int GetNumEntities()
    {
        return static_cast<int>(emat_.size());
    }

    // To be implemented in the future :)

    std::unique_ptr<mfem::SparseMatrix> GetAsSparseMatrix();
    /*
      mfem::SparseMatrix * AssembleGlobal(DofHandler & domain, DofHandler & range);
      mfem::SparseMatrix * AssembleAgglomerate(Topology::entity entity_type, DofAgglomeration & domain, DofAgglomeration & range);
      mfem::SparseMatrix * AssembleAgglomerate(Topology::entity entity_type, int entity_id, DofAgglomeration & domain, DofAgglomeration & range);
      mfem::SparseMatrix * AssembleAgglomerateII(Topology::entity entity_type, int entity_id, DofAgglomeration & domain, DofAgglomeration & range);
      mfem::SparseMatrix * AssembleAgglomerateIB(Topology::entity entity_type, int entity_id, DofAgglomeration & domain, DofAgglomeration & range);
      mfem::SparseMatrix * AssembleAgglomerateBI(Topology::entity entity_type, int entity_id, DofAgglomeration & domain, DofAgglomeration & range);
      mfem::SparseMatrix * AssembleAgglomerateBB(Topology::entity entity_type, int entity_id, DofAgglomeration & domain, DofAgglomeration & range);
    */

    void SetElementalMatrix(int i, std::unique_ptr<mfem::DenseMatrix> smat);
    void ResetElementalMatrix(int i, std::unique_ptr<mfem::DenseMatrix> smat);
    void SetElementalMatrix(int i, const double & val);
    mfem::DenseMatrix & GetElementalMatrix(int i);

    iterator begin() noexcept { return emat_.begin(); }
    const_iterator begin() const noexcept { return emat_.begin(); }
    iterator end() noexcept { return emat_.end(); }
    const_iterator end() const noexcept { return emat_.end(); }

private:
    std::vector<std::unique_ptr<mfem::DenseMatrix>> emat_;
};
}//namespace parelag
#endif /* ELEMENTALMATRICESCONTAINER_HPP_ */
