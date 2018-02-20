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

#ifndef COCHAINPROJECTOR_HPP_
#define COCHAINPROJECTOR_HPP_

#include <memory>
#include <vector>

#include <mfem.hpp>

#include "amge/DofHandler.hpp"
#include "amge/DOFAgglomeration.hpp"
#include "amge/ElementalMatricesContainer.hpp"
#include "linalg/dense/ParELAG_MultiVector.hpp"
#include "topology/Topology.hpp"

namespace parelag
{
//! @class CochainProjector
/**
   \brief Projects from fine space to coarse space, eg from S_h to S_H
*/
class CochainProjector
{
public:
    CochainProjector(AgglomeratedTopology * cTopo,
                     DofHandler * cDof,
                     DofAgglomeration * dofAgg,
                     mfem::SparseMatrix * P);
    virtual ~CochainProjector() = default;

    CochainProjector(CochainProjector const&) = delete;
    CochainProjector(CochainProjector&&) = delete;

    CochainProjector& operator=(CochainProjector const&) = delete;
    CochainProjector& operator=(CochainProjector&&) = delete;

    void CreateDofFunctional(
        AgglomeratedTopology::Entity entity_type,
        int entity_id,
        const MultiVector & localProjector,
        const mfem::SparseMatrix & M_ii);
    void SetDofFunctional(
        AgglomeratedTopology::Entity entity_type,
        int entity_id,
        std::unique_ptr<mfem::DenseMatrix> dof_lin_func );

    //@{ Global Projectors
    // Note res should not point to the same memory area as vFine
    void Project(const MultiVector & vFine, MultiVector & vCoarse);
    void Project(const MultiVector & vFine,
                 MultiVector & vCoarse,
                 MultiVector & res);
    //@}

    void ComputeProjector();
    mfem::SparseMatrix & GetProjectorMatrix();
    const mfem::SparseMatrix & GetProjectorMatrix() const;
    std::unique_ptr<mfem::SparseMatrix> GetIncompleteProjector();

    /**
       Check that Pi is the left-inverse of P, also check that Project()
       does the same thing as the matrix Pi
    */
    void CheckInvariants();

    bool Finalized();

private:

    /// generate random vector, check Pi P v = v
    void CheckInverseProperty();

    std::unique_ptr<mfem::SparseMatrix> assembleInternalProjector(int);

    DofHandlerALG * cDof_;
    DofAgglomeration * dofAgg_;
    mfem::SparseMatrix * P_;

    // Each entry of the array corresponds to a codimensions. Therefore the size is 1 for L2, 2 for Hdiv, etc...
    // If M_agg is the assembled mass matrix on the agglomerated entity (only for interior dofs) and
    // Let localProjector the piece of the interpolation matrix
    // The localCoarseM is given by: localProjector' * M_agg * localProjector
    // The dofLinearFunctional is then: localCoarseM^-1 * localProjector' * M_agg
    std::vector<std::unique_ptr<ElementalMatricesContainer>> dofLinearFunctional_;

    std::unique_ptr<mfem::SparseMatrix> Pi_;
};

}//namespace parelag
#endif /* COCHAINPROJECTOR_HPP_ */
