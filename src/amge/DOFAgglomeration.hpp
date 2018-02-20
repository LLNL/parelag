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

#ifndef DOFAGGLOMERATION_HPP_
#define DOFAGGLOMERATION_HPP_

#include <memory>
#include <vector>

#include <mfem.hpp>

#include "amge/DofHandler.hpp"
#include "topology/Topology.hpp"

namespace parelag
{
//! @class DofAgglomeration
/**
   @brief Keeps track of how Dofs are aggregated; used in particular
   to make local agglomerate matrices for solving local problems

   Most of this class revolves around AEntity_Dof, which tells you the *fine*
   degrees of freedom that are contained in agglomerated (*coarse*) entities.
   In addition, these are ordered with interior dofs first and shared dofs
   are at the end, which makes solving local problems (e.g. extensions) easier
   to manage.
*/
class DofAgglomeration final
{
    using index_type = int;
    using range_type = std::pair<index_type,index_type>;
    using entity_type = AgglomeratedTopology::Entity;

public:
    explicit DofAgglomeration(
        const std::shared_ptr<AgglomeratedTopology>& topo, DofHandler * dof);

    ~DofAgglomeration() = default;

    DofAgglomeration(DofAgglomeration const&) = delete;
    DofAgglomeration& operator=(DofAgglomeration const&) = delete;

    DofAgglomeration(DofAgglomeration&&) = delete;
    DofAgglomeration& operator=(DofAgglomeration&&) = delete;

    void CheckAdof();

    inline int GetNumberFineEntities(entity_type entity)
    {
        return FineTopology_->GetNumberLocalEntities(entity);
    }

    inline int GetNumberCoarseEntities(entity_type entity)
    {
        return CoarseTopology_->GetNumberLocalEntities(entity);
    }

    inline int GetNumberAgglomerateInternalDofs(
        entity_type entity, index_type entity_id) const
    {
        auto range = GetAgglomerateInternalDofRange(entity, entity_id);
        return range.second - range.first;
    }

    DofHandler * GetDofHandler() const noexcept { return Dof_; }

    // FIXME (trb 12/16/15): So we allow access via a pointer, which
    // allows the caller to do whatever they want to this thing,
    // including call, say, LoseData(), rendering it useless, or
    // moving the pointer to a new object, AND we allow access as a
    // const reference, which prevents both of these travesties. Kinda
    // sending mixed messages...
    //mfem::SparseMatrix * GetAEntityDof(entity_type entity)
    //{
    //    return AEntity_Dof_[entity].get();
    //}

    const mfem::SparseMatrix & GetAEntityDof(entity_type entity) const
    {
        return *(AEntity_Dof_[entity]);
    }

    /// Get the range of internal dofs for the given agglomerated entity
    range_type GetAgglomerateInternalDofRange(
        entity_type entity, index_type entity_id) const;

    void GetAgglomerateInternalDofRange(
        entity_type entity, index_type entity_id, int & begin, int & end) const;

    /// Get the range of boundary dofs for the given agglomerated entity
    range_type GetAgglomerateBdrDofRange(
        entity_type entity, index_type entity_id) const;

    void GetAgglomerateBdrDofRange(
        entity_type entity, index_type entity_id, int & begin, int & end) const;

    /// Get the range of dofs for the given agglomerated entity
    range_type GetAgglomerateDofRange(
        entity_type entity, index_type entity_id) const;

    void GetAgglomerateDofRange(
        entity_type entity, index_type entity_id, int & begin, int & end) const;

    void GetViewAgglomerateInternalDofGlobalNumering(
        entity_type entity, index_type entity_id, mfem::Array<int> & gdofs) const;

    void GetViewAgglomerateBdrDofGlobalNumering(
        entity_type entity, index_type entity_id, mfem::Array<int> & gdofs) const;

    void GetViewAgglomerateDofGlobalNumering(
        entity_type entity, index_type entity_id, mfem::Array<int> & gdofs) const;

    // FIXME (trb 06/28/16): This function is very unsafe
    int * GetAgglomerateInternalDofGlobalNumering(
        entity_type entity, index_type entity_id, int * gdofs);

    std::unique_ptr<MultiVector> DistributeGlobalMultiVector(
        entity_type entity, const MultiVector & vg);

    std::unique_ptr<mfem::Vector> DistributeGlobalVector(
        entity_type entity, mfem::Vector & vg);

    /*
     * input:
     * -- entity: 0 -> Element, 1 -> Facet, 2 -> Ridge, 3 -> Peak
     * -- M_e: entity matrix of size nrDof_range x nrDof_domain
     * -- rangeSpace: index in the dof array corresponding to the
     *    range of the operator
     *
     * -- domainSpace: index in the dof array corresponding to the
     *    domain of the operator
     *
     * output:
     * -- Agglomerated Entity matrix of size nADof_range x nADof_domain
     *
     *  Description:
     *  Given M_e = diag( m_ei ) where:
     *  -- m_ei is the local matrix on entity i
     *  computes M_E =  diag( m_EJ ) where:
     *  -- m_EJ is the local matrix on agglomerated Entity J.
     *
     *  Implementation: M_E = RAP(rDof_ADof_range, M_e,
     *  rDof_ADof_domain), where rDof_RDof is the +1,-1 matrix that
     *  maps the set of repeated dof for each entity in FineTopology_ into
     *  the set Agglomerated dof for each Agglomerated entity in
     *  CoarseTopology_.
     */
    friend std::unique_ptr<mfem::SparseMatrix> AssembleAgglomerateMatrix(
        entity_type entity, mfem::SparseMatrix & M_e,
        DofAgglomeration * range, DofAgglomeration * domain);

    friend std::unique_ptr<mfem::SparseMatrix>
    AssembleAgglomerateRowsGlobalColsMatrix(
        entity_type entity, mfem::SparseMatrix & M_e,
        DofAgglomeration * range, DofHandler * domain);

    friend std::unique_ptr<mfem::SparseMatrix> DistributeProjector(
        entity_type entity, const mfem::SparseMatrix & P_t,
        DofHandler * range, DofAgglomeration * domain);

    /*
     * input:
     * -- entity: 0 -> Element, 1 -> Facet, 2 -> Ridge, 3 -> Peak
     * -- M_a: Agglomerated Entity matrix of size nADof_range x nADof_domain
     * -- rangeSpace: index in the dof array corresponding to the
     *    range of the operator
     * -- domainSpace: index in the dof array corresponding to the
     *    domain of the operator
     *
     * output:
     * -- M_g global matrix
     *
     *  Description:
     *  Given M_a = diag( m_ai ) where:
     *  -- m_ai is the local sparse matrix on agglomerate i
     *  computes M_g.
     *
     *  Implementation: M_g = Dof_ADof_range * M_a, ADof_Dof_domain
     */
    friend std::unique_ptr<mfem::SparseMatrix> Assemble(
        entity_type entity, const mfem::SparseMatrix & M_a,
        DofAgglomeration * range, DofAgglomeration * domain);

    /*
     * input:
     * -- entity: 0 -> Element, 1 -> Facet, 2 -> Ridge, 3 -> Peak
     * -- M_g: entity matrix of size nDof_range x nDof_domain
     * -- rangeSpace: index in the dof array corresponding to the
     *    range of the operator
     * -- domainSpace: index in the dof array corresponding to the
     *    domain of the operator
     *
     * output:
     * -- Agglomerated Entity matrix of size nADof_range x nADof_domain
     *
     *  Description:
     *  Given D_g the fully assembled finite element matrix
     *  computes D_E =  diag( d_EJ ) where:
     *  -- d_EJ is the local matrix on agglomerated Entity J.
     *
     *  Implementation: D_E = RAP(Dof_ADof_range', D_G,
     *  Dof_ADof_domain) where Dof_ADof is the +1,-1 matrix that maps
     *  the set of global dof into the set Agglomerate dof for each
     *  Agglomerated entity in CoarseTopology_.
     *
     */
    friend std::unique_ptr<mfem::SparseMatrix> DistributeAgglomerateMatrix(
        entity_type entity, const mfem::SparseMatrix & D_g,
        DofAgglomeration * range, DofAgglomeration * domain);


private:
    // Reorder the j and val arrays of a so that the entries are
    // ordered increasignly with respect to weights.
    //
    // It returns for each row the number of occurrencies of minWeight
    int reorderRow(
        int nentries,int * j,double * a,int minWeight,int maxWeight,
        const std::vector<int> & weights);

    std::shared_ptr<AgglomeratedTopology> FineTopology_;
    std::shared_ptr<AgglomeratedTopology> CoarseTopology_;

    DofHandler * Dof_;

    // The index in this array represents the type of entity (0 =
    // ELEMENT, 1 = FACET, 2 = RIDGE, 3 = PEAK).
    //
    // Each matrix has for each row the ids and the orientation of the
    // dofs in AgglomeratedEntity i.
    std::vector<std::unique_ptr<mfem::SparseMatrix>> AEntity_Dof_;

    // The ADof_Dof_ matrix shares data with AEntity_Dof_; This is
    // it's row_ptr array, which is not shared.
    std::vector<std::vector<int>> ADof_Dof_I_;
    std::vector<std::unique_ptr<mfem::SparseMatrix>> ADof_Dof_;
    std::vector<std::unique_ptr<mfem::SparseMatrix>> ADof_rDof_;

    std::vector<std::vector<int>> AE_nInternalDof_;

    mutable std::vector<int> dofMapper_;

};
}//namespace parelag
#endif /* DOFAGGLOMERATION_HPP_ */
