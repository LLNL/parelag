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

#ifndef DOFHANDLER_HPP_
#define DOFHANDLER_HPP_

#include <array>
#include <memory>
#include <vector>

#include <mfem.hpp>

#include "ParELAG_Constants.hpp"

#include "structures/SharingMap.hpp"
#include "topology/Topology.hpp"

namespace parelag
{
//! @class
/*!
 * @brief Handles parallel numbering, including entityTrueEntity, dofTrueDof, etc.
 *
 */
class DofHandler
{
public:

    typedef AgglomeratedTopology::Entity entity;
    enum {SCALARSPACE = 0, VECTORSPACE = 1};

    DofHandler(MPI_Comm comm, size_t maxCodimensionBaseForDof, size_t nDim);

    /// Constructor that assumes unique owership of user-provided
    /// entity_dof tables. (Be sure to std::move the vector in)
    DofHandler(
        MPI_Comm comm, size_t maxCodimensionBaseForDof, size_t nDim,
        std::vector<std::unique_ptr<const mfem::SparseMatrix>>&& entity_dof_);

    virtual ~DofHandler();

    int SpaceType()
    {
        return (maxCodimensionBaseForDof == 0 ||
                maxCodimensionBaseForDof == nDim) ? SCALARSPACE : VECTORSPACE;
    }

    virtual void BuildEntityDofTables() = 0;
    virtual void BuildEntityDofTable(entity type) = 0;

    // bndrAttributesMarker boolean array [input]
    //
    // dofMarker: boolean array (size GetNDofs): selected dofs are
    // marked with ones.
    //
    // return how many dofs were marked.
    virtual int MarkDofsOnSelectedBndr(
        const mfem::Array<int> & bndrAttributesMarker,
        mfem::Array<int> & dofMarker)
        const = 0;

    const mfem::SparseMatrix & GetEntityDofTable(entity type) const;

    int GetEntityNDof(entity type, int ientity) const
    {
        return GetEntityDofTable(type).RowSize(ientity);
    }

    /// Think about entity_dof. It has some columns with only one nonzero, which
    /// means that dof is contained in one (and only one) entity. It also has
    /// some columns with several nonzeros, which means the dof is shared across
    /// more entities. This routine takes entity_dof and breaks apart the shared
    /// dofs, so that in the returned entity_rdof matrix, every column has
    /// exactly one nonzero.
    const mfem::SparseMatrix & GetEntityRDofTable(entity type) const;

    // NOTE this routine assumes that rdof relative to the same entity
    // are contiguous.
    const mfem::SparseMatrix & GetrDofDofTable(entity type) const;

    // NOTE this routine returns an array of rdof that is contiguous.
    virtual void GetrDof(entity type, int ientity,
                         mfem::Array<int> & dofs) const;

    virtual void GetDofs(entity type, int ientity,
                         mfem::Array<int> & dofs);

    int GetNumberEntities(entity type) const;

    size_t GetMaxCodimensionBaseForDof() const;

    virtual int GetNumberInteriorDofs(entity type);

    virtual void GetInteriorDofs(entity type, int ientity,
                                 mfem::Array<int> & dofs) = 0;

    virtual void GetDofsOnBdr(entity type, int ientity,
                              mfem::Array<int> & dofs) = 0;

    std::unique_ptr<mfem::SparseMatrix> AllocGraphElementMass(entity type);

    /// Number of vertices for jform[0] (H1), edges for jform[1] (Hcurl), etc.
    inline int GetNDofs() const
    {
        return entity_dof[0]->Width();
    }

    /// this needs to be distinguished from GetNDofs and documented better
    inline int GetNrDofs(entity entity_type) const
    {
        return entity_dof[entity_type]->NumNonZeroElems();
    }

    void Average(entity entity_type, const MultiVector & repV,
                 MultiVector & globV);

    void AssembleGlobalMultiVector(
        entity type, const MultiVector & local, MultiVector & global);

    void AssembleGlobalMultiVectorFromInteriorDofs(
        entity type, const MultiVector & local, MultiVector & global);

    // Friend functions
    friend std::unique_ptr<mfem::SparseMatrix> Assemble(
        entity entity_type, const mfem::SparseMatrix & M_e,
        DofHandler & range, DofHandler & domain);

    friend std::unique_ptr<mfem::SparseMatrix> Distribute(
        entity entity_type, const mfem::SparseMatrix & M_g,
        DofHandler & range, DofHandler & domain);

    bool Finalized()
    {
        return finalized[0];
    }

    virtual const SharingMap & GetEntityTrueEntity(int codim) const = 0;

    SharingMap & GetDofTrueDof()
    {
        return dofTrueDof;
    }

    const SharingMap & GetDofTrueDof() const
    {
        return dofTrueDof;
    }

    void CheckInvariants() const;

protected:

    virtual int getNumberOf(int type) const = 0;
    virtual void checkMyInvariants() const = 0;

    std::array<bool,MAX_CODIMENSION> finalized { {false,false,false,false} };
    size_t maxCodimensionBaseForDof;
    size_t nDim;

    std::vector<std::unique_ptr<const mfem::SparseMatrix>> entity_dof;
    mutable std::vector<std::unique_ptr<const mfem::SparseMatrix>> rDof_dof;
    mutable std::vector<std::unique_ptr<const mfem::SparseMatrix>> entity_rdof;

    SharingMap dofTrueDof;
};

class DofHandlerFE final: public DofHandler
{
public:

    typedef DofHandler super;
    typedef super::entity entity;

    DofHandlerFE(MPI_Comm comm,
                 mfem::FiniteElementSpace * fespace_,
                 size_t maxCodimensionBaseForDof_);

    ~DofHandlerFE() = default;

    DofHandlerFE(DofHandlerFE const&) = delete;
    DofHandlerFE& operator=(DofHandlerFE const&) = delete;

    DofHandlerFE(DofHandlerFE&&) = delete;
    DofHandlerFE& operator=(DofHandlerFE&&) = delete;

    virtual void BuildEntityDofTables() override;

    virtual void BuildEntityDofTable(entity type) override;

    virtual int MarkDofsOnSelectedBndr(
        const mfem::Array<int> & bndrAttributesMarker,
        mfem::Array<int> & dofMarker)
        const override;

    virtual void GetInteriorDofs(entity type,
                                 int ientity,
                                 mfem::Array<int> & dofs) override;

    virtual void GetrDof(entity type, int ientity,
                         mfem::Array<int> & dofs) const override
    {
        super::GetrDof(type, ientity, dofs);
    }

    virtual void GetDofsOnBdr(entity, int, mfem::Array<int> &) override
    {
        PARELAG_NOT_IMPLEMENTED();
    }

    virtual const SharingMap & GetEntityTrueEntity(int) const override
    {
        PARELAG_NOT_IMPLEMENTED();
        // Silence compiler warning about reaching end of non-void function
        SharingMap * m = nullptr;
        return *m;
    }

protected:
    virtual int getNumberOf(int type) const override;
    virtual void checkMyInvariants() const override;

private:

    void getElementDof(int entity_id, int * dofs, double * orientation) const;

    void getFacetDof(int entity_id, int * dofs, double * orientation) const;

    void getRidgeDof(int entity_id, int * dofs, double * orientation) const;

    void getPeakDof(int entity_id, int * dofs, double * orientation) const;

    void getDofForEntity(entity entity_type, int entity_id, int * dofs,
                         double * orientation) const;

    int getNumberOfElements() const;

    int getNumberOfFacets() const;

    int getNumberOfRidges() const;

    int getNumberOfPeaks() const;


    int getNumberOfDofForElement(int entity_id);

    int getNumberOfDofForFacet(int entity_id);

    int getNumberOfDofForRidge(int entity_id);

    int getNumberOfDofForPeak(int entity_id);

    int getNumberOfDofForEntity(int entity_type, int entity_id);

    mfem::FiniteElementSpace * fespace;
};


class DofHandlerALG : public DofHandler
{
public:

    enum dof_type{ Empty = 0x0, RangeTSpace = 0x1, NullSpace = 0x2};

    DofHandlerALG(size_t maxCodimensionBaseForDof,
                  const std::shared_ptr<AgglomeratedTopology>& topology);

    DofHandlerALG(int * entity_HasInteriorDofs, size_t maxCodimensionBaseForDof,
                  const std::shared_ptr<AgglomeratedTopology>& topology);

    virtual ~DofHandlerALG() override;

    virtual int GetNumberInteriorDofs(entity type) override;

    int GetNumberInteriorDofs(entity type, int entity_id);

    virtual void BuildEntityDofTables() override;

    virtual void BuildEntityDofTable(entity type) override;

    void AllocateDofTypeArray(int maxSize);

    void SetDofType(int dof, dof_type type);

    virtual int MarkDofsOnSelectedBndr(
        const mfem::Array<int> & bndrAttributesMarker,
        mfem::Array<int> & dofMarker) const override;

    std::unique_ptr<mfem::SparseMatrix>
    GetEntityNullSpaceDofTable(entity type) const;

    std::unique_ptr<mfem::SparseMatrix>
    GetEntityRangeTSpaceDofTable(entity type) const;

    std::unique_ptr<mfem::SparseMatrix>
    GetEntityInternalDofTable(entity type) const;

    void SetNumberOfInteriorDofsNullSpace(
        entity type, int entity_id, int nLocDof);

    void SetNumberOfInteriorDofsRangeTSpace(
        entity type, int entity_id, int nLocDof);

    virtual void GetInteriorDofs(
        entity type, int ientity, mfem::Array<int> & dofs) override;

    virtual void GetDofsOnBdr(
        entity type, int ientity, mfem::Array<int> & dofs) override;

    virtual const SharingMap & GetEntityTrueEntity(int codim) const override
    {
        return Topology_->EntityTrueEntity(codim);
    }

protected:
    virtual int getNumberOf(int type) const override;
    virtual void checkMyInvariants() const override;

private:
    void computeOffset(entity type);
    void buildPeakDofTable();
    void buildRidgeDofTable();
    void buildFacetDofTable();
    void buildElementDofTable();

    void EntityDofFillIAssembly(entity entity_big, entity entity_small,
                                int * i_assembly);
    void EntityDofFillJ(entity entity_big, entity entity_small,
                        int * i_assembly, int * j);

    std::array<int,MAX_CODIMENSION> entity_hasInteriorDofs;
    std::array<int,MAX_CODIMENSION> entityType_nDofs;

    std::shared_ptr<AgglomeratedTopology> Topology_;

    std::array<std::vector<int>,MAX_CODIMENSION> entity_NumberOfInteriorDofsNullSpace;
    std::array<std::vector<int>,MAX_CODIMENSION> entity_NumberOfInteriorDofsRangeTSpace;
    std::array<std::vector<int>,MAX_CODIMENSION> entity_InteriorDofOffsets;

    // This variable will contain nDofs when finalized[ELEMENT] = true.
    int nDofs;

    // Type of Global Dof:  RangeTSpace NullSpace.
    mfem::Array<int> DofType;
};


class DofHandlerSCRATCH final : public DofHandler
{
public:

    DofHandlerSCRATCH(
        MPI_Comm comm,size_t maxCodimensionBaseForDof,size_t nDim,
        std::vector<std::unique_ptr<const mfem::SparseMatrix>>&& entity_dof_):
        DofHandler(comm, maxCodimensionBaseForDof, nDim, std::move(entity_dof_))
    {
    }

    virtual void BuildEntityDofTables() override
    {
        PARELAG_NOT_IMPLEMENTED();
    }

    virtual void BuildEntityDofTable(entity) override
    {
        PARELAG_NOT_IMPLEMENTED();
    }

    virtual int MarkDofsOnSelectedBndr(
        mfem::Array<int> const&, mfem::Array<int> &) const override
    {
        PARELAG_NOT_IMPLEMENTED();
        return 0;
    }

    virtual void GetInteriorDofs(entity, int, mfem::Array<int> &) override
    {
        PARELAG_NOT_IMPLEMENTED();
    }

    virtual void GetDofsOnBdr(entity, int, mfem::Array<int> &) override
    {
        PARELAG_NOT_IMPLEMENTED();
    }

    virtual const SharingMap & GetEntityTrueEntity(int) const override
    {
        PARELAG_NOT_IMPLEMENTED();
        // Silence compiler warning about reaching end of non-void function
        SharingMap * m = nullptr;
        return *m;
    }

protected:

    virtual int getNumberOf(int) const override
    {
        PARELAG_NOT_IMPLEMENTED();
        return 0;
    }

    virtual void checkMyInvariants() const override
    {
        PARELAG_NOT_IMPLEMENTED();
    }
};
}//namespace parelag
#endif /* DOFHANDLER_HPP_ */
