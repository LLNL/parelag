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

#include <numeric>

#include "DofHandler.hpp"

#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "linalg/utilities/ParELAG_SubMatrixExtraction.hpp"
#include "linalg/dense/ParELAG_MultiVector.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

DofHandler::DofHandler(
    MPI_Comm comm,size_t maxCodimensionBaseForDof_,size_t nDim_)
    : maxCodimensionBaseForDof(maxCodimensionBaseForDof_),
      nDim(nDim_),
      entity_dof(maxCodimensionBaseForDof_+1),
      rDof_dof(maxCodimensionBaseForDof_+1),
      entity_rdof(maxCodimensionBaseForDof_+1),
      dofTrueDof(comm)
{}

DofHandler::DofHandler(
    MPI_Comm comm,size_t maxCodimensionBaseForDof_,size_t nDim_,
    std::vector<unique_ptr<const SparseMatrix>>&& entity_dof_)
    : maxCodimensionBaseForDof(maxCodimensionBaseForDof_),
      nDim(nDim_),
      entity_dof(std::move(entity_dof_)),
      rDof_dof(maxCodimensionBaseForDof_+1),
      entity_rdof(maxCodimensionBaseForDof_+1),
      dofTrueDof(comm)
{
    for (size_t i = 0; i < maxCodimensionBaseForDof+1; ++i)
        finalized[i] = true;
}

DofHandler::~DofHandler()
{
}

const SparseMatrix & DofHandler::GetEntityDofTable(entity type) const
{

    PARELAG_TEST_FOR_EXCEPTION(
        !finalized[type],
        std::logic_error,
        "DofHandler::GetEntityDofTable(): "
        "Please call BuildEntityDofTables first.");

    PARELAG_TEST_FOR_EXCEPTION(
        type > (int) maxCodimensionBaseForDof,
        std::logic_error,
        "DofHandler::GetEntityDofTable(): Not a valid entity type.");

    return *entity_dof[type];
}

const SparseMatrix & DofHandler::GetEntityRDofTable(entity type) const
{
    if (!entity_rdof[type])
    {
        const int nrows = entity_dof[type]->Size();
        const int ncols = entity_dof[type]->NumNonZeroElems();
        const int nnz   = entity_dof[type]->NumNonZeroElems();
        int * i = new int[nrows+1];
        int * j = new int[nnz];
        double * a = new double[nnz];

        // Copying allows proper management of the memory for the arrays
        // to remain within the power of the SparseMatrix class in MFEM.
        // This can be avoided if MFEM decides to separate the ownership of
        // the I and J arrays.
        std::copy(entity_dof[type]->GetI(), entity_dof[type]->GetI() + nrows+1, i);

        std::fill(a, a+nnz, 1.0);
        Array<int> dofs;
        for (int irow(0); irow < nrows; ++irow)
        {
            dofs.MakeRef(j+i[irow], i[irow+1] - i[irow]);
            GetrDof(type, irow, dofs);
        }

        entity_rdof[type] = make_unique<const SparseMatrix>(i,j,a,nrows,ncols);
        CheckMatrix(*entity_rdof[type]);
    }

    return *entity_rdof[type];
}

const SparseMatrix & DofHandler::GetrDofDofTable(entity type) const
{
    if (!rDof_dof[type])
    {
        const SparseMatrix & ent_dof(GetEntityDofTable(type));
        const int nnz = ent_dof.NumNonZeroElems();
        int * i_Rdof_dof = new int[nnz+1];
        for (int kk(0); kk < nnz+1; ++kk)
            i_Rdof_dof[kk] = kk;

        // Copying allows proper management of the memory for the arrays
        // to remain within the power of the SparseMatrix class in MFEM.
        // This can be avoided if MFEM decides to separate the ownership of
        // the I and J arrays.
        int * j_Rdof_dof = new int[nnz];
        std::copy(ent_dof.GetJ(), ent_dof.GetJ() + nnz, j_Rdof_dof);

        // The const_cast here does NOT lead to a de facto loss of constness, since
        // the obtained SparseMatrix object is const. This is a legitimate use of a
        // const_cast with the only purpose to simply work with (reasonable) limitations in the
        // interface of the constructors of SparseMatrix in MFEM. Namely, the constructors
        // only take int * arguments independently of whether a const or non-const object
        // is being constructed. Thus, here the const_cast should NOT have any
        // negative semantic consequences and only deteriorates the syntactic aesthetics.
        rDof_dof[type] = make_unique<const SparseMatrix>(i_Rdof_dof,
                                                         j_Rdof_dof,
                                                         const_cast<double *>(ent_dof.GetData()),
                                                         nnz,
                                                         ent_dof.Width(), true, false, false);
        CheckMatrix(*rDof_dof[type]);
    }
    return *rDof_dof[type];
}

void DofHandler::GetrDof(entity type, int ientity, Array<int> & dofs) const
{
    const SparseMatrix & entityDof = GetEntityDofTable(type);
    const int * I = entityDof.GetI();

    int start = I[ientity];
    int stop  = I[ientity+1];
    dofs.SetSize(stop - start);
    int * d = dofs.GetData();

    for (int i(start); i < stop; ++i)
        *(d++) = i;
}

int DofHandler::GetNumberEntities(entity type) const
{
    if (finalized[type])
        return GetEntityDofTable(type).Size();
    else
        return getNumberOf(static_cast<int>(type));
}

size_t DofHandler::GetMaxCodimensionBaseForDof() const
{
    return maxCodimensionBaseForDof;
}

int DofHandler::GetNumberInteriorDofs(entity type)
{
    int n(0);
    Array<int> dofs;

    for (int ientity(0); ientity < getNumberOf(type); ++ientity)
    {
        GetInteriorDofs(type, ientity, dofs);
        n += dofs.Size();
    }

    return n;

}

void DofHandler::GetDofs(entity type, int ientity, Array<int> & dofs)
{
    Vector discard;
    entity_dof[type]->GetRow(ientity, dofs, discard);
}


unique_ptr<SparseMatrix> DofHandler::AllocGraphElementMass(entity type)
{
    const SparseMatrix & my_entity_rdof = GetEntityRDofTable(type);
    unique_ptr<SparseMatrix> rdof_entity{Transpose(my_entity_rdof)};
    return unique_ptr<SparseMatrix>{Mult(*rdof_entity,my_entity_rdof)};
}

void DofHandler::Average(entity entity_type,
                         const MultiVector & repV,
                         MultiVector & globV)
{
    const SparseMatrix & rDof_Dof = GetrDofDofTable(entity_type);
    Vector eta(rDof_Dof.Width());
    eta = 0.0;
    const int * J = rDof_Dof.GetJ();
    int nnz = rDof_Dof.NumNonZeroElems();
    const int * Jend = J+nnz;

    for (; J < Jend; ++J)
        ++eta(*J);

    MatrixTTimesMultiVector(rDof_Dof, repV, globV);
    globV.InverseScale(eta);
}

void DofHandler::AssembleGlobalMultiVector(entity type, const MultiVector & local, MultiVector & global)
{
    Array<int> dofs;
    entity itype(type);
    for (size_t i(type); i < maxCodimensionBaseForDof; ++i)
    {
        itype = static_cast<entity>(i);
        for (int ientity(0); ientity < GetNumberEntities(itype); ++ientity)
        {
            GetInteriorDofs(itype, ientity,dofs);
            global.AddSubMultiVector(dofs, local);
        }
    }
}

void DofHandler::AssembleGlobalMultiVectorFromInteriorDofs(entity type, const MultiVector & local, MultiVector & global)
{
    Array<int> dofs;
    Vector local_view;
    double * data = local.GetData();
    for (int ientity(0); ientity < GetNumberEntities(type); ++ientity)
    {
        GetInteriorDofs(type, ientity,dofs);
        local_view.SetDataAndSize(data, dofs.Size());
        global.AddSubMultiVector(dofs, local_view);
        data += dofs.Size();
    }
}

/**
   This is orphan code, never called?
*/
void DofHandler::CheckInvariants() const
{
    for (size_t codim = 0; codim <= maxCodimensionBaseForDof; ++codim)
        PARELAG_TEST_FOR_EXCEPTION(
            !finalized[codim],
            std::logic_error,
            "DofHandler::CheckInvariants(): DofHandler not finalized.");

    for (size_t codim = 0; codim <= maxCodimensionBaseForDof; ++codim)
    {
        PARELAG_TEST_FOR_EXCEPTION(
            entity_dof[codim]->Size() != getNumberOf(codim),
            std::runtime_error,
            "DofHandler::CheckInvariants(): Not matching sizes entity_dof.");

        PARELAG_TEST_FOR_EXCEPTION(
            entity_rdof[codim]->Size() != getNumberOf(codim),
            std::runtime_error,
            "DofHandler::CheckInvariants. Not Matching sizes entity_rdof.");
    }

    checkMyInvariants();
}

// RAP
unique_ptr<SparseMatrix> Assemble(
    DofHandler::entity entity_type, const SparseMatrix & M_e,
    DofHandler & range, DofHandler & domain)
{
    const SparseMatrix & rng_rdofdof = range.GetrDofDofTable(entity_type);
    const SparseMatrix & dom_rdofdof = domain.GetrDofDofTable(entity_type);

    auto range_dof_rdof = ToUnique(Transpose(rng_rdofdof));
    auto rt_Me = ToUnique(Mult(*range_dof_rdof, M_e));
    return ToUnique(Mult(*rt_Me,dom_rdofdof));
}

//RAP other way around
unique_ptr<SparseMatrix> Distribute(
    DofHandler::entity entity_type,
    const SparseMatrix & M_g,
    DofHandler & range,
    DofHandler & domain)
{
    PARELAG_TEST_FOR_EXCEPTION(
        true,
        std::runtime_error,
        "Distribute(): This method should not be called!");
    const SparseMatrix & range_rdof_dof = *(range.rDof_dof[entity_type]);

    auto r_Mg = ToUnique(Mult(range_rdof_dof, M_g));
    auto domain_dof_rdof = ToUnique(Transpose(*domain.rDof_dof[entity_type]));
    return ToUnique(Mult(*r_Mg, *domain_dof_rdof));
}



//--------------------------------------------------------------------------------------
DofHandlerFE::DofHandlerFE(MPI_Comm comm,
                           FiniteElementSpace * fespace_,
                           size_t maxCodimensionBaseForDof):
    DofHandler(comm, maxCodimensionBaseForDof, fespace_->GetMesh()->Dimension()),
    fespace(fespace_)
{
    ParMesh * pmesh = dynamic_cast<ParMesh *>(fespace->GetMesh());
    elag_assert(pmesh);
    auto pfes = make_unique<ParFiniteElementSpace>(pmesh, fespace->FEColl());
    dofTrueDof.SetUp(pfes.get(), 1);
}

int DofHandlerFE::MarkDofsOnSelectedBndr(
    const Array<int> & bndrAttributesMarker,
    Array<int> & dofMarker) const
{
    dofMarker = 0;
    fespace->GetEssentialVDofs(bndrAttributesMarker, dofMarker);

    int nmarked(0);
    int * end = dofMarker.GetData() + dofMarker.Size();

    for (int * it = dofMarker.GetData(); it != end; ++it)
        if (*it)
        {
            *it = 1;
            ++nmarked;
        }

    return nmarked;
}

void DofHandlerFE::GetInteriorDofs(entity type, int ientity, Array<int> & dofs)
{
    if (type == AgglomeratedTopology::ELEMENT)
    {
        fespace->GetElementInteriorDofs(ientity, dofs);
        return;
    }

    PARELAG_ASSERT(type == AgglomeratedTopology::ELEMENT)
}


void DofHandlerFE::BuildEntityDofTables()
{
    //element_dof facet_dof and ridge_dof peak_dof
    for (size_t entity_type=0; entity_type < maxCodimensionBaseForDof+1; ++entity_type)
        BuildEntityDofTable(static_cast<entity>(entity_type));
}

void DofHandlerFE::BuildEntityDofTable(entity entity_type)
{
    if (finalized[entity_type])
        return;

//              std::cout<<"entity_dof for entity " << entity_type << std::endl;
    int nEntities = getNumberOf(entity_type);
    int * i = new int[nEntities+1]; i[0] = 0;

    int * i_it = i+1;
    int nnz(0);
    for (int ientity(0); ientity < nEntities; ++ientity, ++i_it)
    {
        nnz += getNumberOfDofForEntity(entity_type, ientity);
        *i_it = nnz;
    }

    int * j = new int[nnz];
    double * val = new double[nnz];

    int * j_it = j;
    double * val_it = val;
    int offset(0);

    for (int ientity(0); ientity < nEntities; ++ientity)
    {
        offset = i[ientity];
        getDofForEntity(static_cast<entity>(entity_type), ientity, j_it + offset, val_it + offset);
    }

    entity_dof[entity_type] = make_unique<const SparseMatrix>(
        i,j,val,nEntities,fespace->GetNDofs());
    CheckMatrix(*entity_dof[entity_type]);

    finalized[entity_type] = true;

}

void DofHandlerFE::getElementDof(int entity_id,
                                 int * dofs,
                                 double * orientation) const
{
    Array<int> sdofs;
    fespace->GetElementDofs(entity_id, sdofs);

    int ind;
    for (int i(0); i < sdofs.Size(); ++i)
    {
        ind = sdofs[i];
        if (ind < 0)
        {
            dofs[i] = -1-ind;
            orientation[i] = -1.;
        }
        else
        {
            dofs[i] = ind;
            orientation[i] = 1.;
        }
    }

}

void DofHandlerFE::getFacetDof(
    int entity_id, int * dofs, double * orientation) const
{
    Array<int> sdofs;
    switch (nDim)
    {
    case 2:
        fespace->GetEdgeDofs(entity_id, sdofs);
        break;
    case 3:
        fespace->GetFaceDofs(entity_id, sdofs);
        break;
    default:
        const bool ndim_not_supported = true;
        PARELAG_TEST_FOR_EXCEPTION(
            ndim_not_supported,
            std::runtime_error,
            "DofHandlerFE: getFacetDof(...): "
            "nDim = " << nDim << " is not supported.");
    }

    int ind;
    for (int i(0); i < sdofs.Size(); ++i)
    {
        ind = sdofs[i];
        if (ind < 0)
        {
            dofs[i] = -1-ind;
            orientation[i] = -1.;
        }
        else
        {
            dofs[i] = ind;
            orientation[i] = 1.;
        }
    }

}

void DofHandlerFE::getRidgeDof(
    int entity_id, int * dofs, double * orientation) const
{
    Array<int> sdofs;
    switch (nDim)
    {
    case 2:
        fespace->GetVertexDofs(entity_id, sdofs);
        break;
    case 3:
        fespace->GetEdgeDofs(entity_id, sdofs);
        break;
    default:
        const bool ndim_not_supported = true;
        PARELAG_TEST_FOR_EXCEPTION(
            ndim_not_supported,
            std::runtime_error,
            "DofHandlerFE: getRidgeDof(...): "
            "nDim = " << nDim << " is not supported.");
    }

    int ind;
    for (int i(0); i < sdofs.Size(); ++i)
    {
        ind = sdofs[i];
        if (ind < 0)
        {
            dofs[i] = -1-ind;
            orientation[i] = -1.;
        }
        else
        {
            dofs[i] = ind;
            orientation[i] = 1.;
        }
    }
}

void DofHandlerFE::getPeakDof(
    int entity_id, int * dofs, double * orientation) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        nDim < 3,
        std::runtime_error,
        "DofHandlerFE: getRidgeDof(...): "
        "nDim = " << nDim << " < 3 is not supported.");

    Array<int> sdofs;
    fespace->GetVertexDofs(entity_id, sdofs);

    int ind;
    for (int i(0); i < sdofs.Size(); ++i)
    {
        ind = sdofs[i];
        if (ind < 0)
        {
            dofs[i] = -1-ind;
            orientation[i] = -1.;
        }
        else
        {
            dofs[i] = ind;
            orientation[i] = 1.;
        }
    }
}


void DofHandlerFE::getDofForEntity(
    entity type, int entity_id, int * dofs, double * orientation) const
{
    switch (type)
    {
    case AgglomeratedTopology::ELEMENT:
        getElementDof(entity_id, dofs, orientation);
        break;
    case AgglomeratedTopology::FACET:
        getFacetDof(entity_id, dofs, orientation);
        break;
    case AgglomeratedTopology::RIDGE:
        getRidgeDof(entity_id, dofs, orientation);
        break;
    case AgglomeratedTopology::PEAK:
        getPeakDof(entity_id, dofs, orientation);
        break;
    default:
        const bool bad_entity_type = true;
        PARELAG_TEST_FOR_EXCEPTION(
            bad_entity_type,
            std::runtime_error,
            "DofHandlerFE::getDofForEntity(...): Wrong entity type.");
    }
}

int DofHandlerFE::getNumberOfElements() const
{
    return fespace->GetMesh()->GetNE();
}

int DofHandlerFE::getNumberOfFacets() const
{
    switch (nDim)
    {
    case 1:
        return fespace->GetMesh()->GetNV();
    case 2:
        return fespace->GetMesh()->GetNEdges();
    case 3:
        return fespace->GetMesh()->GetNFaces();
    default:
        return -1;
    }
}
int DofHandlerFE::getNumberOfRidges() const
{
    switch (nDim)
    {
    case 1:
        return -1;
    case 2:
        return fespace->GetMesh()->GetNV();
        break;
    case 3:
        return fespace->GetMesh()->GetNEdges();
        break;
    default:
        return -1;
    }
}

int DofHandlerFE::getNumberOfPeaks() const
{
    if (nDim == 3)
        return fespace->GetMesh()->GetNV();
    else
        return -1;
}

int DofHandlerFE::getNumberOf(int type) const
{
    switch (type)
    {
    case AgglomeratedTopology::ELEMENT:
        return getNumberOfElements();
    case AgglomeratedTopology::FACET:
        return getNumberOfFacets();
    case AgglomeratedTopology::RIDGE:
        return getNumberOfRidges();
    case AgglomeratedTopology::PEAK:
        return getNumberOfPeaks();
    default:
        return -1;
    }
}

void DofHandlerFE::checkMyInvariants() const
{
    // what do we intend to do here? anything?
}

int DofHandlerFE::getNumberOfDofForElement(int entity_id)
{
    return fespace->FEColl()->FiniteElementForGeometry(
        fespace->GetMesh()->GetElementBaseGeometry(entity_id))->GetDof();
}

int DofHandlerFE::getNumberOfDofForFacet(int entity_id)
{
    switch (nDim)
    {
    case 1:
        return fespace->FEColl()->FiniteElementForGeometry(Geometry::POINT)->GetDof();
    case 2:
        return fespace->FEColl()->FiniteElementForGeometry(Geometry::SEGMENT)->GetDof();
    case 3:
        return fespace->FEColl()->FiniteElementForGeometry(
            fespace->GetMesh()->GetFaceBaseGeometry(entity_id))->GetDof();
    default:
        const bool ndim_not_supported = true;
        PARELAG_TEST_FOR_EXCEPTION(
            ndim_not_supported,
            std::runtime_error,
            "DofHandlerFE::getNumberOfDofForFacet(...): "
            "nDim = " << nDim << " is not supported.");
    }
    return -1;
}
int DofHandlerFE::getNumberOfDofForRidge(int /*entity_id*/)
{
    switch (nDim)
    {
    case 2:
        return fespace->FEColl()->FiniteElementForGeometry(Geometry::POINT)->GetDof();
    case 3:
        return fespace->FEColl()->FiniteElementForGeometry(Geometry::SEGMENT)->GetDof();
    default:
        const bool ndim_not_supported = true;
        PARELAG_TEST_FOR_EXCEPTION(
            ndim_not_supported,
            std::runtime_error,
            "DofHandlerFE::getNumberOfDofForRidge(...): "
            "nDim = " << nDim << " is not supported.");
    }
    return -1;
}

int DofHandlerFE::getNumberOfDofForPeak(int /*entity_id*/)
{
    PARELAG_TEST_FOR_EXCEPTION(
        nDim != 3,
        std::runtime_error,
        "DofHandlerFE::getNumberOfDofForPeak(...): "
        "nDim = " << nDim << " != 3 is not supported.");

    return fespace->FEColl()->FiniteElementForGeometry(Geometry::POINT)->GetDof();
}

int DofHandlerFE::getNumberOfDofForEntity(int entity_type, int entity_id)
{
    switch (entity_type)
    {
    case AgglomeratedTopology::ELEMENT:
        return getNumberOfDofForElement(entity_id);
    case AgglomeratedTopology::FACET:
        return getNumberOfDofForFacet(entity_id);
    case AgglomeratedTopology::RIDGE:
        return getNumberOfDofForRidge(entity_id);
    case AgglomeratedTopology::PEAK:
        return getNumberOfDofForPeak(entity_id);
    default:
        return -1;
    }
}



//============================================================================

DofHandlerALG::DofHandlerALG(
    size_t maxCodimensionBaseForDof,
    const std::shared_ptr<AgglomeratedTopology>& topology)
    : DofHandler(topology->GetComm(),
                 maxCodimensionBaseForDof,
                 topology->Dimensions()),
      entity_hasInteriorDofs{},
      entityType_nDofs{},
      Topology_(topology),
      entity_NumberOfInteriorDofsNullSpace{},
      entity_NumberOfInteriorDofsRangeTSpace{},
      entity_InteriorDofOffsets{},
      nDofs{0},
      DofType{0}
{
    PARELAG_ASSERT(Topology_);

    size_t codim(0);
    for (; codim < maxCodimensionBaseForDof; ++codim)
    {
        entity_hasInteriorDofs[codim] = NullSpace | RangeTSpace;
        entity_NumberOfInteriorDofsRangeTSpace[codim].resize(
            Topology_->GetNumberLocalEntities(static_cast<entity>(codim)));
        entity_NumberOfInteriorDofsNullSpace[codim].resize(
            Topology_->GetNumberLocalEntities(static_cast<entity>(codim)));
        entity_InteriorDofOffsets[codim].resize(
            Topology_->GetNumberLocalEntities(static_cast<entity>(codim)) + 1);
    }

    if ((int) codim == Topology_->Dimensions())
    {
        entity_hasInteriorDofs[codim] = RangeTSpace;
        entity_NumberOfInteriorDofsRangeTSpace[codim].resize(
            Topology_->GetNumberLocalEntities(static_cast<entity>(codim)));
        entity_NumberOfInteriorDofsNullSpace[codim].clear();
        entity_InteriorDofOffsets[codim].resize(
            Topology_->GetNumberLocalEntities(static_cast<entity>(codim)) + 1);
    }
    else
    {
        entity_hasInteriorDofs[codim] = NullSpace | RangeTSpace;
        entity_NumberOfInteriorDofsNullSpace[codim].resize(
            Topology_->GetNumberLocalEntities(static_cast<entity>(codim)));
        entity_NumberOfInteriorDofsRangeTSpace[codim].resize(
            Topology_->GetNumberLocalEntities(static_cast<entity>(codim)));
        entity_InteriorDofOffsets[codim].resize(
            Topology_->GetNumberLocalEntities(static_cast<entity>(codim)) + 1);
    }

    for (codim = maxCodimensionBaseForDof+1; codim < MAX_CODIMENSION; ++codim)
    {
        entity_hasInteriorDofs[codim] = Empty;
        entity_NumberOfInteriorDofsNullSpace[codim].clear();
        entity_NumberOfInteriorDofsRangeTSpace[codim].clear();
        entity_InteriorDofOffsets[codim].clear();
    }

    std::fill(entityType_nDofs.begin(), entityType_nDofs.end(), 0);
}

DofHandlerALG::DofHandlerALG(
    int * entity_HasInteriorDofs_,size_t maxCodimensionBaseForDof,
    const std::shared_ptr<AgglomeratedTopology>& topology)
    : DofHandler(topology->GetComm(),
                 maxCodimensionBaseForDof,
                 topology->Dimensions()),
      entity_hasInteriorDofs{},
      entityType_nDofs{},
      Topology_(topology),
      entity_NumberOfInteriorDofsNullSpace{},
      entity_NumberOfInteriorDofsRangeTSpace{},
      entity_InteriorDofOffsets{},
      nDofs{0},
      DofType{0}
{
    PARELAG_ASSERT(Topology_);

    PARELAG_TEST_FOR_EXCEPTION(
        !(entity_HasInteriorDofs_[maxCodimensionBaseForDof] & RangeTSpace),
        std::runtime_error,
        "DofHandlerALG::DofHandlerALG(...): "
        "entity_HasInteriorDofs_[maxCodimensionBaseForDof] must contain "
        "at least the Pasciak-Vassilevski Space.");

    for (size_t codim(0); codim < MAX_CODIMENSION; ++codim)
    {
        entity_hasInteriorDofs[codim] = entity_HasInteriorDofs_[codim];

        if (entity_hasInteriorDofs[codim] & NullSpace)
        {
            entity_NumberOfInteriorDofsNullSpace[codim].resize(
                Topology_->GetNumberLocalEntities(static_cast<entity>(codim)));
        }
        else
        {
            entity_NumberOfInteriorDofsNullSpace[codim].clear();
        }

        PARELAG_TEST_FOR_EXCEPTION(
            !(entity_hasInteriorDofs[codim] & RangeTSpace),
            std::runtime_error,
            "DofHandlerALG::DofHandlerALG(...): "
            "entity_HasInteriorDofs_[codim] must always be true.");

        entity_NumberOfInteriorDofsRangeTSpace[codim].resize(
            Topology_->GetNumberLocalEntities(static_cast<entity>(codim)));

        entity_InteriorDofOffsets[codim].resize(
            Topology_->GetNumberLocalEntities(static_cast<entity>(codim)) + 1);
    }

    std::fill(entityType_nDofs.begin(), entityType_nDofs.end(), 0);
}

DofHandlerALG::~DofHandlerALG()
{
}

int DofHandlerALG::MarkDofsOnSelectedBndr(
    const Array<int> & bndrAttributesMarker, Array<int> & dofMarker) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        dofMarker.Size() != GetNDofs(),
        std::runtime_error,
        "DofHandlerALG::MarkDofsOnSelectedBndr(...): Incorrect array size!");

    dofMarker = 0;
    const TopologyTable & fc_bdnr(Topology_->FacetBdrAttribute());

    int n_fc = fc_bdnr.Size();
    const int * i_fc_bndr = fc_bdnr.GetI();
    const int * j_fc_bndr = fc_bdnr.GetJ();
    const int * i_facet_dof = entity_dof[AgglomeratedTopology::FACET]->GetI();
    const int * j_facet_dof = entity_dof[AgglomeratedTopology::FACET]->GetJ();
    int start(0), end(0);
    for (int ifc = 0; ifc < n_fc; ++ifc)
    {
        end = i_fc_bndr[ifc+1];
        elag_assert(((end-start) == 0) || ((end-start)  == 1));
        if ((end-start) == 1 && bndrAttributesMarker[j_fc_bndr[start]])
        {
            for (const int * it = j_facet_dof + i_facet_dof[ifc];
                it != j_facet_dof + i_facet_dof[ifc+1]; ++it)
            {
                dofMarker[*it] = 1;
            }
        }
        start = end;
    }

    int nMarked(0);
    for (int * it = dofMarker.GetData();
        it != dofMarker.GetData()+ dofMarker.Size(); ++it)
    {
        if (*it)
            ++nMarked;
    }
    return nMarked;
}

int DofHandlerALG::GetNumberInteriorDofs(entity type)
{

    if (type == maxCodimensionBaseForDof)
        return entity_InteriorDofOffsets[type].back();
    else
        return entity_InteriorDofOffsets[type].back() -
            entity_InteriorDofOffsets[type+1].back();
}

int DofHandlerALG::GetNumberInteriorDofs(entity type, int entity_id)
{
    return entity_InteriorDofOffsets[type][entity_id+1] -
        entity_InteriorDofOffsets[type][entity_id];
}

void DofHandlerALG::BuildEntityDofTables()
{
    buildPeakDofTable();
    buildRidgeDofTable();
    buildFacetDofTable();
    buildElementDofTable();
}

void DofHandlerALG::BuildEntityDofTable(entity type)
{
    switch (type)
    {
    case AgglomeratedTopology::PEAK:
        buildPeakDofTable();
        break;
    case AgglomeratedTopology::RIDGE:
        buildRidgeDofTable();
        break;
    case AgglomeratedTopology::FACET:
        buildFacetDofTable();
        break;
    case AgglomeratedTopology::ELEMENT:
        buildElementDofTable();
        break;
    default:
        const bool bad_entity_type = true;
        PARELAG_TEST_FOR_EXCEPTION(
            bad_entity_type,
            std::runtime_error,
            "DofHandlerFE::getDofForEntity(...): Wrong entity type.");
    }
}


void DofHandlerALG::AllocateDofTypeArray(int maxSize)
{
    DofType.SetSize(maxSize, Empty);
}

void DofHandlerALG::SetDofType(int dof, dof_type type)
{
    PARELAG_TEST_FOR_EXCEPTION(
        DofType[dof] != Empty,
        std::runtime_error,
        "DofHandlerALG::SetDofType(...): "
        "DofType[" << dof <<"] is already set to the value " <<
        DofType[dof] << "\nYou can't change the type of Dof.");

    PARELAG_ASSERT(type != Empty);
    DofType[dof] = type;
}


unique_ptr<SparseMatrix>
DofHandlerALG::GetEntityNullSpaceDofTable(entity type) const
{
    elag_assert(DofType.Find(Empty) == -1 || DofType.Find(Empty) >= nDofs);
    elag_assert(DofType.Size() >= nDofs);

    int colStart(0);
    if (type < maxCodimensionBaseForDof)
        colStart = entityType_nDofs[type+1];

    DropEntryAccordingToColumnMarkerAndId drop(DofType, NullSpace, colStart);
    return DropEntriesFromSparseMatrix(*entity_dof[type], drop);
}


unique_ptr<SparseMatrix>
DofHandlerALG::GetEntityRangeTSpaceDofTable(entity type) const
{
    elag_assert(DofType.Find(Empty) == -1 || DofType.Find(Empty) >= nDofs);
    elag_assert(DofType.Size() >= nDofs);

    int colStart(0);
    if (type < maxCodimensionBaseForDof)
        colStart = entityType_nDofs[type+1];

    DropEntryAccordingToColumnMarkerAndId drop(DofType, RangeTSpace, colStart);
    return DropEntriesFromSparseMatrix(*entity_dof[type], drop);
}

unique_ptr<SparseMatrix>
DofHandlerALG::GetEntityInternalDofTable(entity type) const
{
    elag_assert(finalized[type]);

    int colStart(0);
    if (type < maxCodimensionBaseForDof)
        colStart = entityType_nDofs[type+1];

    DropEntryAccordingToId drop(colStart);
    return DropEntriesFromSparseMatrix(*entity_dof[type], drop);
}

void DofHandlerALG::SetNumberOfInteriorDofsNullSpace(
    entity type, int entity_id, int nLocDof)
{
    PARELAG_TEST_FOR_EXCEPTION(
        !(entity_hasInteriorDofs[type] & NullSpace),
        std::runtime_error,
        "DofHandlerALG::SetNumberOfInteriorDofsNullSpace(...): "
        "There is a problem.");

    entity_NumberOfInteriorDofsNullSpace[type][entity_id] = nLocDof;
}

void DofHandlerALG::SetNumberOfInteriorDofsRangeTSpace(
    entity type, int entity_id, int nLocDof)
{
    PARELAG_TEST_FOR_EXCEPTION(
        !(entity_hasInteriorDofs[type] & RangeTSpace),
        std::runtime_error,
        "DofHandlerALG::SetNumberOfInteriorDofsRangeTSpace(...): "
        "There is a problem.");

    entity_NumberOfInteriorDofsRangeTSpace[type][entity_id] = nLocDof;
}

void DofHandlerALG::GetInteriorDofs(entity type, int ientity, Array<int> & dofs)
{
    Vector dummy;
    // all dofs are internal
    if (type == static_cast<entity>(maxCodimensionBaseForDof))
        entity_dof[type]->GetRow(ientity, dofs, dummy);
    else
    {
        Array<int> allDofs;
        int startIndex = entityType_nDofs[type+1];
        entity_dof[type]->GetRow(ientity, allDofs, dummy);

        const int size = allDofs.Size();
        dofs.SetSize(size);
        dofs.SetSize(0);

        int * data = allDofs.GetData();
        for (int * end = data+size; data != end; ++data)
            if (*data >= startIndex)
                dofs.Append(*data);
    }
}


void DofHandlerALG::GetDofsOnBdr(entity type, int ientity, Array<int> & dofs)
{
    int ndofs(0);
    Array<int> bdr_entity;
    Array<int> ibdr_entity_dofs;

    for (int i(maxCodimensionBaseForDof); i > type ;--i)
    {
        entity type_bdr = static_cast<entity>(i);
        Topology_->GetBoundaryOfEntity(type, type_bdr, ientity, bdr_entity);

        for (int * ibdr_entity = bdr_entity.GetData();
            ibdr_entity != bdr_entity.GetData()+bdr_entity.Size();
            ++ibdr_entity)
            ndofs += GetNumberInteriorDofs(type_bdr, *ibdr_entity);

    }

    dofs.SetSize(ndofs);
    int * it = dofs.GetData();

    for (int i(maxCodimensionBaseForDof); i > type; --i)
    {
        entity type_bdr = static_cast<entity>(i);
        Topology_->GetBoundaryOfEntity(type, type_bdr, ientity, bdr_entity);

        for (int * ibdr_entity = bdr_entity.GetData();
            ibdr_entity != bdr_entity.GetData()+bdr_entity.Size();
            ++ibdr_entity)
        {
            GetInteriorDofs(type_bdr, *ibdr_entity, ibdr_entity_dofs);
            it = std::copy(
                ibdr_entity_dofs.GetData(),
                ibdr_entity_dofs.GetData()+ibdr_entity_dofs.Size(), it);
        }

    }
}


int DofHandlerALG::getNumberOf(int type) const
{
    return Topology_->GetNumberLocalEntities(static_cast<entity>(type));
}

void DofHandlerALG::computeOffset(entity type)
{
    if (finalized[type])
        return;

#ifdef ELAG_DEBUG
    PARELAG_TEST_FOR_EXCEPTION(
        type > maxCodimensionBaseForDof,
        std::runtime_error,
        "DofHandlerALG::computeOffset(...): Invalid type!");

    if (type ==  maxCodimensionBaseForDof)
    {
        PARELAG_TEST_FOR_EXCEPTION(
            nDofs != 0 || entityType_nDofs[maxCodimensionBaseForDof] != 0,
            std::runtime_error,
            "DofHandlerALG::computeOffset(...): "
            "Something strange is happening. nDofs should be 0.");
    }
    else
    {
        PARELAG_TEST_FOR_EXCEPTION(
            nDofs < 0,
            std::runtime_error,
            "DofHandlerALG::computeOffset(...): Invalid nDofs.");

        PARELAG_TEST_FOR_EXCEPTION(
            !finalized[type+1],
            std::logic_error,
            "DofHandlerALG::computeOffset(...): "
            "Before calling computedOffset(codim), you need to call "
            "BuildEntityDofTable(codim+1).");

        PARELAG_TEST_FOR_EXCEPTION(
            nDofs != entityType_nDofs[type+1],
            std::runtime_error,
            "DofHandlerALG::computeOffset(...): "
            "Something strange is happening. "
            "nDofs and entityType_nDofs do not agree");
    }

    if (entity_hasInteriorDofs[type] & RangeTSpace)
    {
        for (const auto& val : entity_NumberOfInteriorDofsRangeTSpace[type])
        {
            PARELAG_TEST_FOR_EXCEPTION(
                val < 0,
                std::runtime_error,
                "DofHandlerALG::computeOffset(...): Negative number!");

            PARELAG_TEST_FOR_EXCEPTION(
                val > 100,
                std::runtime_error,
                "DofHandlerALG::computeOffset(...): "
                "So many dofs on a coarse entity are impossible!");
        }
    }

    if (entity_hasInteriorDofs[type] & NullSpace)
    {
        for (const auto& val : entity_NumberOfInteriorDofsNullSpace[type])
        {
            PARELAG_TEST_FOR_EXCEPTION(
                val < 0,
                std::runtime_error,
                "DofHandlerALG::computeOffset(...): Negative number!");

            PARELAG_TEST_FOR_EXCEPTION(
                val > 100,
                std::runtime_error,
                "DofHandlerALG::computeOffset(...): "
                "So many dofs on a coarse entity are impossible!");
        }
    }

#endif

    switch (entity_hasInteriorDofs[type])
    {
    case Empty:
        break;

    case RangeTSpace:
    {
        auto e_offset = entity_InteriorDofOffsets[type].begin();
        auto rangeTSpace_ndof =
            entity_NumberOfInteriorDofsRangeTSpace[type].begin();
        const auto rangeTSpace_ndof_end =
            entity_NumberOfInteriorDofsRangeTSpace[type].end();

        for (; rangeTSpace_ndof != rangeTSpace_ndof_end;
             ++e_offset, ++rangeTSpace_ndof)
        {
            *e_offset = nDofs;
            nDofs += *rangeTSpace_ndof;
        }
        *e_offset = nDofs;
        break;
    }
    case RangeTSpace | NullSpace:
    {
        auto e_offset = entity_InteriorDofOffsets[type].begin();
        auto nullSpace_ndof =
            entity_NumberOfInteriorDofsNullSpace[type].begin();
        auto rangeTSpace_ndof =
            entity_NumberOfInteriorDofsRangeTSpace[type].begin();
        const auto rangeTSpace_ndof_end =
            entity_NumberOfInteriorDofsRangeTSpace[type].end();

        for (; rangeTSpace_ndof != rangeTSpace_ndof_end;
             ++e_offset, ++nullSpace_ndof, ++rangeTSpace_ndof)
        {
            *e_offset = nDofs;
            nDofs += *nullSpace_ndof+*rangeTSpace_ndof;
        }
        *e_offset = nDofs;
        break;
    }
    default:
        const bool invalid_case = true;
        PARELAG_TEST_FOR_EXCEPTION(
            invalid_case,
            std::runtime_error,
            "DofHandlerALG::computeOffset(...): "
            "Invalid entity_hasInteriorDofs[type]");
    }

    entityType_nDofs[type] = nDofs;
}

void DofHandlerALG::buildPeakDofTable()
{
    if (AgglomeratedTopology::PEAK > maxCodimensionBaseForDof)
        return;

    computeOffset(AgglomeratedTopology::PEAK);

    const int nrow =
        Topology_->GetNumberLocalEntities(AgglomeratedTopology::PEAK);
    const int ncol = entityType_nDofs[AgglomeratedTopology::PEAK];

    int * i = new int[nrow+1];
    int * j = new int[ncol];

    std::copy(entity_InteriorDofOffsets[AgglomeratedTopology::PEAK].begin(),
              entity_InteriorDofOffsets[AgglomeratedTopology::PEAK].end(), i);
    std::iota(j,j+ncol,0);

    double * val = new double[ncol];
    std::fill(val, val+ncol, 1.);
    entity_dof[AgglomeratedTopology::PEAK] = make_unique<SparseMatrix>(
        i,j,val,nrow,ncol);
    CheckMatrix(*entity_dof[AgglomeratedTopology::PEAK]);
    finalized[AgglomeratedTopology::PEAK] = true;
}


void DofHandlerALG::buildRidgeDofTable()
{

    if (AgglomeratedTopology::RIDGE > maxCodimensionBaseForDof)
        return;

    computeOffset(AgglomeratedTopology::RIDGE);

    const int nrow =
        Topology_->GetNumberLocalEntities(AgglomeratedTopology::RIDGE);
    const int ncol = entityType_nDofs[AgglomeratedTopology::RIDGE];

    int * i = new int[nrow+2];
    std::fill(i, i+nrow+2, 0);
    int * i_assembly = i+2;

    EntityDofFillIAssembly(AgglomeratedTopology::RIDGE,
                           AgglomeratedTopology::PEAK, i_assembly);
    if (entity_hasInteriorDofs[AgglomeratedTopology::RIDGE])
    {
        const auto& r_offsets =
            entity_InteriorDofOffsets[AgglomeratedTopology::RIDGE];
        for (int iridge(0); iridge < nrow; ++iridge)
            i_assembly[iridge] += r_offsets[iridge+1] - r_offsets[iridge];
    }
    std::partial_sum(i_assembly, i_assembly+nrow, i_assembly);

    // shift back
    i_assembly = i+1;
    int nnz = i_assembly[nrow];
    int * j = new int[nnz];

    EntityDofFillJ(AgglomeratedTopology::RIDGE,
                   AgglomeratedTopology::PEAK, i_assembly, j);
    if (entity_hasInteriorDofs[AgglomeratedTopology::RIDGE])
    {
        const auto& r_offsets =
            entity_InteriorDofOffsets[AgglomeratedTopology::RIDGE];
        for (int iridge(0); iridge < nrow; ++iridge)
        {
            int * j_it = j+i_assembly[iridge];
            for (int jdof = r_offsets[iridge]; jdof < r_offsets[iridge+1];
                 ++jdof, ++j_it)
            {
                *j_it = jdof;
            }
            i_assembly[iridge] += r_offsets[iridge+1] - r_offsets[iridge];
        }
    }

    double * val = new double[nnz];
    std::fill(val, val + nnz, 1.);
    entity_dof[AgglomeratedTopology::RIDGE] = make_unique<SparseMatrix>(
        i,j,val,nrow,ncol);
    CheckMatrix(*entity_dof[AgglomeratedTopology::RIDGE]);
    finalized[AgglomeratedTopology::RIDGE] = true;
}

void DofHandlerALG::buildFacetDofTable()
{
    if (AgglomeratedTopology::FACET > maxCodimensionBaseForDof)
        return;

    computeOffset(AgglomeratedTopology::FACET);

    const int nrow = Topology_->GetNumberLocalEntities(
        AgglomeratedTopology::FACET);
    const int ncol = entityType_nDofs[AgglomeratedTopology::FACET];
    int * i = new int[nrow+2];
    std::fill(i, i+nrow+2, 0);
    int * i_assembly = i+2;

    EntityDofFillIAssembly(AgglomeratedTopology::FACET,
                           AgglomeratedTopology::PEAK, i_assembly);
    EntityDofFillIAssembly(AgglomeratedTopology::FACET,
                           AgglomeratedTopology::RIDGE, i_assembly);
    if (entity_hasInteriorDofs[AgglomeratedTopology::FACET])
    {
        const auto& f_offsets =
            entity_InteriorDofOffsets[AgglomeratedTopology::FACET];
        for (int ifacet = 0; ifacet < nrow; ++ifacet)
            i_assembly[ifacet] += f_offsets[ifacet+1] - f_offsets[ifacet];
    }
    std::partial_sum(i_assembly, i_assembly+nrow, i_assembly);

    // shift back
    i_assembly = i+1;
    int nnz = i_assembly[nrow];
    int * j = new int[nnz];

    EntityDofFillJ(AgglomeratedTopology::FACET,
                   AgglomeratedTopology::PEAK, i_assembly, j);
    EntityDofFillJ(AgglomeratedTopology::FACET,
                   AgglomeratedTopology::RIDGE, i_assembly, j);
    if (entity_hasInteriorDofs[AgglomeratedTopology::FACET])
    {
        const auto& f_offsets =
            entity_InteriorDofOffsets[AgglomeratedTopology::FACET];
        for (int ifacet(0); ifacet < nrow; ++ifacet)
        {
            int * j_it = j+i_assembly[ifacet];
            for (int jdof = f_offsets[ifacet]; jdof < f_offsets[ifacet+1];
                 ++jdof, ++j_it)
            {
                *j_it = jdof;
            }
            i_assembly[ifacet] += f_offsets[ifacet+1] - f_offsets[ifacet];
        }
    }

    double * val = new double[nnz];
    std::fill(val, val + nnz, 1.);
    entity_dof[AgglomeratedTopology::FACET] = make_unique<SparseMatrix>(
        i,j,val,nrow,ncol);
    CheckMatrix(*entity_dof[AgglomeratedTopology::FACET]);
    finalized[AgglomeratedTopology::FACET] = true;
}

/**
   This is just trying to reduce copy/paste in buildElementDofTable(), I am not
   sure what it actually *does*.

   I think it counts row sizes so when the time comes to fill in J, we know
   where to put everything.
*/
void DofHandlerALG::EntityDofFillIAssembly(
    entity entity_big, entity entity_small, int * i_assembly)
{
    if (entity_hasInteriorDofs[entity_small])
    {
        const int nrow = Topology_->GetNumberLocalEntities(entity_big);
        const BooleanMatrix & big_small =
            Topology_->GetConnectivity(static_cast<int>(entity_big),
                                       static_cast<int>(entity_small));
        const int * i_big_small = big_small.GetI();
        const int * j_big_small = big_small.GetJ();
        const auto& p_offsets = entity_InteriorDofOffsets[entity_small];

        for (int ibig(0); ibig < nrow; ++ibig)
        {
            for (const int * j_small = j_big_small + i_big_small[ibig];
                 j_small != j_big_small + i_big_small[ibig+1];
                 ++j_small)
            {
                i_assembly[ibig] += p_offsets[*j_small+1] - p_offsets[*j_small];
            }
        }
    }
}

/**
   See comments on EntityDofFillIAssembly(), same crusade.
   Note that these two routines are strangely similar...
*/
void DofHandlerALG::EntityDofFillJ(
    entity entity_big, entity entity_small, int * i_assembly, int * j)
{
    if (entity_hasInteriorDofs[entity_small])
    {
        const int nrow = Topology_->GetNumberLocalEntities(entity_big);
        const BooleanMatrix & big_small = Topology_->GetConnectivity(
            static_cast<int>(entity_big), static_cast<int>(entity_small));
        const int * i_big_small = big_small.GetI();
        const int * j_big_small = big_small.GetJ();
        const auto& p_offsets = entity_InteriorDofOffsets[entity_small];

        for (int ibig(0); ibig < nrow; ++ibig)
        {
            for (const int * j_small = j_big_small + i_big_small[ibig];
                 j_small != j_big_small + i_big_small[ibig+1];
                 ++j_small)
            {
                int * j_it = j+i_assembly[ibig];
                for (int jdof = p_offsets[*j_small];
                     jdof < p_offsets[*j_small+1];
                     ++jdof, ++j_it)
                {
                    *j_it = jdof;
                }
                i_assembly[ibig] += p_offsets[*j_small+1] - p_offsets[*j_small];
            }
        }
    }
}

void DofHandlerALG::buildElementDofTable()
{
    computeOffset(AgglomeratedTopology::ELEMENT);

    const int nrow = Topology_->GetNumberLocalEntities(
        AgglomeratedTopology::ELEMENT);
    const int ncol = entityType_nDofs[AgglomeratedTopology::ELEMENT];

    int * i = new int[nrow+2];
    std::fill(i, i+nrow+2, 0);
    int * i_assembly = i+2;

    EntityDofFillIAssembly(AgglomeratedTopology::ELEMENT,
                           AgglomeratedTopology::PEAK, i_assembly);
    EntityDofFillIAssembly(AgglomeratedTopology::ELEMENT,
                           AgglomeratedTopology::RIDGE, i_assembly);
    EntityDofFillIAssembly(AgglomeratedTopology::ELEMENT,
                           AgglomeratedTopology::FACET, i_assembly);
    if (entity_hasInteriorDofs[AgglomeratedTopology::ELEMENT])
    {
        const auto& e_offsets =
            entity_InteriorDofOffsets[AgglomeratedTopology::ELEMENT];
        for (int iel(0); iel < nrow; ++iel)
            i_assembly[iel] += e_offsets[iel+1] - e_offsets[iel];
    }
    std::partial_sum(i_assembly, i_assembly+nrow, i_assembly);

    // shift back, whatever that means
    i_assembly = i+1;
    int nnz = i_assembly[nrow];
    int * j = new int[nnz];

    EntityDofFillJ(AgglomeratedTopology::ELEMENT, AgglomeratedTopology::PEAK,
                   i_assembly, j);
    EntityDofFillJ(AgglomeratedTopology::ELEMENT, AgglomeratedTopology::RIDGE,
                   i_assembly, j);
    EntityDofFillJ(AgglomeratedTopology::ELEMENT, AgglomeratedTopology::FACET,
                   i_assembly, j);
    if (entity_hasInteriorDofs[AgglomeratedTopology::ELEMENT])
    {
        const auto& e_offsets =
            entity_InteriorDofOffsets[AgglomeratedTopology::ELEMENT];
        for (int iel(0); iel < nrow; ++iel)
        {
            int * j_it = j+i_assembly[iel];
            for (int jdof = e_offsets[iel];
                jdof < e_offsets[iel+1];
                ++jdof, ++j_it)
            {
                *j_it = jdof;
            }
            i_assembly[iel] += e_offsets[iel+1] - e_offsets[iel];
        }
    }

    double * val = new double[nnz];
    std::fill(val, val + nnz, 1.);
    entity_dof[AgglomeratedTopology::ELEMENT] = make_unique<SparseMatrix>(
        i,j,val,nrow,ncol);
    CheckMatrix(*entity_dof[AgglomeratedTopology::ELEMENT]);
    finalized[AgglomeratedTopology::ELEMENT] = true;
}

void DofHandlerALG::checkMyInvariants() const
{
    PARELAG_NOT_IMPLEMENTED();
}
}//namespace parelag
