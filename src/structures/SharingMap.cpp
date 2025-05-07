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

#include <mfem.hpp>

#include "SharingMap.hpp"

#include "amge/DofHandler.hpp"
#include "hypreExtension/hypreExtension.hpp"
#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "structures/minimalIntersectionSet.hpp"
#include "utilities/elagError.hpp"
#include "utilities/HypreTraits.hpp"
#include "utilities/MemoryUtils.hpp"
#include "utilities/mpiUtils.hpp"
#include "utilities/Trace.hpp"

using namespace mfem;
using std::unique_ptr;

namespace parelag
{

SharingMap::SharingMap(MPI_Comm comm_):
    Comm_(comm_),
    AssumedPID_(0),
    AssumedNumProc_(2),
    entity_start(0),
    trueEntity_start(0),
    entity_trueEntity(nullptr),
    entity_trueEntity_entity(nullptr),
    HelpData_(0),
    HelpTrueData_(0),
    xTrue_(nullptr,hypre_ParVectorDestroy),
    x_(nullptr,hypre_ParVectorDestroy)
{
    if(!HYPRE_AssumedPartitionCheck())
    {
        MPI_Comm_rank(Comm_, &AssumedPID_);
        MPI_Comm_size(Comm_, &AssumedNumProc_);
    }

    entity_start.SetSize(AssumedNumProc_+1);
    trueEntity_start.SetSize(AssumedNumProc_+1);
}

SharingMap::SharingMap(const SharingMap& map):
    Comm_(map.Comm_),
    AssumedPID_(map.AssumedPID_),
    AssumedNumProc_(map.AssumedNumProc_),
    xTrue_(nullptr,hypre_ParVectorDestroy),
    x_(nullptr,hypre_ParVectorDestroy)
{
    // FIXME (trb 12/10/15): I just want to find out where/why the
    // copy ctor is used.
    PARELAG_ASSERT(false);
#ifdef TOM_IS_DONE
    entity_start.SetSize(map.entity_start.Size());
    map.entity_start.Copy(entity_start);
    trueEntity_start.SetSize(map.trueEntity_start.Size());
    map.trueEntity_start.Copy(trueEntity_start);
    entity_trueEntity
        = HypreExtension::DeepCopy(map.entity_trueEntity);
    entity_trueEntity_entity
        = HypreExtension::DeepCopy(map.entity_trueEntity_entity);
    sharedEntityIds.SetSize(map.sharedEntityIds.Size());
    map.sharedEntityIds.Copy(sharedEntityIds);
    nOwnedSharedEntities_ = map.nOwnedSharedEntities_;
    HelpData_.SetSize(map.HelpData_.Size());
    HelpData_ = map.HelpData_;
    HelpTrueData_.SetSize(map.HelpTrueData_.Size());
    HelpTrueData_ = map.HelpTrueData_;

    // Creates x and xTrue_;
    resetHypreParVectors();
#endif
}

void SharingMap::SetUp(unique_ptr<ParallelCSRMatrix> entity_trueEntity_)
{
    elag_trace_enter_block(
        "SharingMap::SetUp(entity_trueEntity_)");

    entity_trueEntity = std::move(entity_trueEntity_);

    hypre_ParCSRMatrix * h_e_tE = *entity_trueEntity;

#if MFEM_HYPRE_VERSION <= 22200
    elag_assert( hypre_ParCSRMatrixOwnsRowStarts(h_e_tE) == 1 );
    elag_assert( hypre_ParCSRMatrixOwnsColStarts(h_e_tE) == 1 );
#endif

    entity_start[0] = entity_trueEntity->RowPart()[0];
    entity_start[1] = entity_trueEntity->RowPart()[1];
    entity_start[2] = entity_trueEntity->M();
    trueEntity_start[0] = entity_trueEntity->ColPart()[0];
    trueEntity_start[1] = entity_trueEntity->ColPart()[1];
    trueEntity_start[2] = entity_trueEntity->N();

    if (!hypre_ParCSRMatrixCommPkg(h_e_tE))
        hypre_MatvecCommPkgCreate(h_e_tE);

    hypre_ParCSRMatrix * tE_e;

    elag_trace("Transpose entity_trueEntity");
    hypre_ParCSRMatrixTranspose2(h_e_tE, &tE_e);
    elag_trace("Transpose entity_trueEntity - done!");

#if MFEM_HYPRE_VERSION <= 22200
    elag_assert( hypre_ParCSRMatrixOwnsRowStarts(tE_e) == 0 );
    elag_assert( hypre_ParCSRMatrixOwnsColStarts(tE_e) == 0 );
#endif

    elag_trace("Compute entity_trueEntity_entity");
    entity_trueEntity_entity = make_unique<ParallelCSRMatrix>(
        hypre_ParMatmul(h_e_tE, tE_e) );
    elag_trace("Compute entity_trueEntity_entity - done!");

    hypre_ParCSRMatrixDestroy(tE_e);

    hypre_ParCSRMatrix * h_e_tE_e = *entity_trueEntity_entity;
#if MFEM_HYPRE_VERSION <= 22200
    PARELAG_ASSERT( hypre_ParCSRMatrixOwnsRowStarts(h_e_tE_e) == 0 );
    PARELAG_ASSERT( hypre_ParCSRMatrixOwnsColStarts(h_e_tE_e) == 0 );
#endif

    resetHypreParVectors();
    storeSharedEntitiesIds();
    elag_trace_leave_block(
        "SharingMap::SetUp(entity_trueEntity_)");
}

void SharingMap::SetUp(Array<int> & entityStart,
                       Array<int> & trueEntityStart,
                       unique_ptr<ParallelCSRMatrix> entity_trueEntity_)
{
    elag_trace_enter_block(
        "SharingMap::SetUp(entityStart, trueEntityStart, entity_trueEntity_)");
    if(&entity_start != &entityStart)
        entityStart.Copy(entity_start);
    if(&trueEntity_start != &trueEntityStart)
        trueEntityStart.Copy(trueEntity_start);
    entity_trueEntity = std::move(entity_trueEntity_);

    hypre_ParCSRMatrix * h_e_tE = *entity_trueEntity;

    if (!hypre_ParCSRMatrixCommPkg(h_e_tE))
        hypre_MatvecCommPkgCreate(h_e_tE);

#if MFEM_HYPRE_VERSION <= 22200
    elag_assert( hypre_ParCSRMatrixOwnsRowStarts(h_e_tE) == 0 );
    elag_assert( hypre_ParCSRMatrixOwnsColStarts(h_e_tE) == 0 );

    hypre_ParCSRMatrixRowStarts(h_e_tE)
        = entity_start.GetData();
    hypre_ParCSRMatrixColStarts(h_e_tE)
        = trueEntity_start.GetData();
#else
    std::copy_n(entity_start.GetData(), 2, hypre_ParCSRMatrixRowStarts(h_e_tE));
    std::copy_n(trueEntity_start.GetData(), 2, hypre_ParCSRMatrixColStarts(h_e_tE));
#endif

    hypre_ParCSRMatrix * tE_e;

    elag_trace("Transpose entity_trueEntity");
    hypre_ParCSRMatrixTranspose2(h_e_tE, &tE_e);
    elag_trace("Transpose entity_trueEntity - done!");

#if MFEM_HYPRE_VERSION <= 22200
    elag_assert( hypre_ParCSRMatrixOwnsRowStarts(tE_e) == 0 );
    elag_assert( hypre_ParCSRMatrixOwnsColStarts(tE_e) == 0 );
#endif

    elag_trace("Compute entity_trueEntity_entity");
    entity_trueEntity_entity = make_unique<ParallelCSRMatrix>(
        hypre_ParMatmul(h_e_tE, tE_e) );
    elag_trace("Compute entity_trueEntity_entity - done!");

    hypre_ParCSRMatrixDestroy(tE_e);

    hypre_ParCSRMatrix * h_e_tE_e = *entity_trueEntity_entity;
#if MFEM_HYPRE_VERSION <= 22200
    PARELAG_ASSERT( hypre_ParCSRMatrixOwnsRowStarts(h_e_tE_e) == 0 );
    PARELAG_ASSERT( hypre_ParCSRMatrixOwnsColStarts(h_e_tE_e) == 0 );
#endif

    resetHypreParVectors();
    storeSharedEntitiesIds();
    elag_trace_leave_block(
        "SharingMap::SetUp(entityStart, trueEntityStart, entity_trueEntity_)");
}

void SharingMap::resetHypreParVectors()
{
    //hypre_ParVectorDestroy(xTrue_);
    //hypre_ParVectorDestroy(x_);
    xTrue_.reset(
        hypre_ParVectorCreate(
            Comm_,trueEntity_start[AssumedNumProc_],
            trueEntity_start.GetData()));

    x_.reset(
        hypre_ParVectorCreate(
            Comm_,entity_start[AssumedNumProc_],entity_start.GetData()));

#if MFEM_HYPRE_VERSION <= 22200
    hypre_ParVectorSetPartitioningOwner(xTrue_.get(), 0);
    hypre_ParVectorSetPartitioningOwner(x_.get(), 0);
#endif
    hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(xTrue_),0);
    hypre_SeqVectorSetDataOwner(hypre_ParVectorLocalVector(x_),0);

    //Trick: we would like xtrue and x to be initialized, but still
    //have null data...  So we first give xTrue_ and x some pointers,
    //then we initialize the vectors, and finally we reset the data
    //pointer to null.
    double  a;
    hypre_VectorData(hypre_ParVectorLocalVector(xTrue_)) = &a;
    hypre_VectorData(hypre_ParVectorLocalVector(x_)) = &a;
    hypre_ParVectorInitialize(xTrue_.get());
    hypre_ParVectorInitialize(x_.get());

    hypre_VectorData(hypre_ParVectorLocalVector(xTrue_)) = nullptr;
    hypre_VectorData(hypre_ParVectorLocalVector(x_)) = nullptr;
}

void SharingMap::storeSharedEntitiesIds()
{
    const int nLocEntities = GetLocalSize();

    hypre_ParCSRMatrix *h_e_tE_e = *entity_trueEntity_entity,
        *h_e_tE = *entity_trueEntity;

    hypre_CSRMatrix * offd_entity_trueEntity_entity
        = hypre_ParCSRMatrixOffd(h_e_tE_e);
    hypre_CSRMatrix * offd_entity_trueEntity
        = hypre_ParCSRMatrixOffd(h_e_tE);

    int nSharedIds = 0;
    {
        int * i_offd = hypre_CSRMatrixI(offd_entity_trueEntity_entity);
        for(int i(0); i < nLocEntities; ++i)
            if(i_offd[i+1] - i_offd[i]) ++nSharedIds;
    }

    int nSharedNotOwnedIds = 0;
    {
        int * i_offd = hypre_CSRMatrixI(offd_entity_trueEntity);
        for(int i(0); i < nLocEntities; ++i)
            if(i_offd[i+1] - i_offd[i]) ++nSharedNotOwnedIds;
    }

    nOwnedSharedEntities_ = nSharedIds - nSharedNotOwnedIds;

    sharedEntityIds.SetSize(nSharedIds);
    const int offset_not_owned = nOwnedSharedEntities_;
    int owned_counter = 0;
    int notowned_counter = 0;

    for(int i(0); i < nLocEntities; ++i)
    {
        switch(IsShared(i))
        {
        case -1:
            sharedEntityIds[offset_not_owned + notowned_counter] = i;
            ++notowned_counter;
            break;
        case 0:
            break;
        case 1:
            sharedEntityIds[owned_counter] = i;
            ++owned_counter;
            break;
        default:
            elag_error_msg(1,"The Impossible Has Happened!");
            break;
        }
    }
    elag_assert(owned_counter == nOwnedSharedEntities_);
    elag_assert(notowned_counter == nSharedNotOwnedIds);
}

void SharingMap::SetUp(ParMesh& pmesh, int codim)
{
    elag_trace_enter_block("SharingMap::SetUp(pmesh = " << &pmesh <<
                           ", codim = " << codim << ")" );

    unique_ptr<FiniteElementCollection> feColl;
    const int ndim = pmesh.Dimension();
    switch(codim)
    {
    case 0:
        feColl = make_unique<L2_FECollection>(0, ndim);
        break;
    case 1:
        feColl = make_unique<RT_FECollection>(0, ndim);
        break;
    case 2:
        if (ndim == 2)
            feColl = make_unique<H1_FECollection>(1,ndim);
        else
            feColl = make_unique<ND_FECollection>(1,ndim);
        break;
    case 3:
        elag_assert(ndim == 3);
        feColl = make_unique<H1_FECollection>(1, ndim);
        break;
    default:
        elag_error_msg(1, "Wrong Codimension number \n");
        feColl = nullptr;
        break;
    }

    auto feSpace = make_unique<ParFiniteElementSpace>(&pmesh,feColl.get());
    SetUp(feSpace.get(), 1);

    elag_trace_leave_block("SharingMap::SetUp(pmesh = " << pmesh <<
                           ", codim = " << codim << ")" );
}

void SharingMap::SetUp(ParFiniteElementSpace * fes, int useDofSign)
{
    elag_trace_enter_block("SharingMap::SetUp(fes = " << fes <<
                           ", useDofSign = " << useDofSign << ")");
    Array<int> estart(fes->GetDofOffsets(), AssumedNumProc_+1);
    Array<int> etstart(fes->GetTrueDofOffsets(), AssumedNumProc_+1);
    elag_trace("Get the dotTrueDof matrix from fes");
    // FIXME SEE BELOW
    fes->Dof_TrueDof_Matrix()->SetOwnerFlags(-1,-1,-1);
    hypre_ParCSRMatrix * mat = fes->Dof_TrueDof_Matrix()->StealData();

    elag_trace("Get the dofTrueDof matrix from fes - done");
    if(useDofSign)
    {
        hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd( mat );
        HYPRE_Int * i_offd = hypre_CSRMatrixI(offd);
        double * a_offd = hypre_CSRMatrixData(offd);
        int nrows = hypre_CSRMatrixNumRows(offd);
        for(int i = 0; i < nrows; ++i)
        {
            elag_assert( (i_offd[i+1] - i_offd[i]) < 2);
            if( (i_offd[i+1] - i_offd[i]) != 0 && fes->GetDofSign(i) == -1)
                a_offd[i_offd[i]] *= -1.;
        }

        hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag( mat );
        HYPRE_Int * i_diag = hypre_CSRMatrixI(diag);
        double * a_diag = hypre_CSRMatrixData(diag);
        for(int i = 0; i < nrows; ++i)
        {
            elag_assert( (i_diag[i+1] - i_diag[i]) < 2);
            if( (i_diag[i+1] - i_diag[i]) != 0 && fes->GetDofSign(i) == -1)
                a_diag[i_diag[i]] *= -1.;
        }

    }
    // FIXME (trb 12/22/15): There must be a better way...
    auto m_mat = make_unique<ParallelCSRMatrix>(mat);
    m_mat->SetOwnerFlags(3,3,1);

    SetUp(estart, etstart, std::move(m_mat));
    elag_trace_leave_block("SharingMap::SetUp(fes = " << fes <<
                           ", useDofSign = " << useDofSign << ")");
}

void SharingMap::SetUp(int localSize)
{
    ParPartialSums_AssumedPartitionCheck(Comm_, localSize, entity_start);

    trueEntity_start.MakeRef(entity_start);

    entity_trueEntity = make_unique<ParallelCSRMatrix>(
        hypre_IdentityParCSRMatrix(
            Comm_,entity_start.Last(),entity_start.GetData()));

    hypre_ParCSRMatrix * tE_e,
        * h_e_tE = *entity_trueEntity;

    if (!hypre_ParCSRMatrixCommPkg(h_e_tE))
        hypre_MatvecCommPkgCreate(h_e_tE);

    hypre_ParCSRMatrixTranspose2(h_e_tE, &tE_e);

#if MFEM_HYPRE_VERSION <= 22200
    elag_assert(hypre_ParCSRMatrixOwnsRowStarts(tE_e) == 0);
    elag_assert(hypre_ParCSRMatrixOwnsColStarts(tE_e) == 0);
#endif

    entity_trueEntity_entity = make_unique<HypreParMatrix>(
        hypre_ParMatmul(h_e_tE, tE_e) );

    hypre_ParCSRMatrixDestroy(tE_e);

    hypre_ParCSRMatrix * h_e_tE_e = *entity_trueEntity_entity;
#if MFEM_HYPRE_VERSION <= 22200
    PARELAG_ASSERT(hypre_ParCSRMatrixOwnsRowStarts(h_e_tE_e) == 0);
    PARELAG_ASSERT(hypre_ParCSRMatrixOwnsColStarts(h_e_tE_e) == 0);
#endif

    resetHypreParVectors();
    storeSharedEntitiesIds();
}

void SharingMap::SetUp(SerialCSRMatrix & e_AE, SharingMap & e_Te)
{
    elag_assert( e_AE.Size() == e_Te.GetLocalSize() );

    int localSize = e_AE.Width();
    ParPartialSums_AssumedPartitionCheck(Comm_, localSize, entity_start);

    ParallelCSRMatrix e_AEd(
        e_Te.Comm_,e_Te.entity_start.Last(),entity_start.Last(),
        e_Te.entity_start.GetData(),entity_start.GetData(),&e_AE);

    // (trb 12/18/15): This is a little wonky. This actually builds
    // this->entity_trueEntity_entity's hypreParCSRMatrix, so
    // currently, this->entity_trueEntity_entity is likely to be a
    // nullptr. Thus, I do nothing here.
    hypre_ParCSRMatrix *h_e_tE_e,
        *other_h_e_tE_e = *(e_Te.entity_trueEntity_entity);

    hypre_BoomerAMGBuildCoarseOperator(e_AEd,other_h_e_tE_e,e_AEd,&h_e_tE_e);

    // (trb 12/18/15): Now I need to create entity_trueEntity_entity
    // with the hypre_ParCSRMatrix that was just created.
    entity_trueEntity_entity = make_unique<ParallelCSRMatrix>(h_e_tE_e);
    // FIXME (trb 12/18/15): Best I can tell, this should own the
    // newly created matrix. How do I make this happen?

    //hypre_BoomerAMG gives by default ownership of the column/row
    //partitioning to the result. We don't want that :)
    hypre_ParCSRMatrixSetRowStartsOwner(h_e_tE_e,0);
    hypre_ParCSRMatrixSetColStartsOwner(h_e_tE_e,0);

    hypre_ParCSRDataTransformationSign(h_e_tE_e);

    entity_trueEntity = ParUnique(h_e_tE_e, trueEntity_start);

    hypre_ParCSRMatrix * h_e_tE = *entity_trueEntity;
    if (!hypre_ParCSRMatrixCommPkg(h_e_tE))
        hypre_MatvecCommPkgCreate(h_e_tE);

    resetHypreParVectors();
    storeSharedEntitiesIds();

    elag_assert( DebugCheck() == 0 );
}

int SharingMap::DebugCheck()
{
    PARELAG_ASSERT(entity_trueEntity);
    PARELAG_ASSERT(entity_trueEntity_entity);

#ifdef ELAG_DEBUG
    hypre_ParCSRMatrix * h_e_tE_e = *entity_trueEntity_entity;
    hypre_CSRMatrix * diag_ee = hypre_ParCSRMatrixDiag(h_e_tE_e);
    PARELAG_ASSERT(
        hypre_CSRMatrixNumNonzeros(diag_ee) == hypre_CSRMatrixNumRows(diag_ee));
    PARELAG_ASSERT(
        hypre_CSRMatrixNumNonzeros(diag_ee) == hypre_CSRMatrixNumCols(diag_ee));

    double * data_diag_ee = hypre_CSRMatrixData(diag_ee);
    for(int i = 0; i <hypre_CSRMatrixNumNonzeros(diag_ee); ++i)
        PARELAG_ASSERT( fabs(data_diag_ee[i] - 1.) < 1e-9 );
#endif

    hypre_ParCSRMatrix * tmp;
    hypre_ParCSRMatrixTranspose2(*entity_trueEntity, &tmp);
    HypreTraits<hypre_ParCSRMatrix>::unique_ptr_t e_T_e(
        hypre_ParMatmul(*entity_trueEntity, tmp),
        hypre_ParCSRMatrixDestroy);

    int ierr =
        hypre_ParCSRMatrixCompare(e_T_e.get(),*entity_trueEntity_entity,1e-9,1);
    hypre_ParCSRMatrixDestroy(tmp);

    if (ierr)
    {
        std::cerr << "ierr = " << ierr << std::endl;
        hypre_ParCSRMatrixPrintIJ(*entity_trueEntity, 0, 0, "e_Te");
        hypre_ParCSRMatrixPrintIJ(*entity_trueEntity_entity, 0, 0, "e_Te_e");
    }

    //hypre_ParCSRMatrixDestroy(e_T_e);

    return ierr;
}

void SharingMap::SetUp(const SerialCSRMatrix & Pi,
                       const SharingMap & fdof_fdof,
                       const SerialCSRMatrix & P)
{
    elag_assert(Pi.Size() == P.Width());
    elag_assert(Pi.Width() == P.Size());
    elag_assert(Pi.Width() == fdof_fdof.GetLocalSize());
    elag_assert(P.Size() == fdof_fdof.GetLocalSize());

    int localSize = Pi.Size();
    ParPartialSums_AssumedPartitionCheck(Comm_, localSize, entity_start);

    entity_trueEntity_entity =
        fdof_fdof.ParMatmultAB(Pi,P,entity_start,entity_start);
    hypre_ParCSRMatrixDeleteZeros(*entity_trueEntity_entity, 1e-6);

    entity_trueEntity = ParUnique(*entity_trueEntity_entity,trueEntity_start);
    hypre_ParCSRMatrix * h_e_tE = *entity_trueEntity;
    if (!hypre_ParCSRMatrixCommPkg(h_e_tE))
        hypre_MatvecCommPkgCreate(h_e_tE);

    resetHypreParVectors();
    storeSharedEntitiesIds();

    elag_assert( DebugCheck() == 0 );
}

void SharingMap::SetUp(SharingMap & original, SparseMatrix & e_idof)
{
    PARELAG_ASSERT( original.GetLocalSize() == e_idof.Size() );

    int * i_e_idof = e_idof.GetI();
    int * j_e_idof = e_idof.GetJ();
    const int nnz_e_idof = e_idof.NumNonZeroElems();

    Vector ones_e_idof(nnz_e_idof);
    ones_e_idof = 1.;
    Vector tag(nnz_e_idof);
    tag = -1.;
    for(int i(0); i < e_idof.Size(); ++i)
        for(int jpos = i_e_idof[i]; jpos != i_e_idof[i+1]; ++jpos)
        {
            // elag_assert( fabs( tag(jpos) + 1. ) < 1e-9 );
            tag(jpos) = jpos - i_e_idof[i] + 1;
        }

    SparseMatrix e_idof_ones(
        i_e_idof,j_e_idof,ones_e_idof,e_idof.Size(),e_idof.Width());
    SparseMatrix e_idof_tag(
        i_e_idof,j_e_idof,tag,e_idof.Size(),e_idof.Width());

    // Prepare the matrix with all ones
    hypre_ParCSRMatrix * e_tE_e = *(original.entity_trueEntity_entity);
    hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(e_tE_e);
    hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd(e_tE_e);

    double * diag_data = hypre_CSRMatrixData(diag);
    double * diag_offd = hypre_CSRMatrixData(offd);

    const int nnz = std::max(hypre_CSRMatrixNumNonzeros(diag),
                             hypre_CSRMatrixNumNonzeros(offd));
    Vector ones(nnz);
    ones = 1.0;

    hypre_CSRMatrixData(diag) = ones.GetData();
    hypre_CSRMatrixData(offd) = ones.GetData();

    //Do the actual computation
    int nDofs = e_idof.Width();
    ParPartialSums_AssumedPartitionCheck(Comm_, nDofs, entity_start);

    auto ones_tag = original.ParMatmultAtB(
        e_idof_ones,e_idof_tag,entity_start,entity_start);
    auto tag_ones = original.ParMatmultAtB(
        e_idof_tag,e_idof_ones,entity_start,entity_start);

    int ierr = hypre_ParCSRMatrixKeepEqualEntries(*ones_tag, *tag_ones);
    PARELAG_ASSERT(ierr == 0);

    entity_trueEntity_entity = std::move(ones_tag);
    entity_trueEntity = ParUnique(*entity_trueEntity_entity, trueEntity_start);

    hypre_ParCSRMatrix * h_e_tE = *entity_trueEntity;
    if (!hypre_ParCSRMatrixCommPkg(h_e_tE))
        hypre_MatvecCommPkgCreate(h_e_tE);

    resetHypreParVectors();
    storeSharedEntitiesIds();

    elag_assert( DebugCheck() == 0 );

    //RESTORE ORIGINAL POINTERS

    e_idof_ones.LoseData();
    e_idof_tag.LoseData();
    hypre_CSRMatrixData(diag) = diag_data;
    hypre_CSRMatrixData(offd) = diag_offd;
}

SharingMap::~SharingMap()
{
    //hypre_ParVectorDestroy(x_);
    //hypre_ParVectorDestroy(xTrue_);
}

int SharingMap::Synchronize(Array<int> & data) const
{
    elag_assert(GetLocalSize() == data.Size());
    Array<int> trueData(GetTrueLocalSize());

    int ierr;
    ierr = IgnoreNonLocal(data, trueData);

    elag_assert( ierr == 0 );

    ierr = Distribute(trueData, data);

    elag_assert( ierr == 0 );

    return ierr;
}

int SharingMap::Distribute(const Array<int> & trueData, Array<int> & data) const
{
    elag_assert(GetTrueLocalSize() == trueData.Size());
    elag_assert(GetLocalSize() == data.Size());

#if 1
    HelpTrueData_.SetSize(GetTrueLocalSize());
    HelpData_.SetSize( GetLocalSize() );

    for(int i = 0; i < trueData.Size(); ++i)
        HelpTrueData_(i) = static_cast<double>( trueData[i] );

    hypre_ParCSRMatrix * h_e_tE = *entity_trueEntity;

    hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(h_e_tE);
    hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd(h_e_tE);

    double * diag_data = hypre_CSRMatrixData(diag);
    double * diag_offd = hypre_CSRMatrixData(offd);

    int nnz = std::max(hypre_CSRMatrixNumNonzeros(diag),
                       hypre_CSRMatrixNumNonzeros(offd));
    Vector ones(nnz);
    ones = 1.0;

    hypre_CSRMatrixData(diag) = ones.GetData();
    hypre_CSRMatrixData(offd) = ones.GetData();

    int ierr = Distribute(HelpTrueData_, HelpData_);

    hypre_CSRMatrixData(diag) = diag_data;
    hypre_CSRMatrixData(offd) = diag_offd;

    round(HelpData_, data);

#else
    int ierr = hypre_ParCSRMatrixMatvecBoolInt(
        1, entity_trueEntity, const_cast<int *>(trueData.GetData()),
        0, data.GetData() );
#endif
    return ierr;
}

int SharingMap::Distribute(const Vector & trueData, Vector & data) const
{

    hypre_VectorData( hypre_ParVectorLocalVector(xTrue_) ) = trueData.GetData();
    hypre_VectorData( hypre_ParVectorLocalVector(x_) )     = data.GetData();

    int ierr = hypre_ParCSRMatrixMatvec(
        1.,*entity_trueEntity,xTrue_.get(),0.,x_.get());

    hypre_VectorData(hypre_ParVectorLocalVector(xTrue_)) = nullptr;
    hypre_VectorData(hypre_ParVectorLocalVector(x_)) = nullptr;

    return ierr;
}

int SharingMap::IgnoreNonLocal(const Array<int> & data,
                               Array<int> & trueData) const
{
    elag_assert(GetTrueLocalSize() == trueData.Size());
    elag_assert(GetLocalSize() == data.Size());
#if 0
    HelpTrueData_.SetSize(GetTrueLocalSize());
    HelpData_.SetSize( GetLocalSize() );

    for(int i = 0; i < data.Size(); ++i)
        HelpData_(i) = static_cast<double>( data[i] );

    int ierr = IgnoreNonLocal(HelpData_, HelpTrueData_);

    for(int i = 0; i < trueData.Size(); ++i)
        trueData[i] = static_cast<int>( HelpTrueData_(i) + 0.5 );
#else
    hypre_ParCSRMatrix * h_e_tE = *entity_trueEntity;
    hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(h_e_tE);

    // Check that all entries are 1.
    {
        double * v = hypre_CSRMatrixData(diag);
        int nnz = hypre_CSRMatrixNumNonzeros(diag);
        for(double * end = v+nnz; v < end; ++v)
            elag_assert(fabs( fabs(*v) - 1.) < 1.e-10 );
    }
    const int * i_diag = hypre_CSRMatrixI(diag);
    const int * j_diag = hypre_CSRMatrixJ(diag);
    int nrows = hypre_CSRMatrixNumRows(diag);
    trueData = 0;

    int val = 0;
    for(int i = 0; i < nrows; ++i)
    {
        val = data[i];
        for(int jpos = i_diag[i]; jpos < i_diag[i+1]; ++jpos)
            trueData[ j_diag[jpos] ] += val;
    }

    int ierr = 0;
#endif
    return ierr;

}

int SharingMap::IgnoreNonLocal(const Vector & data, Vector & trueData) const
{
    hypre_VectorData(hypre_ParVectorLocalVector(xTrue_)) = trueData.GetData();
    hypre_VectorData(hypre_ParVectorLocalVector(x_))     = data.GetData();

    hypre_ParCSRMatrix * h_e_tE = *entity_trueEntity;
    hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(h_e_tE);

    int ierr = hypre_CSRMatrixMatvecT(
        1.,diag,hypre_ParVectorLocalVector(x_),
        0.,hypre_ParVectorLocalVector(xTrue_));

    hypre_VectorData( hypre_ParVectorLocalVector(xTrue_) ) = nullptr;
    hypre_VectorData( hypre_ParVectorLocalVector(x_) )     = nullptr;

    return ierr;
}

int SharingMap::Assemble(const Array<int> & data, Array<int> & trueData) const
{
    elag_assert(GetTrueLocalSize() == trueData.Size());
    elag_assert(GetLocalSize() == data.Size());

#if 1
    HelpTrueData_.SetSize(GetTrueLocalSize());
    HelpData_.SetSize( GetLocalSize() );

    for(int i = 0; i < data.Size(); ++i)
        HelpData_(i) = static_cast<double>( data[i] );

    int ierr = Assemble(HelpData_, HelpTrueData_);

    round(HelpTrueData_, trueData);

#else
    int ierr = hypre_ParCSRMatrixMatvecTBoolInt(
        1.,entity_trueEntity,x_, 0.,xTrue_);
#endif

    return ierr;

}

int SharingMap::Assemble(const Vector & data, Vector & trueData) const
{
    hypre_VectorData(hypre_ParVectorLocalVector(xTrue_) ) = trueData.GetData();
    hypre_VectorData(hypre_ParVectorLocalVector(x_) )     = data.GetData();

    hypre_ParCSRMatrix * h_e_tE = *entity_trueEntity;
    int ierr = hypre_ParCSRMatrixMatvecT(1.,h_e_tE,x_.get(),0.,xTrue_.get());

    hypre_VectorData(hypre_ParVectorLocalVector(xTrue_)) = nullptr;
    hypre_VectorData(hypre_ParVectorLocalVector(x_))     = nullptr;

    return ierr;
}

unique_ptr<ParallelCSRMatrix> SharingMap::ParMatmultAB(
    const SerialCSRMatrix & A, const SerialCSRMatrix & B,
    const Array<int> & row_starts_A,
    const Array<int> & col_starts_B) const
{
    using csrptr_t = HypreTraits<hypre_ParCSRMatrix>::unique_ptr_t;
    elag_assert(A.Size()==row_starts_A[AssumedPID_+1]-row_starts_A[AssumedPID_]);
    elag_assert(A.Width()==GetLocalSize());
    elag_assert(B.Size()==GetLocalSize());
    elag_assert(B.Width()==col_starts_B[AssumedPID_+1]-col_starts_B[AssumedPID_]);

    ParallelCSRMatrix Ad(
        Comm_,row_starts_A.Last(),entity_start.Last(),
        const_cast<int *>(row_starts_A.GetData()),
        const_cast<int *>(entity_start.GetData()),
        const_cast<SerialCSRMatrix *>(&A));

    ParallelCSRMatrix Bd(
        Comm_,entity_start.Last(),col_starts_B.Last(),
        const_cast<int *>(entity_start.GetData()),
        const_cast<int *>(col_starts_B.GetData()),
        const_cast<SerialCSRMatrix *>(&B));

    csrptr_t tmp(hypre_ParMatmul(Ad,*entity_trueEntity_entity),
                 hypre_ParCSRMatrixDestroy);
    hypre_MatvecCommPkgCreate(tmp.get());
    csrptr_t out(hypre_ParMatmul(tmp.get(),Bd),hypre_ParCSRMatrixDestroy);
    hypre_MatvecCommPkgCreate(out.get());

    //hypre_BoomerAMG gives by default ownership of the column/row
    //partitioning to the result. We don't want that :)
    hypre_ParCSRMatrixSetRowStartsOwner(out.get(),0);
    hypre_ParCSRMatrixSetColStartsOwner(out.get(),0);

    return make_unique<ParallelCSRMatrix>(out.release());
}

unique_ptr<ParallelCSRMatrix> SharingMap::ParMatmultAtB(
    SerialCSRMatrix & A,SerialCSRMatrix & B,
    const Array<int> & col_starts_A,const Array<int> & col_starts_B) const
{
    hypre_ParCSRMatrix * out;

    ParallelCSRMatrix Ad(
        Comm_,entity_start.Last(),col_starts_A.Last(),
        const_cast<int *>(entity_start.GetData()),
        const_cast<int *>(col_starts_A.GetData()),&A);

    ParallelCSRMatrix Bd(
        Comm_,entity_start.Last(),col_starts_B.Last(),
        const_cast<int *>(entity_start.GetData()),
        const_cast<int *>(col_starts_B.GetData()),&B);

    hypre_BoomerAMGBuildCoarseOperator(Ad,*entity_trueEntity_entity,Bd,&out);

    // hypre_BoomerAMG gives by default ownership of the column/row
    // partitioning to the result. We don't want that :)
    hypre_ParCSRMatrixSetRowStartsOwner(out,0);
    hypre_ParCSRMatrixSetColStartsOwner(out,0);

    return make_unique<ParallelCSRMatrix>(out);
}

unique_ptr<ParallelCSRMatrix> SharingMap::ParMatmultAtA(
    SerialCSRMatrix & A, const Array<int> & col_starts_A) const
{
    elag_assert(A.Size() == GetLocalSize() );

    hypre_ParCSRMatrix * out;

    ParallelCSRMatrix Ad(Comm_,
                         entity_start.Last(),
                         col_starts_A.Last(),
                         const_cast<int *>(entity_start.GetData()),
                         const_cast<int *>(col_starts_A.GetData()),
                         &A);

    hypre_BoomerAMGBuildCoarseOperator(Ad,*entity_trueEntity_entity,Ad,&out);

    // hypre_BoomerAMG gives by default ownership of the column/row
    // partitioning to the result. We don't want that :)
    hypre_ParCSRMatrixSetRowStartsOwner(out,0);
    hypre_ParCSRMatrixSetColStartsOwner(out,0);

    return make_unique<ParallelCSRMatrix>(out);
}

int SharingMap::IsShared(int localId) const
{
    hypre_ParCSRMatrix * h_e_tE = *entity_trueEntity,
        *h_e_tE_e = *entity_trueEntity_entity;

    hypre_CSRMatrix * diag_e_tE
        = hypre_ParCSRMatrixDiag(h_e_tE);
    hypre_CSRMatrix * offd_e_tE_e
        = hypre_ParCSRMatrixOffd(h_e_tE_e);

    HYPRE_Int * i_diag_e_tE
        = hypre_CSRMatrixI(diag_e_tE);
    HYPRE_Int * i_offd_e_tE_e
        = hypre_CSRMatrixI(offd_e_tE_e);

    int is_shared = 0;

    if( i_offd_e_tE_e[localId+1] - i_offd_e_tE_e[localId])
        is_shared = (i_diag_e_tE[localId+1] - i_diag_e_tE[localId]) ? 1 : -1;

    return is_shared;
}

int SharingMap::GetNumberShared() const
{
    return sharedEntityIds.Size();
}

int SharingMap::GetNumberSharedOwned() const
{
    return nOwnedSharedEntities_;
}

int SharingMap::GetNumberSharedNotOwned() const
{
    return sharedEntityIds.Size() - nOwnedSharedEntities_;
}

const Array<int> & SharingMap::SharedEntitiesId() const
{
    return sharedEntityIds;
}
void SharingMap::ViewOwnedSharedEntitiesId(Array<int> & ownedSharedId)
{
    ownedSharedId.MakeRef(sharedEntityIds.GetData(), nOwnedSharedEntities_);
}
void SharingMap::ViewNotOwnedSharedEntitiesId(Array<int> & notOwnedSharedId)
{
    notOwnedSharedId.MakeRef(sharedEntityIds.GetData() + nOwnedSharedEntities_,
                             sharedEntityIds.Size() - nOwnedSharedEntities_);
}

unique_ptr<SerialCSRMatrix> Distribute(
    SharingMap &,
    ParallelCSRMatrix &,
    SharingMap &)
{
    PARELAG_NOT_IMPLEMENTED();
    return nullptr;
}

unique_ptr<ParallelCSRMatrix> IgnoreNonLocalRange(
    const SharingMap & range,
    SerialCSRMatrix & A,
    const SharingMap & domain )
{
    elag_assert(range.Comm_ == domain.Comm_);
    elag_assert(A.Size() == range.GetLocalSize());
    elag_assert(A.Width() == domain.GetLocalSize());

    ParallelCSRMatrix Bdiag(
        domain.Comm_,range.GetGlobalSize(),domain.GetGlobalSize(),
        const_cast<int *>(range.entity_start.GetData()),
        const_cast<int *>(domain.entity_start.GetData()),&A);

    return IgnoreNonLocalRange(
        *(range.entity_trueEntity),Bdiag,*(domain.entity_trueEntity));
}

#ifdef ParELAG_ENABLE_PETSC
unique_ptr<PetscParMatrix> AssemblePetsc(
    const SharingMap & range,
    SerialCSRMatrix & A,
    const SharingMap & domain,
    Operator::Type tid)
{
    PARELAG_ASSERT(range.Comm_ == domain.Comm_);
    PARELAG_ASSERT(A.Size() == range.GetLocalSize());
    PARELAG_ASSERT(A.Width() == domain.GetLocalSize());

    PetscParMatrix Bdiag(domain.Comm_, range.GetGlobalSize(),
                         domain.GetGlobalSize(),
                         const_cast<int*>(range.entity_start.GetData()),
                         const_cast<int*>(domain.entity_start.GetData()),
                         &A, tid);

    PetscParMatrix p_range_entity_trueEntity(range.entity_trueEntity.get(), tid);
    PetscParMatrix p_domain_entity_trueEntity(
        domain.entity_trueEntity.get(), tid);

    PetscParMatrix * out = RAP(&p_range_entity_trueEntity,
                               &Bdiag, &p_domain_entity_trueEntity);
    return unique_ptr<PetscParMatrix>(out);
}
#endif

unique_ptr<ParallelCSRMatrix> Assemble(
    const SharingMap & range,
    SerialCSRMatrix & A,
    const SharingMap & domain)
{
    PARELAG_ASSERT(range.Comm_ == domain.Comm_);
    PARELAG_ASSERT(A.Size() == range.GetLocalSize());
    PARELAG_ASSERT(A.Width() == domain.GetLocalSize());

    ParallelCSRMatrix Bdiag(
        domain.Comm_,range.GetGlobalSize(), domain.GetGlobalSize(),
        const_cast<int *>(range.entity_start.GetData()),
        const_cast<int *>(domain.entity_start.GetData()), &A);

    // FIXME: Shouldn't hypre deal with this??
    {
        hypre_ParCSRMatrix * r_e_tE = *(range.entity_trueEntity);

        if (!hypre_ParCSRMatrixCommPkg(r_e_tE))
        {
            hypre_MatvecCommPkgCreate(r_e_tE);
        }
    }
    hypre_ParCSRMatrix * out;
    hypre_BoomerAMGBuildCoarseOperator(*(range.entity_trueEntity),
                                       Bdiag,
                                       *(domain.entity_trueEntity),
                                       &out);
    hypre_ParCSRMatrixSetNumNonzeros(out);

    // Warning: hypre_BoomerAMGBuildCoarseOperator steals the
    //   col_starts from P (even if it does not own them)!
    hypre_ParCSRMatrixSetRowStartsOwner(out,0);
    hypre_ParCSRMatrixSetColStartsOwner(out,0);

    return make_unique<ParallelCSRMatrix>(out);
}

unique_ptr<SerialCSRMatrix> AssembleNonLocal(
    SharingMap & range,
    SerialCSRMatrix & A,
    SharingMap & domain )
{
    using parcsr_ptr_t = HypreTraits<hypre_ParCSRMatrix>::unique_ptr_t;
    using sercsr_ptr_t = HypreTraits<hypre_CSRMatrix>::unique_ptr_t;

    elag_assert(range.Comm_ = domain.Comm_);

    // NOTE: We need to handle both the case in which range and domain
    // are the same or different objects.
    //
    // To avoid Segmentation Faults it is important that Step 1-2-3
    // are performed in such order.

    // (1) Extract the diag of the matrix.
    hypre_ParCSRMatrix *d_h_e_tE_e = *(domain.entity_trueEntity_entity),
        *r_h_e_tE_e = *(range.entity_trueEntity_entity);

    hypre_CSRMatrix
        *dom_diag = hypre_ParCSRMatrixDiag(d_h_e_tE_e),
        *ran_diag = hypre_ParCSRMatrixDiag(r_h_e_tE_e);

    // (2) Create a zeros block to replace the diag block.
    sercsr_ptr_t dom_dzeros(hypre_ZerosCSRMatrix(domain.GetLocalSize(),
                                                 domain.GetLocalSize()),
                            hypre_CSRMatrixDestroy),
        ran_dzeros(hypre_ZerosCSRMatrix(range.GetLocalSize(),
                                        range.GetLocalSize()),
                   hypre_CSRMatrixDestroy);

    // (3) Replace the diag blocks and set to one the entries in the offd
    hypre_ParCSRMatrixDiag(d_h_e_tE_e) = dom_dzeros.get();
    hypre_ParCSRMatrixDiag(r_h_e_tE_e) = ran_dzeros.get();


    ParallelCSRMatrix Bdiag(domain.Comm_,
                            range.GetGlobalSize(),
                            domain.GetGlobalSize(),
                            range.entity_start.GetData(),
                            domain.entity_start.GetData(),
                            &A);

    parcsr_ptr_t offd_A(hypre_ParMatmul(r_h_e_tE_e, Bdiag),
                        hypre_ParCSRMatrixDestroy);
    parcsr_ptr_t offd_A_offd(hypre_ParMatmul(offd_A.get(), d_h_e_tE_e),
                             hypre_ParCSRMatrixDestroy);

    hypre_CSRMatrix * res = hypre_ParCSRMatrixDiag(offd_A_offd);
    SparseMatrix tmp(
        hypre_CSRMatrixI(res),hypre_CSRMatrixJ(res),hypre_CSRMatrixData(res),
        hypre_CSRMatrixNumRows(res),hypre_CSRMatrixNumCols(res),
        false,false,true);

    auto out = ToUnique(Add(A, tmp));

    hypre_ParCSRMatrixDiag(d_h_e_tE_e) = dom_diag;
    hypre_ParCSRMatrixDiag(r_h_e_tE_e) = ran_diag;

    return out;
}

void SharingMap::round(const Vector & d, Array<int> & a) const
{
    const int n = d.Size();

    elag_assert(n == a.Size());
    const double * dd = d.GetData();
    int * aa = a.GetData();

    for(int i = 0; i < n; ++i)
        aa[i] = static_cast<int>( ::round( dd[i] ) );
}

unique_ptr<HypreParVector>
SharingMap::TrueDataToParTrueData(Vector trueData)
{
    return make_unique<HypreParVector>(
        Comm_,trueEntity_start[AssumedNumProc_],trueData.GetData(),
        trueEntity_start.GetData());
}
}//namespace parelag
