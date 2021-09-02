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

// TODO: MFEM error clean.
// TODO: Fix all unique_ptr constructors that take a unique_ptr

#include "DeRhamSequence.hpp"

#include "hypreExtension/hypreExtension.hpp"
#include "linalg/dense/ParELAG_InnerProduct.hpp"
#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "linalg/solver_core/ParELAG_SaddlePointSolver.hpp"
#include "linalg/utilities/ParELAG_SubMatrixExtraction.hpp"
#include "linalg/dense/ParELAG_SVDCalculator.hpp"
#include "structures/SharedEntityCommunication.hpp"
#include "utilities/MemoryUtils.hpp"
#include "utilities/mpiUtils.hpp"

#include <cmath>

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

constexpr double DEFAULT_EQUALITY_TOL = 1e-9;
constexpr double LOOSE_EQUALITY_TOL = 1e-6;

DeRhamSequence::DeRhamSequence(
    const std::shared_ptr<AgglomeratedTopology>& topo, int nspaces):
    Comm_(topo->GetComm()),
    jformStart_(topo->Dimensions() - topo->Codimensions()),
    Topo_(topo),
    nForms_(nspaces),
    Dof_(nForms_),
    DofAgg_(nForms_),
    D_(nForms_-1),
    M_{},
    Targets_(0),
    PV_Traces_(nForms_),
    LocalTargets_{},
    P_(nForms_),
    Pi_(nForms_),
    CoarserSequence_{},
    FinerSequence_{},
    SVD_Tolerance_(1e-9),
    SmallestEntry_(std::numeric_limits<double>::epsilon())
{
    DeRhamSequence_os << " SVD Tolerance: " << SVD_Tolerance_ << std::endl;
    DeRhamSequence_os << " Small Entry: "   << SmallestEntry_ << std::endl;
}

void DeRhamSequence::SetTargets(const Array<MultiVector *> & targets)
{
    PARELAG_TEST_FOR_EXCEPTION(
        targets.Size() != nForms_,
        std::runtime_error,
        "Size of target does not match." );

    // (1) We copy the targets
    Targets_.resize(nForms_);
    for (int jform(jformStart_); jform < nForms_; ++jform)
    {
        if (targets[jform] == nullptr)
        {
            Targets_[jform] = make_unique<MultiVector>(0,Dof_[jform]->GetNDofs());
        }
        else
        {
            PARELAG_TEST_FOR_EXCEPTION(
                targets[jform]->Size() != Dof_[jform]->GetNDofs(),
                std::logic_error,
                "Target for form " << jform << " has wrong size! "
                "Size is " << targets[jform]->Size() <<
                "; should be " << Dof_[jform]->GetNDofs() << ". :(");

            Targets_[jform] = make_unique<MultiVector>();
            targets[jform]->Copy(*Targets_[jform]);
        }
    }
    // TODO check the exactness of the targets
}

// FIXME (trb 12/24/15): Add a "StealTargets" function? In the above
// function, they are deep-copied, and only in one of the examples are
// the original targets re-used later in the program, and then only for comparison purposes. Perhaps also a "GetTarget()" or "ViewTarget()" function?

void DeRhamSequence::AgglomerateDofs()
{
    for (int jform = jformStart_; jform < nForms_; ++jform)
    {
        if (!DofAgg_[jform])
        {
            DofAgg_[jform] = make_unique<DofAgglomeration>(Topo_,Dof_[jform].get());
#ifdef ELAG_DEBUG
            DofAgg_[jform]->CheckAdof();
#endif
        }
    }
}

void DeRhamSequence::SetLocalTargets(
    AgglomeratedTopology::Entity codim, int jform,
    const Array<MultiVector *>& localfunctions)
{
    const int nAEs = localfunctions.Size();
    AgglomerateDofs();

    const int nDim = Topo_->Dimensions();
    const int idx = (nDim-jform)*(nForms_-jform)/2 + codim;

    // Assertions!
    elag_assert(DofAgg_[jform]);
    elag_assert(DofAgg_[jform]->GetAEntityDof(codim).Height() == nAEs);
    elag_assert(Topo_);
    elag_assert(Topo_->CoarserTopology());

    LocalTargets_[idx].resize(nAEs);

    // Just copy the agglomerated entity "traces".
    for(int iAE=0; iAE < nAEs; ++iAE)
    {
        elag_assert(AgglomeratedTopology::ELEMENT != codim ||
                    !Topo_->CoarserTopology()->EntityTrueEntity(codim).IsShared(iAE));

        const int nAEdofs
            = DofAgg_[jform]->GetAEntityDof(codim).RowSize(iAE);

        if (localfunctions[iAE] == nullptr)
        {
            LocalTargets_[idx][iAE] = make_unique<MultiVector>(0, nAEdofs);
        }
        else
        {
            elag_assert(localfunctions[iAE]->Size() == nAEdofs);
            LocalTargets_[idx][iAE] = make_unique<MultiVector>();
            localfunctions[iAE]->Copy(*(LocalTargets_[idx][iAE]));
        }
    }
}

void DeRhamSequence::OwnLocalTargets(
    AgglomeratedTopology::Entity codim,
    int jform,
    std::vector<unique_ptr<MultiVector>> localfunctions)
{
    const int nDim = Topo_->Dimensions();
    const int idx = (nDim-jform)*(nForms_-jform)/2 + codim;

    LocalTargets_[idx] = std::move(localfunctions);

#ifdef ELAG_DEBUG
    AgglomerateDofs();

    const auto locfuncsize = LocalTargets_[idx].size();
    using size_type = decltype(locfuncsize);

    PARELAG_ASSERT(DofAgg_[jform]);
    const auto table_height = DofAgg_[jform]->GetAEntityDof(codim).Height();
    PARELAG_ASSERT(static_cast<size_type>(table_height) == locfuncsize);
    PARELAG_ASSERT(Topo_);
    PARELAG_ASSERT(Topo_->CoarserTopology());
    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    for(auto iAE=decltype(locfuncsize){0}; iAE < locfuncsize; ++iAE)
    {
        PARELAG_ASSERT(at_elem != codim ||
                       !Topo_->CoarserTopology()->EntityTrueEntity(codim).IsShared(iAE));
        PARELAG_ASSERT(LocalTargets_[idx][iAE]);
        PARELAG_ASSERT(LocalTargets_[idx][iAE]->Size() ==
                       DofAgg_[jform]->GetAEntityDof(codim).RowSize(iAE));
    }
#endif
}

void DeRhamSequence::PopulateLocalTargetsFromForm(const int jform)
{
    elag_assert(jform == 0); // this routine may be okay for other forms...

    AgglomerateDofs();
    populateLowerCodims(jform);
    targetDerivativesInForm(jform+1);
    populateLowerCodims(jform+1);
}

void DeRhamSequence::populateLowerCodims(const int jform)
{
    // TODO: In its current shape this should work fine on one
    //       processor, but care is needed for the parallel case when
    //       traces on faces and edges come from multiple processors.
    //       In the H(curl) and H(div) cases (unlike the H1 case) we
    //       also need to be extra careful since there is also
    //       orientation associated with faces (for H(div) and
    //       H(curl)) and edges (for H(curl)).  This should be
    //       automatic within a processor, since \b localfunctions on
    //       each agglomerated element are expressed respecting the
    //       within processor orientation and dofs for each
    //       agglomerated entity have the within processor orientation
    //       (this is because \b AEntity_dof in \b DofAgg_ always have
    //       1.0 as data entries). So locally on the processor we can
    //       simply collect all traces from all agglomerated elements
    //       by a simple restriction (which is just a numbering
    //       thing). On shared agglomerated entities the within entity
    //       numberings on separate processors will most likely not
    //       coincide (for H1, H(div), and H(curl)) and the
    //       orientations within processors will also be different
    //       (for H(div) and H(curl)).  This is where care is needed
    //       when communicating.

    if (nForms_ - 1 <= jform || jform < jformStart_)
        return;

    int codim = 0;

    const int nDim = Topo_->Dimensions();
    const int idx = (nDim-jform)*(nForms_-jform)/2 + codim;

    const std::vector<unique_ptr<MultiVector>> & localfunctions
        = LocalTargets_[idx];

    const auto locfuncsize = localfunctions.size();
    if (!locfuncsize)
        return;

    elag_assert(DofAgg_[jform]);
    elag_assert(Topo_);
    elag_assert(Topo_->CoarserTopology());
    const SparseMatrix& AElement_dof =
        DofAgg_[jform]->GetAEntityDof((AgglomeratedTopology::Entity)codim);
    elag_assert(static_cast<decltype(locfuncsize)>(AElement_dof.Height()) == locfuncsize);
    const int lndofs = AElement_dof.Width();
    elag_assert(Dof_[jform]->GetNDofs() == lndofs);
    Array<int> dofMarker(lndofs);
#ifdef ELAG_DEBUG
    dofMarker = -1;
    for (auto i=decltype(locfuncsize){0}; i < locfuncsize; ++i)
    {
        elag_assert(localfunctions[i]);
        elag_assert(localfunctions[i]->Size() == AElement_dof.RowSize(i));
        elag_assert(!Topo_->CoarserTopology()->EntityTrueEntity(codim).IsShared(i));
    }
#endif

    // Process lower "codimensions".
    elag_assert(1 <= nForms_ - jform - 1);
    // XXX: These guys in \b AEntities_AElement will most likely end
    //      up with zero entries, but we count on the sparsity pattern
    //      in the CSR representation to correctly reflect the
    //      topology, even if an entry is with value 0.0.  All we need
    //      is the relations between the lower "codimension" of
    //      agglomerated entities and the largest "codimension"
    //      agglomerated entities (agglomerated elements). There may
    //      be a better way to obtain these relations.
    std::vector<unique_ptr<const SparseMatrix>>
        AEntities_AElement(nForms_-jform-1);
    AEntities_AElement[0].reset(Transpose(Topo_->CoarserTopology()->GetB(0)));
    for (int i=1; i < nForms_ - jform - 1; ++i)
    {
        unique_ptr<const SparseMatrix> tmp_spmat{
            Transpose(Topo_->CoarserTopology()->GetB(i))};
        AEntities_AElement[i].reset(Mult(*tmp_spmat, *AEntities_AElement[i-1]));
    }
    Array<int> nlocalvects;
    for (codim = 1; codim < nForms_ - jform; ++codim)
    {
        const SparseMatrix& AEntity_dof =
            DofAgg_[jform]->GetAEntityDof((AgglomeratedTopology::Entity)codim);
        const SparseMatrix * const AEntity_AElement =
            AEntities_AElement[codim - 1].get();
        const int nAEs = AEntity_dof.Height();
        elag_assert(Topo_->CoarserTopology()->EntityTrueEntity(codim).GetLocalSize() == nAEs);
        nlocalvects.SetSize(nAEs);

        SharedEntityCommunication<MultiVector>
            sec(Comm_,*(Topo_->CoarserTopology()->EntityTrueEntity(codim).get_entity_trueEntity()));
        sec.ReducePrepare();

        // loop over codimension entities, eg faces
        for (int i=0; i < nAEs; ++i)
        {
            const int * const AEntityDofs = AEntity_dof.GetRowColumns(i);
            const int AEntitySize = AEntity_dof.RowSize(i);
            elag_assert(AEntitySize > 0);
            elag_assert((4 == nForms_ && codim < 3) ||
                        (3 == nForms_ && codim < 2) || 1 == AEntitySize);
            const int * const AElements = AEntity_AElement->GetRowColumns(i);
            const int nAElements = AEntity_AElement->RowSize(i);
            elag_assert(codim > 1 || 2 >= nAElements);
            nlocalvects[i] = 0;
            for (int j=0; j < nAElements; ++j)
            {
                elag_assert(decltype(locfuncsize)(AElements[j]) < locfuncsize);
                nlocalvects[i] += localfunctions[AElements[j]]->NumberOfVectors();
            }

            MultiVector localtars(nlocalvects[i], AEntitySize);

            // Obtain the actual restrictions from all local (for a processor)
            // agglomerated elements sharing the current agglomerated entity.
            // (eg. loop over all elements that contain this face)
            int collected_vects = 0;
            for (int j=0; j < nAElements; ++j)
            {
                const int AElement = AElements[j];
                elag_assert(decltype(locfuncsize)(AElement) < locfuncsize);
                const MultiVector * const localfuncs =
                    localfunctions[AElement].get();
                const int nvects = localfuncs->NumberOfVectors();
                const int * const AElementDofs =
                    AElement_dof.GetRowColumns(AElement);
                const int AElementSize = AElement_dof.RowSize(AElement);
                elag_assert(AEntitySize <= AElementSize);
                elag_assert(localfuncs->Size() == AElementSize);
                // NOTE: Dof orientation should not be an issue within processor,
                //       since \b localfunctions on each agglomerated element
                //       are expressed respecting within processor orientation
                //       and dofs for each agglomerated entity have the within
                //       processor orientation (this is because \b AEntity_dof in
                //       \b DofAgg_ always have 1.0 as data entries). So locally
                //       on the processor we can simply collect all traces from
                //       all agglomerated elements by a simple restriction (which
                //       is just a numbering thing). On shared agglomerated
                //       entities the within entity numberings
                //       on separate processors will most likely not coincide
                //       (for H1, H(div), and H(curl)) and the orientations
                //       within processors will also be different (for H(div)
                //       and H(curl)).
                //       This is where care is needed when communicating.
                // XXX: Everything below in this for loop is NOT very efficiently
                //      implemented, I think. First, we may get in a situation
                //      where we compute the portion of the \b dofMarker many
                //      times. Second, we may have memory locality issues and
                //      more cache misses.
                for (int k=0; k < AElementSize; ++k)
                {
                    elag_assert(-1 == dofMarker[AElementDofs[k]]);
                    dofMarker[AElementDofs[k]] = k;
                    elag_assert(AElement_dof.GetRowEntries(AElement)[k] > 0.);
                }
                PARELAG_ASSERT(collected_vects + nvects <= nlocalvects[i]);
                for (int k=0; k < nvects; ++k)
                {
                    double *tdata =
                        localtars.GetDataFromVector(collected_vects + k);
                    const double * const sdata = localfuncs->GetDataFromVector(k);
                    for (int l=0; l < AEntitySize; ++l)
                    {
                        elag_assert(AEntityDofs[l] < lndofs);
                        elag_assert(0 <= dofMarker[AEntityDofs[l]] &&
                                    dofMarker[AEntityDofs[l]] < AElementSize);
                        *(tdata++) = sdata[dofMarker[AEntityDofs[l]]];
                        elag_assert(AEntity_dof.GetRowEntries(i)[l] > 0.);
                    }
                }
                collected_vects += nvects;
#ifdef ELAG_DEBUG
                for (int k=0; k < AElementSize; ++k)
                {
                    PARELAG_ASSERT(-1 != dofMarker[AElementDofs[k]]);
                    dofMarker[AElementDofs[k]] = -1;
                }
#endif
            }
            elag_assert(collected_vects == nlocalvects[i]);
            sec.ReduceSend(i,localtars);
        }
        // TODO: the implementation here does not communicate
        //       orientations (may only work for H1) but for other
        //       spaces we may need to.
        MultiVector ** temp = sec.Collect();

        const int nDim = Topo_->Dimensions();
        const int idx = (nDim-jform)*(nForms_-jform)/2 + codim;

        LocalTargets_[idx].resize(nAEs);
        for (int i=0; i<nAEs; ++i)
        {
            const int AEntitySize = AEntity_dof.RowSize(i);
            const int num_neighbors = sec.NumNeighbors(i);
            if (sec.OwnedByMe(i))
            {
                int nlocalvects = 0;
                for (int neighbor=0; neighbor<num_neighbors; ++neighbor)
                {
                    PARELAG_ASSERT(temp[i][neighbor].Size() == AEntitySize);
                    nlocalvects += temp[i][neighbor].NumberOfVectors();
                }
                LocalTargets_[idx][i]
                    = make_unique<MultiVector>(nlocalvects, AEntitySize);
                int counter = 0;
                MultiVector * localtar = LocalTargets_[idx][i].get();
                for (int neighbor=0; neighbor<num_neighbors; ++neighbor)
                {
                    const int temp_total_size = temp[i][neighbor].Size() *
                        temp[i][neighbor].NumberOfVectors();
                    PARELAG_ASSERT(counter <
                                   localtar->Size()*localtar->NumberOfVectors());
                    std::memcpy(localtar->GetData() + counter,
                                temp[i][neighbor].GetData(),
                                temp_total_size * sizeof(double));
                    counter += temp_total_size;
                }
                delete [] temp[i];
            }
            else
            {
                LocalTargets_[idx][i] = make_unique<MultiVector>(0,AEntitySize);
                PARELAG_ASSERT(temp[i] == nullptr);
            }
        }
        delete [] temp;

        // sorry Tom, not changing the SharedEntityCommunication interface to
        // handle std::array<std::vector<std::unique_ptr>>>>>> just yet
        temp = new MultiVector*[nAEs];
        for (int i=0; i<nAEs; ++i)
            temp[i] = LocalTargets_[idx][i].get();
        sec.Broadcast(temp);
        delete [] temp;

        /*
        int myid,comm_size;
        MPI_Comm_rank(MPI_COMM_WORLD,&myid);
        MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
        for (int r=0; r<comm_size; ++r)
        {
            if (r==myid)
            {
                std::cout << "[" << r << "]" << std::endl;
                for (int i=0; i<nAEs; ++i)
                {
                    Vector * vec = (Vector*) LocalTargets_[idx][i].get();
                    int vecsize = vec->Size();
                    double norm = LocalTargets_[idx][i]->Norml2();
                    std::cout << "  jform = " << jform << ", codim = " << codim << ", idx = " << idx
                              << ", i = " << i << ", owned = " << sec.OwnedByMe(i)
                              << ", Norml2 = " << norm << ", size = "
                              << LocalTargets_[idx][i]->Size() << ", number vecs = "
                              << LocalTargets_[idx][i]->NumberOfVectors()
                              << ", mfem::Vec size = " << vecsize << std::endl;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        */

        // This block basically undoes the sec.Broadcast(), making the
        // LocalTargets_ a nothing-object on processors that do not own it.
        // I think this does not make sense, but it occasionally reproduces
        // useful previous behavior that resulted from communicating
        // a MultiVector as a Vector and losing its NumberOfVectors() attribute.
        // Now we communicate them correctly and do the nonsensical thing as a
        // postprocessing step.
        for (int i=0; i<nAEs; ++i)
        {
            if (!sec.OwnedByMe(i))
            {
                int mvsize = LocalTargets_[idx][i]->Size();
                LocalTargets_[idx][i]->SetSizeAndNumberOfVectors(
                    mvsize, 0);
            }
        }
    }
}

void DeRhamSequence::targetDerivativesInForm(const int jform)
{
    elag_assert(jform == 1);

    constexpr int codim = 0;
    const int nDim = Topo_->Dimensions();
    const int idx = (nDim-jform)*(nForms_-jform)/2 + codim;
    const int idx1 = (nDim-jform+1)*(nForms_-jform+1)/2 + codim;

    const std::vector<unique_ptr<MultiVector>>& localH1functions
        = LocalTargets_[idx1];

    if (!localH1functions.size())
        return;

    elag_assert(DofAgg_[jform]);
    elag_assert(DofAgg_[jform-1]);
    elag_assert(Topo_);
    elag_assert(Topo_->CoarserTopology());
    const int nAEs = localH1functions.size();
    const SparseMatrix& AElement_H1dof =
        DofAgg_[jform-1]->GetAEntityDof((AgglomeratedTopology::Entity)codim);
    const SparseMatrix& AElement_HCURLdof =
        DofAgg_[jform]->GetAEntityDof((AgglomeratedTopology::Entity)codim);
    elag_assert(AElement_H1dof.Height() == nAEs);
    elag_assert(AElement_HCURLdof.Height() == nAEs);
    elag_assert(LocalTargets_[idx].size() == 0);

    LocalTargets_[idx].resize(nAEs);
    std::vector<unique_ptr<MultiVector>> & localHCURLfunctions
        = LocalTargets_[idx];
    // "Local" because it is for a processor.
    const SparseMatrix * const LocalGrad = GetDerivativeOperator(jform-1);
    elag_assert(LocalGrad);
    elag_assert(LocalGrad->Width() == AElement_H1dof.Width());
    elag_assert(LocalGrad->Height() == AElement_HCURLdof.Width());
    elag_assert(LocalGrad->Width() == Dof_[jform-1]->GetNDofs());
    elag_assert(LocalGrad->Height() == Dof_[jform]->GetNDofs());
    Array<int> dofMarker(LocalGrad->Width());
    dofMarker = -1;

    // Compute gradients on agglomerated elements.
    for(int iAE=0; iAE < nAEs; ++iAE)
    {
        elag_assert(localH1functions[iAE]);
        elag_assert(!localHCURLfunctions[iAE]);
        elag_assert(!Topo_->CoarserTopology()->EntityTrueEntity(codim).IsShared(iAE));
        const int * const AElementH1dofs = AElement_H1dof.GetRowColumns(iAE);
        const int AElementH1size = AElement_H1dof.RowSize(iAE);
        elag_assert(AElementH1size > 0);
        elag_assert(localH1functions[iAE]->Size() == AElementH1size);
        const int * const AElementHCURLdofs = AElement_HCURLdof.GetRowColumns(iAE);
        const int AElementHCURLsize = AElement_HCURLdof.RowSize(iAE);
        elag_assert(AElementHCURLsize > 0);

        if(!localH1functions[iAE]->NumberOfVectors())
        {
            localHCURLfunctions[iAE]
                = make_unique<MultiVector>(0, AElementHCURLsize);
            continue;
        }

        // XXX: I am quite positive that this is the right local version of the operator -- the respective submatrix.
        //      In other parts of the code in this deRham class \b DistributeAgglomerateMatrix() is called for similar purpose.
        //      A quick look in that function seems to show that it does exactly the same as what we did here.
        //      It does not even seem more efficient, since it probably does literally the same thing in terms of processor instructions.
        //      It may turn more efficient due to potentially better memory locality. If somehow it turns out that \b DistributeAgglomerateMatrix()
        //      is actually different from this simple restriction (submatrix extraction) bellow (excluding that \b DistributeAgglomerateMatrix()
        //      performs all work in advance and produces a "DG-like" matrix), then we will be forced to use that function.
        //      However, it seems natural that the values of the H(curl) gradient restricted to an AElement should depend only on
        //      the values of the H(1) functions on (restricted to) the AElement.
        // NOTE: \b LocalGrad provides gradients locally on the processor expressed in terms of H(curl) dofs respecting their on processor orientation
        //       (local orientation but "global" from AE point of view on the processor). Since \b AEGrad is merely a submatrix the resulting local
        //       for the AE gradients will be on the local for the AE H(curl) dofs with their on processor ("global" for the AE) orientation but
        //       ordered in terms of the local for the AE numbering.
        Array<int> rows(const_cast<int *>(AElementHCURLdofs), AElementHCURLsize);
        Array<int> cols(const_cast<int *>(AElementH1dofs), AElementH1size);
        unique_ptr<SparseMatrix> AEGrad{
            ExtractRowAndColumns(*LocalGrad,rows,cols,dofMarker) };

        elag_assert(AEGrad);
        elag_assert(AEGrad->Height() == AElementHCURLsize);
        elag_assert(AEGrad->Width() == AElementH1size);
        /*
          localHCURLfunctions[iAE] = make_unique<MultiVector>(
          localH1functions[iAE]->NumberOfVectors(), AElementHCURLsize );
          MatrixTimesMultiVector(*AEGrad, *(localH1functions[iAE]),
          *(localHCURLfunctions[iAE]));
          */
        localHCURLfunctions[iAE] = MatrixTimesMultiVector(
            *AEGrad,*(localH1functions[iAE]) );
        elag_assert(localHCURLfunctions[iAE]->NumberOfVectors() ==
                    localH1functions[iAE]->NumberOfVectors());
        elag_assert(localHCURLfunctions[iAE]->Size() == AElementHCURLsize);
    }
}

std::shared_ptr<DeRhamSequence> DeRhamSequence::Coarsen()
{
    auto coarser_sequence = std::make_shared<DeRhamSequenceAlg>(
        Topo_->CoarserTopology(), nForms_);
    coarser_sequence->jformStart_ = jformStart_;

    // Update the weak_ptrs
    CoarserSequence_ = coarser_sequence;
    coarser_sequence->FinerSequence_ = shared_from_this();

    // build DofAgg_
    AgglomerateDofs();

    for (int codim = 0; codim < nForms_; ++codim)
    {
        int jform = nForms_-codim-1;
        if (jform < jformStart_)
            break;

        coarser_sequence->Dof_[jform] = make_unique<DofHandlerALG>(
            codim, Topo_->CoarserTopology() );

        P_[jform] = make_unique<SparseMatrix>(
            Dof_[jform]->GetNDofs(),
            EstimateUpperBoundNCoarseDof(jform) );
        Pi_[jform] = make_unique<CochainProjector>(
            coarser_sequence->Topo_.get(),
            coarser_sequence->Dof_[jform].get(),
            DofAgg_[jform].get(),
            P_[jform].get() );

        ComputeCoarseTraces(jform);

        if (codim > 0)
        {
            hFacetExtension(jform);
            if (codim > 1)
            {
                const AgglomeratedTopology::Entity codim_dom_ridge
                    = static_cast<AgglomeratedTopology::Entity>(
                        nForms_ - jform - 3);
                hRidgePeakExtension(jform, codim_dom_ridge);
                if (codim > 2)
                {
                    const AgglomeratedTopology::Entity codim_dom_peak
                        = static_cast<AgglomeratedTopology::Entity>(
                            nForms_ - jform - 4);
                    hRidgePeakExtension(jform, codim_dom_peak);
                }
            }

            // The projector for jform can be finalized now.
            // Also Dof_[jform] is finalized;
            P_[jform]->SetWidth();
            P_[jform]->Finalize();
            coarser_sequence->D_[jform]->SetWidth(P_[jform]->Width());
            coarser_sequence->D_[jform]->Finalize();
        }
        else
        {
            P_[jform]->SetWidth();
            P_[jform]->Finalize();
        }

        Pi_[jform]->ComputeProjector();
        if (jform == nForms_-1)
        {
            coarser_sequence->Dof_[jform]->GetDofTrueDof().SetUp(
                Pi_[jform]->GetProjectorMatrix(),
                Dof_[jform]->GetDofTrueDof(),
                *(P_[jform]) );
        }
        else
        {
            unique_ptr<SparseMatrix> unextendedP
                = getUnextendedInterpolator(jform);
            unique_ptr<SparseMatrix> incompletePi
                = Pi_[jform]->GetIncompleteProjector();
            //HypreExtension::PrintAsHypreParCSRMatrix(Topo_->GetComm(), *unextendedP, "uP");
            //HypreExtension::PrintAsHypreParCSRMatrix(Topo_->GetComm(), *incompletePi, "iPi");
            coarser_sequence->Dof_[jform]->GetDofTrueDof().SetUp(
                *incompletePi,
                Dof_[jform]->GetDofTrueDof(),
                *unextendedP );
        }

#ifdef ELAG_DEBUG
        PARELAG_TEST_FOR_EXCEPTION(
            !coarser_sequence->Dof_[jform]->Finalized(),
            std::runtime_error,
            " Coarser Sequence Dof Handler " << jform
            << " is not finalized!");

        PARELAG_TEST_FOR_EXCEPTION(
            coarser_sequence->Dof_[jform]->GetNDofs() != P_[jform]->Width(),
            std::runtime_error,
            "coarser_sequence->Dof_[jform]->GetNDofs() != P_[jform]->Width()");
#endif

    }

    //(1) We coarsen the targets
    coarser_sequence->Targets_.resize(nForms_);
    for(int jform(jformStart_); jform < nForms_; ++jform)
    {
        coarser_sequence->Targets_[jform] = make_unique<MultiVector>(
            Targets_[jform]->NumberOfVectors(),
            P_[jform]->Width() );
        Pi_[jform]->Project(*(Targets_[jform]),
                            *(coarser_sequence->Targets_[jform]));
    }

    // Coarsen constant one representation
    MultiVector fine_const(L2_const_rep_.GetData(), 1, L2_const_rep_.Size());
    mfem::Vector& coarse_rep = coarser_sequence->GetL2ConstRepresentation();
    coarse_rep.SetSize(coarser_sequence->Dof_[nForms_-1]->GetNDofs());
    MultiVector coarse_const(coarse_rep.GetData(), 1, coarse_rep.Size());
    Pi_[nForms_-1]->Project(fine_const, coarse_const);

    return coarser_sequence;
}

void DeRhamSequence::CheckInvariants()
{
    CheckCoarseMassMatrix();
    CheckTrueCoarseMassMatrix();
    CheckD();
    CheckTrueD();
    CheckDP();
    CheckTrueDP();
    CheckCoarseDerivativeMatrix();
    CheckTrueCoarseDerivativeMatrix();
    CheckPi();
}

void DeRhamSequence::CheckD()
{
    for(int jform(jformStart_); jform < nForms_-1; ++jform)
    {
        PARELAG_TEST_FOR_EXCEPTION(
            D_[jform]->NumNonZeroElems() == 0,
            std::runtime_error,
            "nnz(D_" << jform << ") = 0!");

        PARELAG_TEST_FOR_EXCEPTION(
            D_[jform]->MaxNorm() < LOOSE_EQUALITY_TOL,
            std::runtime_error,
            "maxNorm(D_" << jform << ") = " << D_[jform]->MaxNorm());
    }

    for(int jform(jformStart_); jform < nForms_-2; ++jform)
    {
        unique_ptr<SparseMatrix> DD{Mult(*D_[jform+1],*D_[jform])};
        const double err = DD->MaxNorm();
        PARELAG_TEST_FOR_EXCEPTION(
            err > DEFAULT_EQUALITY_TOL,
            std::runtime_error,
            "||D_" << jform+1 << " * D_" << jform << "|| = " << err);
    }
}

void DeRhamSequence::CheckTrueD()
{
    std::vector<unique_ptr<ParallelCSRMatrix>> trueD(nForms_-1);

    for(int jform(jformStart_); jform < nForms_-1; ++jform)
        trueD[jform] = ComputeTrueD(jform);

    for(int jform(jformStart_); jform < nForms_-2; ++jform)
    {
        unique_ptr<ParallelCSRMatrix> DD{
            ParMult(trueD[jform+1].get(),trueD[jform].get()) };

        const double err = hypre_ParCSRMatrixNormlinf(*DD);
        PARELAG_TEST_FOR_EXCEPTION(
            err > DEFAULT_EQUALITY_TOL,
            std::runtime_error,
            "|| D_" << jform+1 << " * D_"<< jform << " || = " << err);
    }
}

void DeRhamSequence::CheckDP()
{
    if (auto coarser_sequence = CoarserSequence_.lock())
    {
        for (int jform(jformStart_); jform < nForms_-1; ++jform)
        {
            SparseMatrix * Pu = P_[jform].get();
            SparseMatrix * Pp = P_[jform+1].get();
            SparseMatrix * Dfine = D_[jform].get();
            SparseMatrix * Dcoarse = coarser_sequence->D_[jform].get();

            unique_ptr<SparseMatrix> DfPu{Mult(*Dfine, *Pu)};
            unique_ptr<SparseMatrix> PpDc{Mult(*Pp, *Dcoarse)};

            std::stringstream name1, name2;
            name1 << "D_{" << jform << ",fine}*P_" << jform;
            name2 << "P_" << jform+1 << "* D_{" << jform << ",coarse}";

            bool out = AreAlmostEqual(*DfPu,
                                      *PpDc,
                                      name1.str(),
                                      name2.str(),
                                      LOOSE_EQUALITY_TOL);
            PARELAG_ASSERT(out);
        }
    }
}

void DeRhamSequence::CheckTrueDP()
{
    if (auto coarser_sequence = CoarserSequence_.lock())
    {
        for (int jform(jformStart_); jform < nForms_-1; ++jform)
        {
            auto Pu = ComputeTrueP(jform),
                Pp = ComputeTrueP(jform+1),
                Dfine = ComputeTrueD(jform),
                Dcoarse = coarser_sequence->ComputeTrueD(jform);

            unique_ptr<ParallelCSRMatrix>
                DfPu{ParMult(Dfine.get(), Pu.get())},
                PpDc{ParMult(Pp.get(), Dcoarse.get())};

                HYPRE_Int ierr = hypre_ParCSRMatrixCompare(*DfPu,*PpDc,DEFAULT_EQUALITY_TOL,1);
                PARELAG_ASSERT(ierr == 0);
        }
    }
}

void DeRhamSequence::CheckCoarseMassMatrix()
{
    if (auto coarser_sequence = CoarserSequence_.lock())
    {
        for (int jform(jformStart_); jform < nForms_; ++jform)
        {
            auto Mfine = ComputeMassOperator(jform);
            auto Mcoarse = coarser_sequence->ComputeMassOperator(jform);

            SparseMatrix * Pj = P_[jform].get();
            unique_ptr<SparseMatrix> Pt{Transpose(*Pj)};

            unique_ptr<SparseMatrix> tmp{Mult(*Pt, *Mfine)};
            unique_ptr<SparseMatrix> Mrap{Mult(*tmp, *Pj)};

            std::stringstream name1, name2;
            name1 << "Mcoarse_" << jform;
            name2 << "Mrap_" << jform;

            PARELAG_ASSERT(AreAlmostEqual(*Mcoarse,
                                          *Mrap,
                                          name1.str(),
                                          name2.str(),
                                          LOOSE_EQUALITY_TOL,
                                          false));
        }
    }
}

void DeRhamSequence::CheckTrueCoarseMassMatrix()
{
    if (auto coarser_sequence = CoarserSequence_.lock())
    {
        for (int jform(jformStart_); jform < nForms_; ++jform)
        {
            auto Mfine = ComputeTrueMassOperator(jform);
            auto Mcoarse = coarser_sequence->ComputeTrueMassOperator(jform);
            auto Pj = this->ComputeTrueP(jform);

            unique_ptr<ParallelCSRMatrix> Mrap{RAP(Mfine.get(), Pj.get())};

            int ierr = hypre_ParCSRMatrixCompare(*Mcoarse, *Mrap,
                                                 DEFAULT_EQUALITY_TOL, 1);
            PARELAG_ASSERT(ierr == 0);
        }
    }
}

void DeRhamSequence::CheckCoarseDerivativeMatrix()
{
    if (auto coarser_sequence = CoarserSequence_.lock())
    {
        for (int jform(jformStart_); jform < nForms_ - 1; ++jform)
        {
            // FIXME: const SparseMatrix& ?
            SparseMatrix * Df = D_[jform].get();
            SparseMatrix * Dc = coarser_sequence->D_[jform].get();
            SparseMatrix * Pu = P_[jform].get();
            const SparseMatrix & Pip = Pi_[jform+1]->GetProjectorMatrix();
            unique_ptr<SparseMatrix> DfPu{Mult(*Df, *Pu)};
            unique_ptr<SparseMatrix> Dc2{Mult(Pip, *DfPu)};

            std::stringstream name1;
            name1 <<"coarseD"<<jform;
            std::stringstream name2;
            name2 <<"galCoarseD"<<jform;
            PARELAG_ASSERT(AreAlmostEqual(*Dc,
                                          *Dc2,
                                          name1.str(),
                                          name2.str(),
                                          DEFAULT_EQUALITY_TOL));
        }
    }
}

void DeRhamSequence::CheckTrueCoarseDerivativeMatrix()
{
    if (auto coarser_sequence = CoarserSequence_.lock())
    {
        for (int jform(jformStart_); jform < nForms_ - 1; ++jform)
        {
            unique_ptr<ParallelCSRMatrix>
                Df = ComputeTrueD(jform),
                Dc = coarser_sequence->ComputeTrueD(jform),
                Pu = ComputeTrueP(jform),
                Pip = ComputeTruePi(jform+1);

            unique_ptr<ParallelCSRMatrix>
                DfPu{ParMult(Df.get(), Pu.get())},
                Dc2{ParMult(Pip.get(), DfPu.get())};

                int ierr = hypre_ParCSRMatrixCompare(*Dc, *Dc2, DEFAULT_EQUALITY_TOL, 1);
                PARELAG_ASSERT(ierr == 0 );
        }
    }
}


void DeRhamSequence::CheckPi()
{
    if (auto coarser_sequence = CoarserSequence_.lock())
    {
        for (int jform(jformStart_); jform < nForms_; ++jform)
        {
            DeRhamSequence_os << "Check Pi of " << jform << std::endl;

            // (1) check that || (I - Pi P) x || = 0 for random x
            Pi_[jform]->CheckInvariants();

            // (2) check how close we are to the Targets_
            MultiVector ct(Targets_[jform]->NumberOfVectors(), P_[jform]->Width());
            MultiVector res(Targets_[jform]->NumberOfVectors(), P_[jform]->Size());
            Pi_[jform]->Project(*(Targets_[jform]), ct, res);

            auto Mg = ComputeMassOperator(jform);
            WeightedInnerProduct dot(*Mg);
            Vector view;

            bool showres(false);
            for (int i(0); i < res.NumberOfVectors(); ++i)
            {
                res.GetVectorView(i, view);
                double view_dot_view = dot(view, view);

                if (view_dot_view > DEFAULT_EQUALITY_TOL)
                    DeRhamSequence_os << "|| t - Pi t || = "
                                      << view_dot_view << std::endl;

                if (view_dot_view > DEFAULT_EQUALITY_TOL)
                    showres = true;
            }

            if (showres)
            {
                show(jform, res);
                ct = 0.0;
                Pi_[jform]->Project(res, ct);

                if (ct.Normlinf() > DEFAULT_EQUALITY_TOL)
                    DeRhamSequence_os << "|| Pi( I - P*Pi ) t || = "
                                      << ct.Normlinf() << std::endl;

                if (jform != nForms_-1)
                {
                    MultiVector dres(res.NumberOfVectors(), GetNumberOfDofs(jform+1));
                    MatrixTimesMultiVector(*(D_[jform]), res, dres);

                    if(dres.Normlinf() > DEFAULT_EQUALITY_TOL)
                        DeRhamSequence_os << "|| D( I - P*Pi ) t || = "
                                          << dres.Normlinf() << std::endl;
                    show(jform+1, dres);
                }

                Array<int> all_bdr(Topo_->FacetBdrAttribute().Width()),
                    dofMarker(Dof_[jform]->GetNDofs() );
                all_bdr = 1; dofMarker = 0;
                Dof_[jform]->MarkDofsOnSelectedBndr(all_bdr, dofMarker);

                Vector s(Dof_[jform]->GetNDofs() );
                std::copy(dofMarker.GetData(), dofMarker.GetData()+Dof_[jform]->GetNDofs(), s.GetData());
                res.Scale(s);

                if(res.Normlinf() > DEFAULT_EQUALITY_TOL)
                    DeRhamSequence_os << "|| [t - Pi t](boundary) ||_inf = "
                                      << res.Normlinf() << std::endl;
            }
        }
    }
}

void DeRhamSequence::ComputeSpaceInterpolationError(
    int jform,
    const MultiVector & fineVector )
{
    if (const auto finer_sequence = FinerSequence_.lock())
    {
        DeRhamSequence * seq = this;

        while(auto tmp = seq->FinerSequence_.lock())
            seq = tmp.get();

        const int nVectors = fineVector.NumberOfVectors();
        const int fSize = seq->Dof_[jform]->GetNDofs();
        const int cSize = Dof_[jform]->GetNDofs();

        PARELAG_TEST_FOR_EXCEPTION(
            fineVector.Size() != fSize,
            std::logic_error,
            "DeRhamSequence::ComputeSpaceInterpolationError(): "
            "fineVector size is wrong.");

        auto v = make_unique<MultiVector>(nVectors,fSize);
        *v = fineVector;
        auto vhelp = make_unique<MultiVector>(nVectors,seq->P_[jform]->Width());

        // FIXME (trb 12/14/15): Double-check the pointer swap is valid...
        while(seq != this)
        {
            vhelp->SetSizeAndNumberOfVectors(seq->P_[jform]->Width(), nVectors);
            seq->Pi_[jform]->Project(*v, *vhelp);
            std::swap(vhelp,v);
            seq = seq->CoarserSequence_.lock().get();
        }

        PARELAG_TEST_FOR_EXCEPTION(
            v->Size() != cSize,
            std::runtime_error,
            "DeRhamSequence::ComputeSpaceInterpolationError(): "
            "Size of coarse v is wrong.");

        while(auto tmp = seq->FinerSequence_.lock())
        {
            seq = tmp.get();
            vhelp->SetSizeAndNumberOfVectors(seq->P_[jform]->Size(), nVectors);
            MatrixTimesMultiVector(*seq->P_[jform], *v, *vhelp);
            std::swap(vhelp,v);
        }

        PARELAG_TEST_FOR_EXCEPTION(
            v->Size() != fSize,
            std::runtime_error,
            "DeRhamSequence::ComputeSpaceInterpolationError(): "
            "Size of fine v is wrong.");

        MultiVector difference(nVectors, fSize);
        subtract(fineVector, *v, difference);

        auto Mg = seq->ComputeMassOperator(jform);

        Vector view_diff, view_fineV;
        Array<double> L2diff(nVectors), L2fineVector(nVectors);

        DeRhamSequence_os << std::setw(14) << cSize;
        for(int iVect(0); iVect < nVectors; ++iVect)
        {
            difference.GetVectorView(iVect, view_diff);
            L2diff[iVect] = Mg->InnerProduct(view_diff, view_diff);

            const_cast<MultiVector &>(fineVector).GetVectorView(iVect,
                                                                view_fineV);
            L2fineVector[iVect] = Mg->InnerProduct(view_fineV, view_fineV);
            DeRhamSequence_os << std::setw(14)
                              << sqrt(L2diff[iVect] / L2fineVector[iVect]);
        }
        // FIXME (trb 12/14/15): This is originally deleted here. It
        // seems there is a bit of work left to do in this function,
        // so I keep that behavior.
        Mg.reset();

        if (jform < nForms_ - 1)
        {
            auto Wg = seq->ComputeMassOperator(jform+1);
            const int wSize = seq->D_[jform]->Size();
            MultiVector ddiff(nVectors, wSize), dv(nVectors, wSize);
            MatrixTimesMultiVector(*(seq->D_[jform]), difference, ddiff);
            MatrixTimesMultiVector(*(seq->D_[jform]), fineVector, dv);

            Vector view_diff, view_fineV;
            double L2ddiff, L2dv;

            for(int iVect(0); iVect < nVectors; ++iVect)
            {
                ddiff.GetVectorView(iVect, view_diff);
                L2ddiff = Wg->InnerProduct(view_diff, view_diff);

                dv.GetVectorView(iVect, view_fineV);
                L2dv = Wg->InnerProduct(view_fineV, view_fineV);
                if(fabs(L2fineVector[iVect]+L2dv) < DEFAULT_EQUALITY_TOL)
                    L2dv = 1.;
                DeRhamSequence_os << std::setw(14) << sqrt((L2diff[iVect] + L2ddiff)/(L2fineVector[iVect] + L2dv ) );
            }
        }

        DeRhamSequence_os << std::endl;
    }

    if (auto coarser_sequence = CoarserSequence_.lock())
        coarser_sequence->ComputeSpaceInterpolationError(jform, fineVector);
}

unique_ptr<ParallelCSRMatrix>
DeRhamSequence::ComputeTrueD(int jform) const
{
    return IgnoreNonLocalRange(Dof_[jform+1]->GetDofTrueDof(),
                               *(D_[jform]),
                               Dof_[jform]->GetDofTrueDof());
}

unique_ptr<ParallelCSRMatrix>
DeRhamSequence::ComputeTrueD(int jform,
                             Array<int> & ess_label)
{
    auto Dbc = ComputeDerivativeOperator(jform, ess_label);
    return IgnoreNonLocalRange(Dof_[jform+1]->GetDofTrueDof(),
                               *Dbc,
                               Dof_[jform]->GetDofTrueDof());
}

unique_ptr<ParallelCSRMatrix>
DeRhamSequence::ComputeTrueM(int jform)
{
    auto Mj = ComputeMassOperator(jform);
    return Assemble(Dof_[jform]->GetDofTrueDof(),
                    *Mj,
                    Dof_[jform]->GetDofTrueDof());
}

unique_ptr<ParallelCSRMatrix>
DeRhamSequence::ComputeTrueM(int jform, Vector & elemMatrixScaling)
{
    auto Mj = ComputeMassOperator(jform, elemMatrixScaling);
    return Assemble(Dof_[jform]->GetDofTrueDof(),
                    *Mj,
                    Dof_[jform]->GetDofTrueDof());
}

unique_ptr<ParallelCSRMatrix> DeRhamSequence::ComputeTrueP(int jform) const
{
    auto coarser_sequence = CoarserSequence_.lock();
    PARELAG_ASSERT(coarser_sequence);

    return IgnoreNonLocalRange(
        Dof_[jform]->GetDofTrueDof(),
        *(P_[jform]),
        coarser_sequence->Dof_[jform]->GetDofTrueDof());
}

const ParallelCSRMatrix& DeRhamSequence::GetTrueP(int jform) const
{
   if (trueP_.size() == 0)
   {
      trueP_.resize(nForms_);
   }
   if (!trueP_[jform])
   {
      trueP_[jform] = ComputeTrueP(jform);
   }
   return *trueP_[jform];
}

unique_ptr<ParallelCSRMatrix>
DeRhamSequence::ComputeTrueP(int jform, Array<int> & ess_label) const
{
    auto coarser_sequence = CoarserSequence_.lock();
    PARELAG_ASSERT(coarser_sequence);

    auto Pj = GetP(jform, ess_label);
    return IgnoreNonLocalRange(
        Dof_[jform]->GetDofTrueDof(),
        *Pj,
        coarser_sequence->Dof_[jform]->GetDofTrueDof());
}

unique_ptr<ParallelCSRMatrix> DeRhamSequence::ComputeTruePi(int jform)
{
    auto coarser_sequence = CoarserSequence_.lock();
    PARELAG_ASSERT(coarser_sequence);

    SerialCSRMatrix & myPi = GetPi(jform)->GetProjectorMatrix();
    return IgnoreNonLocalRange(
        coarser_sequence->Dof_[jform]->GetDofTrueDof(),
        myPi,
        Dof_[jform]->GetDofTrueDof());
}

const ParallelCSRMatrix& DeRhamSequence::GetTruePi(int jform)
{
   if (truePi_.size() == 0)
   {
      truePi_.resize(nForms_);
   }
   if (!truePi_[jform])
   {
      truePi_[jform] = ComputeTruePi(jform);
   }
   return *truePi_[jform];
}

unique_ptr<ParallelCSRMatrix>
DeRhamSequence::ComputeTrueProjectorFromH1ConformingSpace(int jform) const
{
    PARELAG_NOT_IMPLEMENTED();
#ifdef NOT_YET_IMPLEMENTED_COMPILE
    auto myPi = ComputeProjectorFromH1ConformingSpace(jform);
    SharingMap vectorH1(Dof_[jform]->GetDofTrueDof().GetComm());
    // vectorH1.SetUp(Dof_[jform]->GetDofTrueDof(), nForms_-1);
    return IgnoreNonLocalRange(Dof_[jform]->GetDofTrueDof(), *myPi, vectorH1);
#else
    (void) jform;
    return nullptr;
#endif
}

void DeRhamSequence::ComputeTrueProjectorFromH1ConformingSpace(
    int jform,
    std::unique_ptr<ParallelCSRMatrix>& Pix,
    std::unique_ptr<ParallelCSRMatrix>& Piy,
    std::unique_ptr<ParallelCSRMatrix>& Piz ) const
{
    // FIXME (trb 12/10/15): This is ugly. It seems this is used in
    // some of the HypreExtension classes, and I don't want to touch
    // those yet. Hence, .release()...

    Array2D<SparseMatrix *> myPixyz(1,nForms_-1);
    {
        auto myPi = ComputeProjectorFromH1ConformingSpace(jform);
        ExtractComponents(*myPi, myPixyz, Ordering::byVDIM);
    }

    Pix = IgnoreNonLocalRange(Dof_[jform]->GetDofTrueDof(),
                              *(myPixyz(0,0)),
                              Dof_[0]->GetDofTrueDof());
    delete myPixyz(0,0);
    Piy = IgnoreNonLocalRange(Dof_[jform]->GetDofTrueDof(),
                              *(myPixyz(0,1)),
                              Dof_[0]->GetDofTrueDof());
    delete myPixyz(0,1);
    if(nForms_ == 3)
        Piz = nullptr;
    else
    {
        Piz = IgnoreNonLocalRange(Dof_[jform]->GetDofTrueDof(),
                                  *(myPixyz(0,2)),
                                  Dof_[0]->GetDofTrueDof());
        delete myPixyz(0,2);
    }
}

unique_ptr<SparseMatrix>
DeRhamSequence::ComputeDerivativeOperator(int jform, Array<int> & ess_label)
{
    auto out = DeepCopy(*(D_[jform]));
    Array<int> marker(out->Width());
    marker = 0;
    Dof_[jform]->MarkDofsOnSelectedBndr(ess_label, marker);
    out->EliminateCols(marker);
    return out;
}

unique_ptr<SparseMatrix>
DeRhamSequence::GetP(int jform,Array<int> & ess_label) const
{
    auto coarser_sequence = CoarserSequence_.lock();
    PARELAG_ASSERT(coarser_sequence);

    auto out = DeepCopy(*(P_[jform]));
    Array<int> marker(out->Width());
    marker = 0;
    coarser_sequence->Dof_[jform]->MarkDofsOnSelectedBndr(ess_label, marker);
    out->EliminateCols(marker);
    return out;
}

void DeRhamSequence::DumpD()
{
    for(int jform(jformStart_); jform < nForms_-1; ++jform)
    {
        std::stringstream fname;
        fname << "D" << jform << ".mtx";
        std::ofstream fid(fname.str());
        D_[jform]->PrintMatlab(fid);
    }
}

void DeRhamSequence::DumpP()
{
    if (auto coarser_sequence = CoarserSequence_.lock())
    {
        for (int jform(jformStart_); jform < nForms_; ++jform)
        {
            std::stringstream fname;
            fname << "P" << jform << ".mtx";
            std::ofstream fid(fname.str());
            P_[jform]->PrintMatlab(fid);
        }
    }
    else
    {
        std::cout << "This is already the coarser level" << std::endl;
    }
}

unique_ptr<SparseMatrix>
DeRhamSequence::ComputeLumpedMassOperator(int jform)
{
    if(jform == nForms_-1)
        return ComputeMassOperator(jform);
    else
    {
        const int ndof = Dof_[jform]->GetNDofs();
        auto out = diagonalMatrix(ndof);
        double * a = out->GetData();

        Array<int> rdof, gdof;
        DenseMatrix locmatrix;
        Vector locdiag;
        Vector evals;
        double eval_min;

        // Compute the idx for M_
        const int nDim = Topo_->Dimensions();
        const int idx = (nDim-jform)*(nForms_-jform)/2
            + AgglomeratedTopology::ELEMENT;

        for(int i(0); i < Topo_->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT); ++i)
        {
            Dof_[jform]->GetrDof(AgglomeratedTopology::ELEMENT,i,rdof);
            Dof_[jform]->GetDofs(AgglomeratedTopology::ELEMENT,i,gdof);

            const int nlocdof = rdof.Size();
            locmatrix.SetSize(nlocdof,nlocdof);
            M_[idx]->GetSubMatrix(rdof,rdof,locmatrix);
            locmatrix.GetDiag(locdiag);
            locmatrix.InvSymmetricScaling(locdiag);
            locmatrix.Eigenvalues(evals);
            eval_min = evals.Min();
            for(int idof(0); idof < nlocdof; ++idof)
                a[gdof[idof]] += eval_min * locdiag(idof);
        }
        return out;
    }
}

unique_ptr<SparseMatrix>
DeRhamSequence::ComputeMassOperator(int jform,Vector & elemMatrixScaling)
{
    const int nElements
        = Topo_->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT);

    PARELAG_TEST_FOR_EXCEPTION(
        elemMatrixScaling.Size() != nElements,
        std::runtime_error,
        "DeRhamSequence::ComputeMassOperator(): "
        "elemMatrixScaling has the wrong size. Size is "
        << elemMatrixScaling.Size() << ". Should be " << nElements << ".");

    const int nDim = Topo_->Dimensions();
    const int idx = (nDim-jform)*(nForms_-jform)/2
        + AgglomeratedTopology::ELEMENT;

    const int nrows = M_[idx]->Size();
    const int ncols = M_[idx]->Width();
    const int nnz = M_[idx]->NumNonZeroElems();

    int * i = M_[idx]->GetI();
    int * j = M_[idx]->GetJ();
    std::vector<double> data(nnz);
    std::copy(M_[idx]->GetData(),M_[idx]->GetData()+nnz,data.begin());

    SparseMatrix sM(i,j,data.data(), nrows, ncols, false, false, true);
    {
        int irow(0);
        double s;

        for(int ie(0); ie < nElements; ++ie)
        {
            s = elemMatrixScaling(ie);
            const int nLocalDofs
                = Dof_[jform]->GetEntityNDof(AgglomeratedTopology::ELEMENT,ie);
            for(int iloc(0); iloc < nLocalDofs; ++iloc)
                sM.ScaleRow(irow++, s);
        }
    }

    auto out = Assemble(
        AgglomeratedTopology::ELEMENT,sM,*Dof_[jform],*Dof_[jform] );

    return out;
}

unique_ptr<SparseMatrix>
DeRhamSequence::ComputeLumpedMassOperator(int jform,Vector & elemMatrixScaling)
{
    if(jform == nForms_-1)
        return ComputeMassOperator(jform, elemMatrixScaling);
    else
    {
        const int ndof = Dof_[jform]->GetNDofs();
        auto out = diagonalMatrix(ndof);
        double * a = out->GetData();

        Array<int> rdof, gdof;
        DenseMatrix locmatrix;
        Vector locdiag;
        Vector evals;

        const int nDim = Topo_->Dimensions();
        const int idx = (nDim-jform)*(nForms_-jform)/2
            + AgglomeratedTopology::ELEMENT;

        for(int i(0); i < Topo_->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT); ++i)
        {
            Dof_[jform]->GetrDof(AgglomeratedTopology::ELEMENT,i,rdof);
            Dof_[jform]->GetDofs(AgglomeratedTopology::ELEMENT,i,gdof);
            const int nlocdof = rdof.Size();

            locmatrix.SetSize(nlocdof,nlocdof);
            M_[idx]->GetSubMatrix(rdof,rdof,locmatrix);
            locmatrix.GetDiag(locdiag);
            locmatrix.InvSymmetricScaling(locdiag);
            locmatrix.Eigenvalues(evals);
            const double eval_min = evals.Min() * elemMatrixScaling(i);

            for(int idof(0); idof < nlocdof; ++idof)
                a[gdof[idof]] += eval_min * locdiag(idof);
        }
        return out;
    }
}

DeRhamSequence::~DeRhamSequence()
{
}

int DeRhamSequence::EstimateUpperBoundNCoarseDof(int jform)
{
    auto coarser_sequence = CoarserSequence_.lock();
    PARELAG_ASSERT(coarser_sequence);

    const int nDim = Topo_->Dimensions();
    //const int idx = (nDim-jform)*(nForms_-jform)/2 + codim;

    const int nTargets = Targets_[jform]->NumberOfVectors();
    const int ndTargets
        = (jform < nDim) ? Targets_[jform+1]->NumberOfVectors() : 0;

    const int nCodims = nForms_-jform;
    const int dcodims = (jform < nDim) ? nForms_-jform-1 : 0 ;

    int idx = (nDim-jform)*(nForms_-jform)/2;
    int max_nlocaltargets = 0;
    for (int codim=0; codim < nCodims; ++codim)
    {
        for (auto& mymv : LocalTargets_[idx])
        {
            elag_assert(mymv);
            const int candidate = mymv->NumberOfVectors();
            if (max_nlocaltargets < candidate)
                max_nlocaltargets = candidate;
        }
        ++idx;
    }

    idx = (nDim-jform-1)*(nForms_-jform-1)/2;
    int max_nlocaldtargets = 0;
    for (int codim=0; codim < dcodims; ++codim)
    {
        for (auto& mymv : LocalTargets_[idx])
        {
            elag_assert(mymv);
            const int candidate = mymv->NumberOfVectors();
            if (max_nlocaltargets < candidate)
                max_nlocaltargets = candidate;
        }
        ++idx;
    }

    int ret = 0;

    if(nDim == 3)
    {

        const int
            nv = coarser_sequence->Topo_->GetNumberLocalEntities(AgglomeratedTopology::PEAK),
            ned = coarser_sequence->Topo_->GetNumberLocalEntities(AgglomeratedTopology::RIDGE),
            nf = coarser_sequence->Topo_->GetNumberLocalEntities(AgglomeratedTopology::FACET),
            nel = coarser_sequence->Topo_->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT);

        switch(jform)
        {
        case 0:
            ret = nv + (ned+nf+nel) * (ndTargets+max_nlocaldtargets); //If the gradient of the H1targets is a subset of the Hcurl Targets,
            //then I have 1 dof for each vertex, and one dof for
            // each Hcurl target with 0 curl.
            break;
        case 1:
            ret = ned * (max_nlocaltargets+nTargets+1) + (nf+nel) * (max_nlocaltargets+nTargets+ndTargets+max_nlocaldtargets);
            break;
        case 2:
            ret = nf*(max_nlocaltargets+nTargets+1) + nel*(max_nlocaltargets+nTargets+ndTargets+max_nlocaldtargets);
            break;
        case 3:
            ret = nel*(max_nlocaltargets+nTargets+1);
            break;
        default:
            mfem_error("Not a valid form");
            return -1;
        }

    }
    else //nDimensions == 2
    {
        const int
            nv = coarser_sequence->Topo_->GetNumberLocalEntities(AgglomeratedTopology::RIDGE),
            ned = coarser_sequence->Topo_->GetNumberLocalEntities(AgglomeratedTopology::FACET),
            nel = coarser_sequence->Topo_->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT);

        switch(jform)
        {
        case 0:
            ret = nv + (ned+nel) * (max_nlocaltargets+nTargets+ndTargets+max_nlocaldtargets);
            break;
        case 1:
            ret = ned * (max_nlocaltargets+nTargets+1) + nel * (max_nlocaltargets+nTargets+ndTargets+max_nlocaldtargets);
            break;
        case 2:
            ret = nel*(max_nlocaltargets+nTargets+1);
            break;
        default:
            mfem_error("Not a valid form");
            return -1;
        }
    }

    elag_assert(ret > 0);
    return ret;
}


void DeRhamSequence::Compute0formCoarseTraces()
{
    auto coarser_sequence = CoarserSequence_.lock();
    PARELAG_ASSERT(coarser_sequence);

    int jform(0);
    AgglomeratedTopology::Entity codim = static_cast<AgglomeratedTopology::Entity>(Topo_->Dimensions()-jform);
    const int nDofs = Dof_[jform]->GetNDofs();

    const SparseMatrix & AEntity_dof
        = DofAgg_[jform]->GetAEntityDof(codim);
    const int * const i_AEntity_dof = AEntity_dof.GetI();
    const int * const j_AEntity_dof = AEntity_dof.GetJ();
    const int nAE = AEntity_dof.Size();

    PV_Traces_[jform] = make_unique<Vector>(nDofs);
    Vector & pv(*PV_Traces_[jform]);
    computePVTraces(codim, pv);

    DofHandlerALG * jdof = dynamic_cast<DofHandlerALG*>(coarser_sequence->Dof_[jform].get());

    PARELAG_TEST_FOR_EXCEPTION(
        jdof == nullptr,
        std::runtime_error,
        "DeRhamSequence::ComputeCoarseTraces(): jdof is not a DofHandlerALG.");

    jdof->AllocateDofTypeArray(EstimateUpperBoundNCoarseDof(jform) );

    int fdof;
    for (int iAE(0); iAE < nAE; ++iAE)
    {
        PARELAG_TEST_FOR_EXCEPTION(
            i_AEntity_dof[iAE+1] - i_AEntity_dof[iAE] != 1,
            std::runtime_error,
            "DeRhamSequence::compute0formCoarseTraces: likely topology error, possibly disconnected edge.");

        fdof = j_AEntity_dof[iAE];
        P_[jform]->Set(fdof, iAE, 1.);
        jdof->SetDofType(iAE, DofHandlerALG::RangeTSpace);
        jdof->SetNumberOfInteriorDofsRangeTSpace(codim,iAE,1);
        auto subm = make_unique<DenseMatrix>(1,1);
        (*subm) = 1.;
        Pi_[jform]->SetDofFunctional(codim, iAE, std::move(subm));
    }

    jdof->BuildEntityDofTable(codim);

    int * i_m = new int[nAE+1];
    int * j_m = new int[nAE];
    double * a_m = new double[nAE];

    fillSparseIdentity(i_m, j_m, a_m, nAE);

    const int nDim = Topo_->Dimensions();
    const int idx = (nDim-jform)*(nForms_-jform)/2 + codim;
    coarser_sequence->M_[idx] = make_unique<SparseMatrix>(
        i_m, j_m, a_m, nAE, nAE );
}

void DeRhamSequence::ComputeCoarseTraces(int jform)
{
    DeRhamSequence_os << "Compute Coarse Traces " << jform << std::endl;
    AgglomeratedTopology::Entity codim
        = static_cast<AgglomeratedTopology::Entity>(Topo_->Dimensions()-jform);

    if (jform == 0)
    {
        Compute0formCoarseTraces();
        return;
    }

    const int nDim = Topo_->Dimensions();
    const int idx = (nDim-jform)*(nForms_-jform)/2 + codim;

    if ((Targets_[jform] && Targets_[jform]->NumberOfVectors()) ||
        (LocalTargets_[idx].size()))
        ComputeCoarseTracesWithTargets(jform);
    else
        ComputeCoarseTracesNoTargets(jform);
}

void DeRhamSequence::ComputeCoarseTracesNoTargets(int jform)
{
    auto coarser_sequence = CoarserSequence_.lock();
    PARELAG_ASSERT(coarser_sequence);

    PARELAG_ASSERT(jform > 0);

    AgglomeratedTopology::Entity codim =
        static_cast<AgglomeratedTopology::Entity>(Topo_->Dimensions()-jform);
    const int nDofs = Dof_[jform]->GetNDofs();

    const SparseMatrix & AEntity_dof = DofAgg_[jform]->GetAEntityDof(codim);
    const int * const i_AEntity_dof = AEntity_dof.GetI();
    const int * const j_AEntity_dof = AEntity_dof.GetJ();
    const int nAE   = AEntity_dof.Size();

    PV_Traces_[jform] = make_unique<Vector>(nDofs);
    Vector & pv(*PV_Traces_[jform]);
    computePVTraces(codim, pv);

    DofHandlerALG * jdof = dynamic_cast<DofHandlerALG*>(coarser_sequence->Dof_[jform].get());

    PARELAG_TEST_FOR_EXCEPTION(
        jdof == nullptr,
        std::logic_error,
        "DeRhamSequence::ComputeCoarseTraces(): "
        "Cast of jdof to DofHandlerALG failed!");

    jdof->AllocateDofTypeArray(EstimateUpperBoundNCoarseDof(jform) );

    const int nDim = Topo_->Dimensions();
    const int idx = (nDim-jform)*(nForms_-jform)/2 + codim;

    auto M_d = AssembleAgglomerateMatrix(codim,
                                         *(M_[idx]),
                                         DofAgg_[jform].get(),
                                         DofAgg_[jform].get());

    Array<int> rows;
    Vector local_pv_trace;
    MultiVector basis;

    int * Imass = new int[nAE+1];
    int * Jmass = new int[nAE];
    double * Amass = new double[nAE];

    for (int iAE(0); iAE < nAE; ++iAE)
    {
        int start    = i_AEntity_dof[iAE];
        int end      = i_AEntity_dof[iAE+1];
        int loc_size = end-start;

        for (const int * fdof = j_AEntity_dof+start;
             fdof != j_AEntity_dof+end;
             ++fdof)
        {
            P_[jform]->Set(*fdof, iAE, pv(*fdof));
        }

        jdof->SetDofType(iAE, DofHandlerALG::RangeTSpace);
        jdof->SetNumberOfInteriorDofsRangeTSpace(codim, iAE, 1);
        jdof->SetNumberOfInteriorDofsNullSpace(codim, iAE, 0);

        rows.MakeRef(const_cast<int *>(j_AEntity_dof)+start, loc_size);
        pv.GetSubVector(rows, local_pv_trace);
        auto Mloc = ExtractSubMatrix(*M_d, start, end, start, end);

        double mass = Mloc->InnerProduct(local_pv_trace, local_pv_trace);

        Jmass[iAE] = Imass[iAE] = iAE;
        Amass[iAE] = mass;

        MultiVector basis(local_pv_trace.GetData(), 1, loc_size);
        Pi_[jform]->CreateDofFunctional(codim, iAE, basis, *Mloc);
    }
    Imass[nAE] = nAE;

    jdof->BuildEntityDofTable(codim);
    coarser_sequence->M_[idx] = make_unique<SparseMatrix>(
        Imass, Jmass, Amass, nAE, nAE);
}

/**
   Takes a Vector and MultiVector and piles their columns together Matlab style
   to make a DenseMatrix, as in C = [a | b];

   Problems: this is right in the middle of the DeRhamSequence class but is not
   actually a member function. The data types don't make a lot of sense.
*/
void concatenate(const Vector & a, const MultiVector & b, DenseMatrix & C)
{
    int nrow_a = a.Size();
    int nrow_C = C.Height();
    int ncol_C = C.Width();
#ifdef ELAG_DEBUG
    int nrow_b = b.Size();
    int ncol_b = b.NumberOfVectors();
#endif

    elag_assert(nrow_a == nrow_b);
    elag_assert(nrow_a == nrow_C);
    elag_assert(ncol_C <= ncol_b + 1);
    elag_assert(b.LeadingDimension() == nrow_b);

    const double * a_data = a.GetData();
    double * C_data = C.Data();

    C_data = std::copy(a_data, a_data+nrow_a, C_data);
    if (b.LeadingDimension() == b.Size())
    {
        const double * b_data = b.GetData();
        int hw = nrow_C * (ncol_C-1);
        if (hw > 0)
            std::copy(b_data, b_data + hw, C_data);
    }
    else
    {
        elag_error(1);
    }
}

void DeRhamSequence::ComputeCoarseTracesWithTargets(int jform)
{
/*
 * for each AEntity:
 * (1) get the local PV vector
 * (2) get the local targets
 * (3) orthogonalize the targets
 * (4) do SVD
 * (5) set the correct number of NullSpaceDofs
 * (6) compute XDof tables
 * (7) fill in P
 * (8) Compute coarse Mass Matrix
 */
    auto coarser_sequence = CoarserSequence_.lock();
    PARELAG_ASSERT(coarser_sequence);

    PARELAG_ASSERT(jform > 0);

    AgglomeratedTopology::Entity codim =
        static_cast<AgglomeratedTopology::Entity>(Topo_->Dimensions() - jform);

    SharingMap & AE_TrueAE = Topo_->CoarserTopology()->EntityTrueEntity(codim);
    SharingMap & dof_TrueDof = Dof_[jform]->GetDofTrueDof();

    const int nDofs = Dof_[jform]->GetNDofs();

    const SparseMatrix & AEntity_dof = DofAgg_[jform]->GetAEntityDof(codim);
    const int * const i_AEntity_dof = AEntity_dof.GetI();
    const int * const j_AEntity_dof = AEntity_dof.GetJ();
    const int nAE   = AEntity_dof.Size();
    const int nTargets = Targets_[jform]->NumberOfVectors();
    const int max_nfDof_per_AE = AEntity_dof.MaxRowSize();

    // get PV vector, put in pv
    PV_Traces_[jform] = make_unique<Vector>(nDofs);
    Vector & pv(*(PV_Traces_[jform]));
    computePVTraces(codim, pv);

    const int nDim = Topo_->Dimensions();
    const int idx = (nDim-jform)*(nForms_-jform)/2 + codim;

    auto M_d = AssembleAgglomerateMatrix(codim,
                                         *(M_[idx]),
                                         DofAgg_[jform].get(),
                                         DofAgg_[jform].get());
    Vector diagM(M_d->Size()), scaling(M_d->Size());
    M_d->GetDiag(diagM);
    std::transform(diagM.GetData(),
                   diagM.GetData()+diagM.Size(),
                   scaling.GetData(),
                   static_cast<double (*)(double)>(std::sqrt));

#ifdef ELAG_DEBUG
    if((nDofs != pv.Size()) || (nDofs != Targets_[jform]->Size()) )
        mfem_error("ComputeCoarseTraces #1");

    if(M_d->Size() != M_d->Width() )/*|| M_d->Size() != M_d->NumNonZeroElems() )*/
        mfem_error("ComputeCoarseTraces #3");
#endif

    ElementalMatricesContainer my_p(nAE);
    ElementalMatricesContainer mass(nAE);

    int max_nlocaltargets = 0;
    const auto LocTargSize = LocalTargets_[idx].size();
    if (LocTargSize > 0)
    {
        elag_assert(LocTargSize == decltype(LocTargSize)(nAE));
        for (auto& mymv : LocalTargets_[idx])
        {
            elag_assert(mymv);
            const int candidate = mymv->NumberOfVectors();
            if (max_nlocaltargets < candidate)
                max_nlocaltargets = candidate;
        }
    }

    Vector scaling_view, diagM_view;
    Vector s(nTargets + max_nlocaltargets);

    SVD_Calculator svd;
    svd.setFlagON();
    svd.AllocateOptimalSize(max_nfDof_per_AE, nTargets + max_nlocaltargets);

    double s_max_tol;
    Array<int> AE_dofs_offsets(nAE+1);
    AE_dofs_offsets[0] = 0;

    // Temporary Variable will be overwritten later.
    int * AE_ndofs = AE_dofs_offsets.GetData() + 1;
    Vector loc_pv;
    MultiVector loc_targets;

    //========== FIRST LOOP: Compute SVDs and set AE_ndofs ===========//
    // my_p and mass get filled in here
    for (int iAE(0); iAE < nAE; ++iAE)
    {
        if (AE_TrueAE.IsShared(iAE) == -1 )
        {
            // if the entity is shared and it does not belong to this processor
            AE_ndofs[iAE] = 0;
        }
        else  // if this processor owns the entity
        {
            const int start    = i_AEntity_dof[iAE];
            const int end      = i_AEntity_dof[iAE+1];
            const int loc_size = end-start;
            Array<int> dof_in_AE(const_cast<int *>(j_AEntity_dof) + start,
                                 loc_size);

            // FIXME (trb 12/28/15): This will segfault in the case
            // that LocalTargets_[idx] have not been set yet.
            PARELAG_ASSERT(LocalTargets_[idx].size() == 0 ||
                           (LocalTargets_[idx][iAE]->Size() == loc_size));
            const int nlocaltargets = LocalTargets_[idx].size()?LocalTargets_[idx][iAE]->NumberOfVectors():0;

            loc_pv.SetSize(loc_size);
            loc_targets.SetSizeAndNumberOfVectors(loc_size, nTargets + nlocaltargets + 1);
            Targets_[jform]->GetSubMultiVector(dof_in_AE, loc_targets);
            if (nlocaltargets > 0)
            {
                loc_targets.SetSizeAndNumberOfVectors(loc_size, nTargets + nlocaltargets);
                MultiVector loc_tars(loc_targets.GetData() + nTargets*loc_size, nlocaltargets, loc_size);
                LocalTargets_[idx][iAE]->Copy(loc_tars);
            }

            pv.GetSubVector(dof_in_AE, loc_pv);

            auto Mloc = ExtractSubMatrix(*M_d, start, end, start, end);
            WeightedInnerProduct inner_product(*Mloc);
            Deflate(loc_targets, loc_pv, inner_product);

            if (loc_targets.NumberOfVectors() > 0)
            {
                if (IsDiagonal(*Mloc) )
                {
                    diagM_view.SetDataAndSize(diagM.GetData()+start, loc_size);
                    scaling_view.SetDataAndSize(scaling.GetData()+start, loc_size);
                    svd.ComputeON(scaling_view, loc_targets, s);
                }
                else
                {
                    DenseMatrix MM(Mloc->Size());
                    Full(*Mloc,MM);
                    svd.ComputeON(MM, loc_targets, s);
                }
            }
            else
            {
                s.SetSize(0);
            }

            s_max_tol = inner_product(loc_pv, loc_pv) * SVD_Tolerance_;
            int i(0);
            for(; i < s.Size(); ++i)
            {
                if (s(i) < s_max_tol)
                    break;
            }
            AE_ndofs[iAE] = i+1;

            double pv_dot_pv = inner_product(loc_pv, loc_pv);
            loc_targets *= sqrt(pv_dot_pv);

            auto p_loc = make_unique<DenseMatrix>(loc_size, AE_ndofs[iAE]);
            concatenate(loc_pv, loc_targets, *p_loc);

            auto cElemMass = make_unique<DenseMatrix>(AE_ndofs[iAE],
                                                      AE_ndofs[iAE]);
            inner_product(*p_loc, *p_loc, *cElemMass);
            cElemMass->Symmetrize();
#ifdef ELAG_DEBUG
            {
                DenseMatrix tmp(*cElemMass);
                int n = tmp.Height();
                for (int i = 0; i < n; ++i)
                    tmp(i,i) -= pv_dot_pv;

                double err = tmp.MaxMaxNorm();
                if (err > DEFAULT_EQUALITY_TOL)
                {
                    std::cout << "Form: " << jform << " iAE: " << iAE
                              << std::endl;
                    std::cout << "IsShared: " << AE_TrueAE.IsShared(iAE)
                              << std::endl;

                    std::cout<<"AE_ndofs: " << AE_ndofs[iAE] << std::endl;
                    s.Print(std::cout << "SingularValues =" << std::endl,
                            s.Size());
                    p_loc->PrintMatlab(std::cout<<"P_loc ="<<std::endl);
                    cElemMass->PrintMatlab(std::cout<<"cElemMass ="<<std::endl);
                    Mloc->PrintMatlab(std::cout<<"M_iAE ="<<std::endl);
                }

                PARELAG_ASSERT(err < DEFAULT_EQUALITY_TOL);
            }
#endif
            MultiVector allDataAsMV(p_loc->Data(), p_loc->Width(), loc_size);
            Pi_[jform]->CreateDofFunctional(codim, iAE, allDataAsMV, *Mloc);

            my_p.SetElementalMatrix(iAE, std::move(p_loc));
            mass.SetElementalMatrix(iAE, std::move(cElemMass) );
        }
    }

    {
        Array<int> tmp(AE_ndofs, nAE);
        AE_TrueAE.Synchronize(tmp);
    }

    DofHandlerALG * jdof = dynamic_cast<DofHandlerALG*>(coarser_sequence->Dof_[jform].get());
    PARELAG_TEST_FOR_EXCEPTION(
        !jdof,
        std::runtime_error,
        "DeRhamSequence::ComputeCoarseTraces(): "
        "Cast to DofHandlerALG failed!");
    jdof->AllocateDofTypeArray(EstimateUpperBoundNCoarseDof(jform));
    int cdof_counter = 0;

    //======= SECOND LOOP: Build the (coarse) entity_dof table ================//
    for (int iAE(0); iAE < nAE; ++iAE)
    {
        jdof->SetDofType(cdof_counter++, DofHandlerALG::RangeTSpace);
        jdof->SetNumberOfInteriorDofsRangeTSpace(codim, iAE, 1);
        for (int i(0); i < AE_ndofs[iAE]-1; ++i)
            jdof->SetDofType(cdof_counter++, DofHandlerALG::NullSpace);
        jdof->SetNumberOfInteriorDofsNullSpace(codim, iAE, AE_ndofs[iAE]-1);
    }
    jdof->BuildEntityDofTable(codim);

    int nCdof = jdof->GetNrDofs(codim);
    PARELAG_TEST_FOR_EXCEPTION(
        nCdof != cdof_counter,
        std::runtime_error,
        "DeRhamSequence::ComputeCoarseTraces(): "
        "nCdof does not match!");

    const SparseMatrix & AEntity_CDof = jdof->GetEntityDofTable(codim);
    const int * i_AEntity_CDof = AEntity_CDof.GetI();
    const int * const j_AEntity_CDof = AEntity_CDof.GetJ();

    //========== THIRD LOOP: Assemble P_partial ========================//
    Array<int> rows;
    Array<int> cols;
    auto P_partial = make_unique<SparseMatrix>(nDofs, AEntity_CDof.Width());

    for (int iAE(0); iAE < nAE; ++iAE)
    {
        if (AE_TrueAE.IsShared(iAE) != -1)
        {
            // fine dofs in AE
            const int start    = i_AEntity_dof[iAE];
            const int end      = i_AEntity_dof[iAE+1];
            const int loc_size = end - start;
            // coarse dofs in AE
            const int col_start    = i_AEntity_CDof[iAE];
            const int col_end      = i_AEntity_CDof[iAE+1];

            rows.MakeRef(const_cast<int *>(j_AEntity_dof) + start, loc_size);
            cols.MakeRef(const_cast<int *>(j_AEntity_CDof) + col_start,
                         col_end - col_start);

#ifdef ELAG_DEBUG
            if (rows.Size() > 0)
            {
                PARELAG_TEST_FOR_EXCEPTION(
                    rows.Max() >= P_[jform]->Size(),
                    std::runtime_error,
                    "DeRhamSequence::ComputeCoarseTraces(): "
                    "rows.Max() >= P_[jform]->Size()");
            }

            if (cols.Size() > 0)
            {
                PARELAG_TEST_FOR_EXCEPTION(
                    cols.Max() >= P_[jform]->Width(),
                    std::runtime_error,
                    "DeRhamSequence::ComputeCoarseTraces(): "
                    "cols.Max() >= P_[jform]->Width()");
            }

            PARELAG_ASSERT(my_p.GetElementalMatrix(iAE).Height() == rows.Size());
            PARELAG_ASSERT(my_p.GetElementalMatrix(iAE).Width() == cols.Size());
#endif
            P_partial->SetSubMatrix(rows, cols, my_p.GetElementalMatrix(iAE));
        }
    }
    P_partial->Finalize();

    SharingMap cdof_TrueCDof(Comm_);
    cdof_TrueCDof.SetUp(AE_TrueAE, const_cast<SparseMatrix&>(AEntity_CDof) );
    auto P_ok = AssembleNonLocal(dof_TrueDof,*P_partial,cdof_TrueCDof);

    AddOpenFormat(*P_[jform], *P_ok);

    // Done with P_ok
    P_ok.reset();
    P_partial.reset();

    for (int iAE(0); iAE < nAE; ++iAE)
    {
        if(AE_TrueAE.IsShared(iAE) == -1 )	//If the entity is shared and it does not belong to this processor
        {
            const int start    = i_AEntity_dof[iAE];
            const int end      = i_AEntity_dof[iAE+1];
            const int loc_size = end-start;
            const int c_start    = i_AEntity_CDof[iAE];
            const int c_end      = i_AEntity_CDof[iAE+1];
            const int c_size = c_end-c_start;
            Array<int> dof_in_AE(const_cast<int *>(j_AEntity_dof) + start, loc_size);
            Array<int> cdof_in_AE(const_cast<int *>(j_AEntity_CDof) + c_start, c_size );

            auto Mloc = ExtractSubMatrix(*M_d, start, end, start, end);

            WeightedInnerProduct inner_product(*Mloc);

            elag_assert(AE_ndofs[iAE] == c_size);

            DenseMatrix p_loc(loc_size, AE_ndofs[iAE]);
            P_[jform]->GetSubMatrix(dof_in_AE, cdof_in_AE, p_loc);

            auto cElemMass = make_unique<DenseMatrix>(AE_ndofs[iAE],
                                                      AE_ndofs[iAE]);
            inner_product(p_loc, p_loc, *cElemMass);
            cElemMass->Symmetrize();

#ifdef ELAG_DEBUG
            {
                DenseMatrix tmp(*cElemMass);
                const int n = tmp.Height();
                const double pv_dot_pv = tmp(0,0);
                for(int i = 0; i < n; ++i)
                    tmp(i,i) -= pv_dot_pv;

                const double err = tmp.MaxMaxNorm();
                if(err > DEFAULT_EQUALITY_TOL)
                {
                    std::cout << "Form: " << jform << " iAE: " << iAE
                              << std::endl;

                    std::cout<<"AE_ndofs: " << AE_ndofs[iAE] << std::endl;
                    s.Print(std::cout<<"SingularValues ="<<std::endl,s.Size());
                    p_loc.PrintMatlab(std::cout<<"P_loc ="<<std::endl);
                    cElemMass->PrintMatlab(std::cout<<"cElemMass ="<<std::endl);
                    Mloc->PrintMatlab(std::cout<<"M_iAE ="<<std::endl);
                }
                PARELAG_ASSERT(err < DEFAULT_EQUALITY_TOL);
            }
#endif
            mass.SetElementalMatrix(iAE, std::move(cElemMass));

            MultiVector allDataAsMV(p_loc.Data(), p_loc.Width(), loc_size);
            Pi_[jform]->CreateDofFunctional(codim, iAE, allDataAsMV, *Mloc);
        }
    }

    DeRhamSequence_os << "*** Number of PV dofs        = "
                      << nAE << std::endl
                      << "*** Number of NullSpace Dofs = "
                      << i_AEntity_CDof[nAE] - nAE << std::endl;

    coarser_sequence->M_[idx] = mass.GetAsSparseMatrix();
}

void DeRhamSequence::PartitionLocalTargets(
    int nTargets, int nlocaltargets,
    MultiVector& localtargets, int bdr_size, int internal_size,
    MultiVector& localtargets_interior, MultiVector& localtargets_boundary)
{
    int nrhs_Null = nTargets + nlocaltargets;
    PARELAG_ASSERT(localtargets.Size() == bdr_size + internal_size);

    localtargets_interior.SetSizeAndNumberOfVectors(internal_size, nrhs_Null);
    localtargets_boundary.SetSizeAndNumberOfVectors(bdr_size, nrhs_Null);
    MultiVector loc_inttars(
        localtargets_interior.GetData() + nTargets*internal_size,
        nlocaltargets, internal_size);
    MultiVector loc_bdrtars(
        localtargets_boundary.GetData() + nTargets*bdr_size,
        nlocaltargets, bdr_size);
    Array<int> intdofs(internal_size);
    Array<int> bdrdofs(bdr_size);
    for (int i=0; i < intdofs.Size(); ++i)
        intdofs[i] = i;
    for (int i=0; i < bdrdofs.Size(); ++i)
        bdrdofs[i] = i + intdofs.Size();
    localtargets.GetSubMultiVector(intdofs, loc_inttars);
    localtargets.GetSubMultiVector(bdrdofs, loc_bdrtars);
}

unique_ptr<MultiVector> DeRhamSequence::BuildCochainProjector(
    int nrhs_RangeT, int nrhs_Null,
    const Array<int>& coarseUDof_InternalRangeT,
    const Array<int>& coarseUDof_InternalNull,
    SparseMatrix& Projector,
    const Array<int>& fineUDof_Internal,
    int usize) const
{
    const int n_internal = nrhs_RangeT + nrhs_Null;
    Array<int> allInternalCDofs(n_internal);
    int * tmp =  std::copy(
        coarseUDof_InternalRangeT.GetData(),
        coarseUDof_InternalRangeT.GetData() + nrhs_RangeT,
        allInternalCDofs.GetData());
    tmp = std::copy(coarseUDof_InternalNull.GetData(),
                    coarseUDof_InternalNull.GetData() + nrhs_Null, tmp);
    DenseMatrix localBasis(fineUDof_Internal.Size(), n_internal);
    Projector.GetSubMatrix(fineUDof_Internal, allInternalCDofs,
                           localBasis);
    unique_ptr<MultiVector> myVect = make_unique<MultiVector>(
        localBasis.Data(), n_internal, usize);
    localBasis.ClearExternalData();
    myVect->MakeDataOwner();
    return myVect;
}

unique_ptr<DenseMatrix> DeRhamSequence::CoarsenMassMatrixPart(
    int nlocalbasis, const Array<int>& coarseUDof_on_Bdr,
    const Array<int>& coarseUDof_InternalRangeT,
    const Array<int>& coarseUDof_InternalNull,
    SparseMatrix& Projector, const Array<int>& fineUDof,
    const SparseMatrix& M_aa) const
{
    Array<int> allCDofs(nlocalbasis);
    int * tmp = std::copy(
        coarseUDof_on_Bdr.GetData(),
        coarseUDof_on_Bdr.GetData() + coarseUDof_on_Bdr.Size(),
        allCDofs.GetData());
    tmp = std::copy(
        coarseUDof_InternalRangeT.GetData(),
        coarseUDof_InternalRangeT.GetData() + coarseUDof_InternalRangeT.Size(),
        tmp);
    tmp = std::copy(
        coarseUDof_InternalNull.GetData(),
        coarseUDof_InternalNull.GetData() + coarseUDof_InternalNull.Size(),
        tmp);
    DenseMatrix localBasis(fineUDof.Size(), nlocalbasis);
    Projector.GetSubMatrix(fineUDof, allCDofs, localBasis);
    auto cElemMass = make_unique<DenseMatrix>(nlocalbasis, nlocalbasis);
    WeightedInnerProduct dot2(M_aa);
    dot2(localBasis, localBasis, *cElemMass);
    cElemMass->Symmetrize();
    return cElemMass;
}

/**
   Here we do block solvers for the following systems. For the PV
   extension and the NullDof extension from traces, we solve:

   [ A_ii   B_ai^T    0   ] [ u ]     [A_ib*R_d ]
   [ B_ai     0       T^T ] [ p ] = - [B_ab*R_d ],
   [ 0        T       0   ] [ l ]     [ 0       ]

   for the RangeTDof bubble functions we solve:

   [ A_ii   B_ai^T    0   ] [ u ]     [ 0 ]
   [ B_ai     0       T^T ] [ p ] = - [ W*cP ]
   [ 0        T       0   ] [ l ]     [ 0 ]

   where:

   u is coarse degrees of freedom, ie, columns of the Projector operator
        between the coarse and fine space.
   p is the sparse matrix of lagrangian multipliers (that can be ignored)
   l is the sparse matrix representing the coarse derivative operator

   A_ii is the sparse matrix discretizing (u,v) restricted to the interal
        dofs of each agglomerate
   A_ib is the sparse rectangular matrix discretizing (u,v) whose rows are
        restricted to the interal dofs of each agglomerate, and the columns
        are restricted to the boundary dofs of each agglomerate.
        (implemented as Mloc_ib (?))
   B_ai is the sparse rectangular matrix discretizing <\der u, q> whose
        rows and columns are restricted to the internal dofs of each agglomerate.
   B_ab is the sparse rectangular matrix discretizing <\der u, q> whose
        columns are restricted to the bdr dofs of each agglomerate.
   T    is the sparse matrix discretizing <Q,q> where Q belongs to the coarse
        PV (jform+1) and q belongs to the fine (iform+1).
   R_d  is the thing we are extending, which is input to this method and
        comes in P_[]
    cP  is coarse "pressure" dofs which are targets previously determined
        on interior

   We do not always know how the notation above corresponds to entities in
   the actual code (TODO!!)

   One of many things I do not like about this method is how it takes P_[] as
   both input and output, would maybe like to separate the traces/targets from
   previous parts of the coarsening process from the extensions/bubbles that
   get produced here instead of doing things in place.
*/
void DeRhamSequence::hFacetExtension(int jform)
{
    auto coarser_sequence = CoarserSequence_.lock();
    PARELAG_ASSERT(coarser_sequence);

    DeRhamSequence_os << "Enter hFacetExtension of jform " << jform << std::endl;

    AgglomeratedTopology::Entity codim_bdr
        = static_cast<AgglomeratedTopology::Entity>(nForms_-jform-1);
    AgglomeratedTopology::Entity codim_dom
        = static_cast<AgglomeratedTopology::Entity>(nForms_-jform-2);

    DofHandlerALG * uCDofHandler
        = dynamic_cast<DofHandlerALG *>(coarser_sequence->Dof_[jform].get());
    DofHandlerALG * pCDofHandler
        = dynamic_cast<DofHandlerALG *>(coarser_sequence->Dof_[jform+1].get());

    PARELAG_TEST_FOR_EXCEPTION(
        uCDofHandler == nullptr || pCDofHandler == nullptr,
        std::runtime_error,
        "The coarse dof handlers are not of DofHandlerALG type.");

    const int nDim = Topo_->Dimensions();
    const int idx = (nDim-jform)*(nForms_-jform)/2 + codim_dom;

    // Agglomerate-based Mass Matrix (primal variable) [eg. velocity]
    auto M_d = AssembleAgglomerateMatrix(codim_dom,
                                         *(M_[idx]),
                                         DofAgg_[jform].get(),
                                         DofAgg_[jform].get());

    // Agglomerate-based Derivative operator [eg. divergence]
    auto D_d = DistributeAgglomerateMatrix(codim_dom,
                                           *(D_[jform]),
                                           DofAgg_[jform+1].get(),
                                           DofAgg_[jform].get());

    // Agglomerate-based Mass Matrix (derivative) [eg. pressure]
    const int idx_w = (nDim-jform-1)*(nForms_-jform-1)/2 + codim_dom;
    auto W_d = AssembleAgglomerateMatrix(codim_dom,
                                         *(M_[idx_w]),
                                         DofAgg_[jform+1].get(),
                                         DofAgg_[jform+1].get());
    // "Galerkin" divergence operator
    unique_ptr<SparseMatrix> B_d{Mult(*W_d, *D_d)};

    // vectors containing the fluxes on the boundary, stored in matrix
    const int use_actual_width = 1;
    unique_ptr<SparseMatrix> R_t{
        TransposeAbstractSparseMatrix(*P_[jform], use_actual_width) };
    unique_ptr<SparseMatrix> Dt{Transpose(*D_[jform])};
    // next two assemble agglomerate matrices into one matrix
    auto M_gd = Assemble(codim_dom,*M_d,DofAgg_[jform].get(),nullptr);
    auto W_gd = Assemble(codim_dom,*W_d,DofAgg_[jform+1].get(),nullptr);
    R_t->SetWidth(M_gd->Size());
    unique_ptr<SparseMatrix> minusRtM_d{Mult(*R_t, *M_gd)};
    *minusRtM_d *= -1.0;
    R_t->SetWidth(Dt->Size());
    unique_ptr<SparseMatrix> minusRtDt{Mult(*R_t, *Dt)};
    *minusRtDt *= -1.0;
    minusRtDt->SetWidth(W_gd->Size() );
    unique_ptr<SparseMatrix> minusRtBt_d{Mult(*minusRtDt, *W_gd)};

    // free some memory, may be unnecessary
    minusRtDt.reset();
    Dt.reset();
    W_gd.reset();
    M_gd.reset();

    // hAE_AFCD connects "agglomerated entities" in the rows to the dofs
    // contained in their boundary (I think...)
    unique_ptr<SparseMatrix> hAE_AFCD{
        Mult(coarser_sequence->Topo_->GetB(codim_dom),
             coarser_sequence->Dof_[jform]->GetEntityDofTable(codim_bdr)) };
    const int nhAE = hAE_AFCD->Size();
    const int * i_hAE_AFCD = hAE_AFCD->GetI();
    const int * j_hAE_AFCD = hAE_AFCD->GetJ();

    // SparseMatrix to find the global PV dof for jform+1.
    auto AE_PVdof = pCDofHandler->GetEntityRangeTSpaceDofTable(codim_dom);
    const int * const i_AE_PVdof = AE_PVdof->GetI();
    const int * const j_AE_PVdof = AE_PVdof->GetJ();

#ifdef ELAG_DEBUG
    {
        const int * it = i_AE_PVdof;
        for (int i=0; i<nhAE; ++i, ++it)
            PARELAG_ASSERT(*it == i);
    }
#endif

    // SparseMatrix to find the global NullSpace dof for jform+1
    // (this new dofs of jform will go in RangeTSpace)
    auto AE_NullSpaceDof = pCDofHandler->GetEntityNullSpaceDofTable(codim_dom);
    // Tom thinks next should be some kind of smart pointers(?)
    const int * const i_AE_NullSpaceDof = AE_NullSpaceDof->GetI();
    const int * const j_AE_NullSpaceDof = AE_NullSpaceDof->GetJ();

    int nrhs_ext, nrhs_RangeT, nrhs_Null;
    MultiVector rhs_view_u, rhs_view_p, rhs_view_l;

    Array<int> coarseUDof_on_Bdr, coarseUDof_InternalRangeT,
        coarseUDof_InternalNull;
    Array<int> coarsePDof;
    Array<int> fineUDof_Internal, fineUDof_Bdr, fineUDof, finePDof;

    int max_nlocaltargets = 0;
    const auto loctargsize = LocalTargets_[idx].size();
    if (loctargsize > 0)
    {
        elag_assert(loctargsize == decltype(loctargsize)(nhAE));
        for (auto& mymv : LocalTargets_[idx])
        {
            elag_assert(mymv);
            const int candidate = mymv->NumberOfVectors();
            if (max_nlocaltargets < candidate)
                max_nlocaltargets = candidate;
        }
    }

    // Upper bound on the number of coarse dofs, assuming that targets
    // are linearly independent from all other types of dofs.
    int estimateNumberCoarseDofs(0);
    // Traces dofs + Derivative dofs + number of AE
    estimateNumberCoarseDofs = R_t->Size()+P_[jform+1]->Width();
    // Maximum allowed number of Bubble AE dofs - number of AE
    estimateNumberCoarseDofs += nhAE * (Targets_[jform]->NumberOfVectors()
                                        + max_nlocaltargets - 1);

    coarser_sequence->D_[jform] = make_unique<SparseMatrix>(
        P_[jform+1]->Width(),
        estimateNumberCoarseDofs);
    ElementalMatricesContainer mass(nhAE);

    DenseMatrix subm;
    Vector subv;
    int nNullSpaceDofs(0), nRangeTDofs(0);
    int coarseDofCounter(R_t->Size());
    // The dofs we add are bubbles on the domain, so their
    // global numbering is higher than the dof of the traces

    SVD_Calculator svd;
    svd.setFlagON();
    const int nTargets = Targets_[jform]->NumberOfVectors();
    const int max_nfDof_per_AE
        = DofAgg_[jform]->GetAEntityDof(codim_dom).MaxRowSize();
    if (nTargets + max_nlocaltargets)
        svd.AllocateOptimalSize(max_nfDof_per_AE, nTargets + max_nlocaltargets);
    Vector sv(nTargets + max_nlocaltargets);

    for (int iAE(0); iAE < nhAE; ++iAE)
    {
        FacetSaddlePoint ssps(codim_dom, *DofAgg_[jform],
                              *DofAgg_[jform+1], P_[jform+1].get(),
                              *AE_PVdof, *M_d, *B_d, *W_d, iAE);

        int uBdrStart, uBdrEnd;
        DofAgg_[jform]->GetViewAgglomerateInternalDofGlobalNumering(
            codim_dom, iAE, fineUDof_Internal);
        DofAgg_[jform+1]->GetViewAgglomerateInternalDofGlobalNumering(
            codim_dom, iAE, finePDof);
        DofAgg_[jform]->GetViewAgglomerateBdrDofGlobalNumering(
            codim_dom, iAE, fineUDof_Bdr);
        DofAgg_[jform]->GetAgglomerateBdrDofRange(
            codim_dom, iAE, uBdrStart, uBdrEnd);

        // (3) Solve the harmonic extension of PV vectors, place in P_
        nrhs_ext = i_hAE_AFCD[iAE+1] - i_hAE_AFCD[iAE];
        coarseUDof_on_Bdr.MakeRef(
            const_cast<int *>(j_hAE_AFCD+i_hAE_AFCD[iAE]),
            i_hAE_AFCD[iAE+1]-i_hAE_AFCD[iAE]);

        ssps.SetUpRhs(nrhs_ext, rhs_view_u, rhs_view_p, rhs_view_l);
        GetRows(*minusRtM_d, coarseUDof_on_Bdr, ssps.GetUInternalStart(),
                ssps.GetUInternalEnd(), rhs_view_u);
        GetRows(*minusRtBt_d, coarseUDof_on_Bdr, ssps.GetPInternalStart(),
                ssps.GetPInternalEnd(), rhs_view_p);

        ssps.Solve(rhs_view_u, rhs_view_p, rhs_view_l);

        subm.SetSize(ssps.GetLocalOffsets(1), nrhs_ext);
        rhs_view_u.CopyToDenseMatrix(0, rhs_view_u.NumberOfVectors(),
                                     0, rhs_view_u.Size(), subm);
        P_[jform]->SetSubMatrix(fineUDof_Internal, coarseUDof_on_Bdr, subm);

        // coarsen the ExteriorDerivative
        subv.SetSize(nrhs_ext);
        for (int isol(0); isol <nrhs_ext; ++isol)
        {
            if (fabs(rhs_view_l.GetDataFromVector(isol)[0]) > SmallestEntry_)
                subv(isol) = -rhs_view_l.GetDataFromVector(isol)[0];
            else
                subv(isol) = 0.0;
        }
        coarser_sequence->D_[jform]->SetRow(j_AE_PVdof[ i_AE_PVdof[iAE] ],
                                            coarseUDof_on_Bdr, subv);

        // (4) Solve for RangeT Dofs (bubble functions, targets of previous jform)
        nrhs_RangeT = i_AE_NullSpaceDof[iAE+1] - i_AE_NullSpaceDof[iAE];
        coarseUDof_InternalRangeT.SetSize(nrhs_RangeT);
        uCDofHandler->SetNumberOfInteriorDofsRangeTSpace(codim_dom, iAE, nrhs_RangeT);
        nRangeTDofs += nrhs_RangeT;
        if (nrhs_RangeT)
        {
            for(int i(0); i<nrhs_RangeT; ++i)
                coarseUDof_InternalRangeT[i] = coarseDofCounter++;
            coarsePDof.MakeRef(
                const_cast<int *>(j_AE_NullSpaceDof+i_AE_NullSpaceDof[iAE]),
                nrhs_RangeT);

            subm.SetSize(ssps.GetLocalOffsets(2)-ssps.GetLocalOffsets(1), nrhs_RangeT);
            P_[jform+1]->GetSubMatrix(finePDof,coarsePDof,subm);

            // Get local rhs
            ssps.SetUpRhs(nrhs_RangeT, rhs_view_u, rhs_view_p, rhs_view_l);
            MultiVector subm_as_mv(subm.Data(), nrhs_RangeT,
                                   ssps.GetLocalOffsets(2)-ssps.GetLocalOffsets(1));
            MatrixTimesMultiVector(ssps.GetWloc(), subm_as_mv, rhs_view_p);
            ssps.Solve(rhs_view_u, rhs_view_p, rhs_view_l);

            subm.SetSize(ssps.GetLocalOffsets(1), nrhs_RangeT);
            rhs_view_u.CopyToDenseMatrix(0, rhs_view_u.NumberOfVectors(),
                                         0, rhs_view_u.Size(), subm);
            P_[jform]->SetSubMatrix(fineUDof_Internal,coarseUDof_InternalRangeT,
                                    subm);
            for (int i(0); i<nrhs_RangeT; ++i)
            {
                coarser_sequence->D_[jform]->Set(
                    coarsePDof[i], coarseUDof_InternalRangeT[i], 1.);
                uCDofHandler->SetDofType(coarseUDof_InternalRangeT[i],
                                         DofHandlerALG::RangeTSpace);
            }
        }

        // (5) Solve for Null Dofs (extension of targets on the traces into interior)
        const int nlocaltargets
            = loctargsize ? LocalTargets_[idx][iAE]->NumberOfVectors() : 0;
        if (ssps.GetLocalOffsets(1) > nrhs_RangeT && (nTargets + nlocaltargets))
        {
            nrhs_Null = nTargets + nlocaltargets;
            MultiVector localTargets_Interior(nrhs_Null, fineUDof_Internal.Size());
            MultiVector localTargets_Bdr(nrhs_Null, fineUDof_Bdr.Size());
            Targets_[jform]->GetSubMultiVector(fineUDof_Internal,
                                               localTargets_Interior);
            Targets_[jform]->GetSubMultiVector(fineUDof_Bdr, localTargets_Bdr);

            if (nlocaltargets > 0)
            {
                PartitionLocalTargets(
                    nTargets, nlocaltargets, *LocalTargets_[idx][iAE],
                    fineUDof_Bdr.Size(), fineUDof_Internal.Size(),
                    localTargets_Interior, localTargets_Bdr);
            }

            // Get local rhs
            auto Mloc_ib = ExtractSubMatrix(*M_d, ssps.GetUInternalStart(),
                                            ssps.GetUInternalEnd(),
                                            uBdrStart, uBdrEnd);
            ssps.SetUpRhs(nrhs_Null, rhs_view_u, rhs_view_p, rhs_view_l);
            MatrixTimesMultiVector(-1., *Mloc_ib,
                                   localTargets_Bdr, rhs_view_u);
            MatrixTimesMultiVector(ssps.GetBloc(), localTargets_Interior,
                                   rhs_view_p);
            ssps.Solve(rhs_view_u, rhs_view_p, rhs_view_l);
            add(1., localTargets_Interior, -1., rhs_view_u, localTargets_Interior);

#ifdef ELAG_DEBUG
            {
                // non-divergence-free bubble
                MultiVector dummy(
                    nrhs_Null,ssps.GetLocalOffsets(2) - ssps.GetLocalOffsets(1));
                MatrixTimesMultiVector(ssps.GetBloc(), localTargets_Interior,
                                       dummy);
                if (dummy.Normlinf() > LOOSE_EQUALITY_TOL)
                {
                    std::cerr << "* Warning hFacet: Form " << jform
                              << " Agglomerated Element " << iAE
                              << " of codim "<< codim_dom
                              << ": ||D u _loc||_inf = " << dummy.Normlinf()
                              << " (expect 0)" << std::endl;
                }
            }
#endif
            svd.ComputeON(localTargets_Interior, sv);
            const double s_max_tol = SVD_Tolerance_;

            nrhs_Null = 0;
            coarseUDof_InternalNull.SetSize(nrhs_Null );
            for (; nrhs_Null < sv.Size(); ++nrhs_Null)
            {
                if (sv(nrhs_Null) < s_max_tol)
                    break;

                uCDofHandler->SetDofType(coarseDofCounter, DofHandlerALG::NullSpace);
                coarseUDof_InternalNull.Append(coarseDofCounter++);
            }
            nNullSpaceDofs += nrhs_Null;

            uCDofHandler->SetNumberOfInteriorDofsNullSpace(codim_dom, iAE,
                                                           nrhs_Null);
            subm.SetSize(ssps.GetLocalOffsets(1) - ssps.GetLocalOffsets(0),
                         nrhs_Null);
            localTargets_Interior.CopyToDenseMatrix(
                0, nrhs_Null, ssps.GetLocalOffsets(0), ssps.GetLocalOffsets(1),
                subm);
            P_[jform]->SetSubMatrix(fineUDof_Internal, coarseUDof_InternalNull,
                                    subm);
        }
        else // no null dofs
        {
            nrhs_Null = 0;
            uCDofHandler->SetNumberOfInteriorDofsNullSpace(codim_dom, iAE,
                                                           nrhs_Null);
            coarseUDof_InternalNull.SetSize(nrhs_Null );
        }

        // (5) cochain Projector Pi from fine to coarse
        {
            unique_ptr<MultiVector> myVect = BuildCochainProjector(
                nrhs_RangeT, nrhs_Null, coarseUDof_InternalRangeT,
                coarseUDof_InternalNull, *P_[jform], fineUDof_Internal,
                ssps.GetLocalOffsets(1) - ssps.GetLocalOffsets(0));
            Pi_[jform]->CreateDofFunctional(codim_dom, iAE, *myVect,
                                            ssps.GetMloc());
        }

        // (6) Coarsen part of the mass matrix
        {
            int nlocalbasis = nrhs_ext + nrhs_RangeT + nrhs_Null;
            int uAllStart, uAllEnd;
            DofAgg_[jform]->GetAgglomerateDofRange(
                codim_dom, iAE, uAllStart, uAllEnd);
            DofAgg_[jform]->GetViewAgglomerateDofGlobalNumering(
                codim_dom, iAE, fineUDof);
            auto M_aa = ExtractSubMatrix(*M_d, uAllStart, uAllEnd,
                                         uAllStart, uAllEnd);
            auto cElemMass = CoarsenMassMatrixPart(
                nlocalbasis, coarseUDof_on_Bdr,
                coarseUDof_InternalRangeT, coarseUDof_InternalNull,
                *P_[jform], fineUDof, *M_aa);
            mass.SetElementalMatrix(iAE, std::move(cElemMass));
        }
    }

    DeRhamSequence_os << "*** Number of dofs that have been extended "
                      << R_t->Size() << std::endl
                      << "*** Number of RangeTSpace Dofs             "
                      << nRangeTDofs << std::endl
                      << "*** Number of NullSpace Dofs               "
                      << nNullSpaceDofs << std::endl;

    // to be consistent with previous non-C++11 deletion
    AE_PVdof.reset();
    AE_NullSpaceDof.reset();

    uCDofHandler->BuildEntityDofTable(codim_dom);
    coarser_sequence->M_[idx] = mass.GetAsSparseMatrix();

    PARELAG_TEST_FOR_EXCEPTION(
        (coarseDofCounter != R_t->Size() + nRangeTDofs + nNullSpaceDofs ) ||
        (uCDofHandler->GetNumberInteriorDofs(codim_dom ) != nRangeTDofs + nNullSpaceDofs ),
        std::runtime_error,
        "DeRhamSequence::hFacetExtension(): "
        "Number of Interior Dof according to DofHandler: "
        << uCDofHandler->GetNumberInteriorDofs(codim_dom)
        << std::endl << "Actual number of Interior Dof: "
        << coarseDofCounter - R_t->Size() << std::endl);
}

unique_ptr<SparseMatrix> DeRhamSequence::GetMinusC(
    int jform, AgglomeratedTopology::Entity codim_dom)
{
    const int nDim = Topo_->Dimensions();
    // in the canonical (documented) case, this is divergence on RT space
    auto D2_d = DistributeAgglomerateMatrix(codim_dom,
                                            *(D_[jform+1]),
                                            DofAgg_[jform+2].get(),
                                            DofAgg_[jform+1].get());
    const int idx_w2 = (nDim-jform-2)*(nForms_-jform-2)/2 + codim_dom;
    // this is a mass matrix on the RT space
    auto W2_d = AssembleAgglomerateMatrix(codim_dom,
                                          *(M_[idx_w2]),
                                          DofAgg_[jform+2].get(),
                                          DofAgg_[jform+2].get());

    // minusC block
    unique_ptr<SparseMatrix> D2T_d(Transpose(*D2_d));
    unique_ptr<SparseMatrix> tmp(Mult(*D2T_d,*W2_d));
    // we have something like (div^T) (div), for the div-div operator in Lashuk
    // and Vassilevski (6.43)
    unique_ptr<SparseMatrix> minusC_d(Mult(*tmp,*D2_d));
    (*minusC_d) *= -1.0;
    return minusC_d;
}

/**
   This code is called several times, this documentation is for one (hopefully
   representative) case, namely the extension of Nedelec/H(curl) functions from
   a face to a volume, which is (6.43) in Lashuk/Vassilevski.

   The local linear system takes the form

     [ M    B^T ][ q_T ]    [ -M_ib q ]
     [ B    -C  ][ r_T ] =  [ -B_ib q ]

   where M is a local mass matrix on the Nedelec space, B is a local curl
   operator, and C is a discretization of the (div, div) bilinear form on the
   Raviart-Thomas space. These are all defined on the interior of the
   agglomerated volume to which we are extending.

   M_ib is a rectangular mass-like matrix, while B_ib is a rectangular curl-like
   matrix. q is the thing we are extending from face to volume, and comes as
   input to this routine in the P_ matrix. The extension is placed in q_T which
   then gets put in P_, so P_ is both input and output here.
*/
void DeRhamSequence::hRidgePeakExtension(
    int jform, const AgglomeratedTopology::Entity codim_dom)
{
    auto coarser_sequence = CoarserSequence_.lock();
    PARELAG_ASSERT(coarser_sequence);

    DeRhamSequence_os << "Enter hRidgePeakExtension of " << jform << std::endl;
    const bool ridge_stuff = (codim_dom == nForms_ - jform - 3);

    const int nAE = coarser_sequence->Topo_->GetNumberLocalEntities(codim_dom);

    DofHandlerALG * uCDofHandler
        = dynamic_cast<DofHandlerALG *>(coarser_sequence->Dof_[jform].get());
    DofHandlerALG * pCDofHandler
        = dynamic_cast<DofHandlerALG *>(coarser_sequence->Dof_[jform+1].get());
    PARELAG_TEST_FOR_EXCEPTION(
        uCDofHandler == nullptr || pCDofHandler == nullptr,
        std::logic_error,
        "DeRhamSequence::hRidgePeakExtension(): "
        "The coarse dof handlers are not of DofHandlerALG type.");

    // (1) Extend all the traces (RIDGE-based or FACET-based) by using the
    //     correct exterior derivative from the previous space (THIS Dof are
    //     already labeled).
    // (2) For each dof in the NullSpace of jform+1 that vanished on the boundary
    //     build a shape function in the RangeT Space of jform.
    // (3) If anything is left of the targets build bubble shape functions for
    //     Null Space of jform.
#ifdef ELAG_DEBUG
    {
        unique_ptr<SparseMatrix> DD(Mult(*(D_[jform+1]),*(D_[jform])));
        const double DDnorm = DD->MaxNorm();
        if (DDnorm > DEFAULT_EQUALITY_TOL)
            std::cerr << "DeRhamSequence not globally exact! "
                      << "DDnorm " << DDnorm << std::endl;
    }
#endif

    // Agglomerate-based Mass Matrix (primal variable)
    const int nDim = Topo_->Dimensions();
    const int idx = (nDim-jform)*(nForms_-jform)/2 + codim_dom;

    auto M_d = AssembleAgglomerateMatrix(codim_dom,
                                         *(M_[idx]),
                                         DofAgg_[jform].get(),
                                         DofAgg_[jform].get());

    // Agglomerate-based Derivative operator (primal variable --> derivative )
    auto D_d = DistributeAgglomerateMatrix(codim_dom,
                                           *(D_[jform]),
                                           DofAgg_[jform+1].get(),
                                           DofAgg_[jform].get());

    // Agglomerate-based Mass Matrix (derivative)
    const int idx_w = (nDim-jform-1)*(nForms_-jform-1)/2 + codim_dom;
    auto W_d = AssembleAgglomerateMatrix(codim_dom,
                                         *(M_[idx_w]),
                                         DofAgg_[jform+1].get(),
                                         DofAgg_[jform+1].get());

    // B matrix ("Galerkin" derivative)
    unique_ptr<SparseMatrix> B_d(Mult(*W_d,*D_d));
    // C matrix (some (div, div) like bilinear form
    unique_ptr<SparseMatrix> minusC_d = GetMinusC(jform, codim_dom);
    // Matrix containing the target derivative on the coarse space
    unique_ptr<SparseMatrix> PDc(
        MultAbstractSparseMatrix(*(P_[jform+1]),*(coarser_sequence->D_[jform])));

#ifdef ELAG_DEBUG
    CheckLocalExactnessCommuting(jform, codim_dom);
#endif

    // Matrix containing the lifting of the bc on the boundary
    // on input this makes sense on the faces that enclose the volume, on exit
    // P_ then has these traces extended to the interior
    unique_ptr<SparseMatrix> R_t{
        TransposeAbstractSparseMatrix(*P_[jform], 1)};
    auto M_gd = Assemble(codim_dom,*M_d,DofAgg_[jform].get(),nullptr);
    auto W_gd = Assemble(codim_dom,*W_d,DofAgg_[jform+1].get(),nullptr);
    R_t->SetWidth(M_gd->Size());
    unique_ptr<SparseMatrix> minusRtM_d{Mult(*R_t, *M_gd)};
    *minusRtM_d *= -1.0;
    unique_ptr<SparseMatrix> Dt{Transpose(*D_[jform])};
    R_t->SetWidth(Dt->Size());
    unique_ptr<SparseMatrix> minusRtDt{Mult(*R_t, *Dt)};
    *minusRtDt *= -1.0;
    if (ridge_stuff)
        minusRtDt->SetWidth(W_gd->Size());
    unique_ptr<SparseMatrix> minusRtBt_d{Mult(*minusRtDt, *W_gd)};

    minusRtDt.reset();
    Dt.reset();
    W_gd.reset();
    M_gd.reset();

    // SparseMatrix to find the global NullSpace dof for jform+1 (these
    // new dofs of jform will go in RangeTSpace, used in bubble solve)
    auto AE_NullSpaceDof = pCDofHandler->GetEntityNullSpaceDofTable(codim_dom);
    const int * const i_AE_NullSpaceDof = AE_NullSpaceDof->GetI();
    const int * const j_AE_NullSpaceDof = AE_NullSpaceDof->GetJ();

    int uBdrStart, uBdrEnd;
    int pIBStart, pIBEnd;
    int nrhs_ext, nrhs_RangeT, nrhs_Null;
    MultiVector rhs_view_u, rhs_view_p, rhs_view_l;
    Array<int> coarseUDof_on_Bdr, coarseUDof_InternalRangeT,
        coarseUDof_InternalNull, coarsePDof_Internal;
    Array<int> fineUDof_Internal, fineUDof_Bdr, finePDof_Internal;
    Array<int> fineUDof, finePDof;

    int max_nlocaltargets = 0;
    if (ridge_stuff)
    {
        const auto loctargsize = LocalTargets_[idx].size();
        if (loctargsize > 0)
        {
            PARELAG_ASSERT(loctargsize == decltype(loctargsize)(nAE));
            for (auto& mymv : LocalTargets_[idx])
            {
                PARELAG_ASSERT(mymv);
                const int candidate = mymv->NumberOfVectors();
                if (max_nlocaltargets < candidate)
                    max_nlocaltargets = candidate;
            }
        }
    }

    ElementalMatricesContainer mass(nAE);
    DenseMatrix subm;
    Vector subv;

    // The dofs we add are bubbles on the domain, so their global numbering is
    // higher then the dof of the traces
    int coarseDofCounter(R_t->Size());
    int nNullSpaceDofs(0), nRangeTDofs(0);

    SVD_Calculator svd;
    Vector sv;
    const int nTargets = Targets_[jform]->NumberOfVectors();
    if (ridge_stuff)
    {
        const int max_nfDof_per_AE
            = DofAgg_[jform]->GetAEntityDof(codim_dom).MaxRowSize();
        svd.setFlagON();
        if (nTargets + max_nlocaltargets)
            svd.AllocateOptimalSize(max_nfDof_per_AE,
                                    nTargets + max_nlocaltargets);
        sv.SetSize(nTargets + max_nlocaltargets);
    }

    for (int iAE(0); iAE < nAE; ++iAE)
    {
        RidgePeakSaddlePoint ssps(codim_dom,
                                  *DofAgg_[jform],
                                  *DofAgg_[jform+1],
                                  *M_d, *B_d, *minusC_d,
                                  iAE);

        DofAgg_[jform]->GetAgglomerateBdrDofRange(
            codim_dom, iAE, uBdrStart, uBdrEnd);
        DofAgg_[jform+1]->GetAgglomerateDofRange(
            codim_dom, iAE, pIBStart, pIBEnd);

        uCDofHandler->GetDofsOnBdr(codim_dom, iAE, coarseUDof_on_Bdr);
        nrhs_ext = coarseUDof_on_Bdr.Size();
        nrhs_RangeT = 0;
        nrhs_Null = 0;

        DofAgg_[jform]->GetViewAgglomerateInternalDofGlobalNumering(
            codim_dom, iAE, fineUDof_Internal);
        DofAgg_[jform+1]->GetViewAgglomerateInternalDofGlobalNumering(
            codim_dom, iAE, finePDof_Internal);

        DofAgg_[jform]->GetViewAgglomerateDofGlobalNumering(
            codim_dom, iAE, fineUDof);
        DofAgg_[jform+1]->GetViewAgglomerateDofGlobalNumering(
            codim_dom, iAE, finePDof);
        DofAgg_[jform]->GetViewAgglomerateBdrDofGlobalNumering(
            codim_dom, iAE,fineUDof_Bdr);

        // get local matrices for constructing right hand side
        auto Mloc_ib = ExtractSubMatrix(
            *M_d, ssps.GetUInternalStart(), ssps.GetUInternalEnd(),
            uBdrStart, uBdrEnd);
        auto Wloc_iA = ExtractSubMatrix(
            *W_d, ssps.GetPInternalStart(), ssps.GetPInternalEnd(),
            pIBStart, pIBEnd);
        auto Wloc = ExtractSubMatrix(
            *W_d, ssps.GetPInternalStart(), ssps.GetPInternalEnd(),
            ssps.GetPInternalStart(), ssps.GetPInternalEnd());

        ssps.SetUpRhs(nrhs_ext, rhs_view_u, rhs_view_p);
        GetRows(*minusRtM_d, coarseUDof_on_Bdr,
                ssps.GetUInternalStart(), ssps.GetUInternalEnd(), rhs_view_u);
        GetRows(*minusRtBt_d, coarseUDof_on_Bdr,
                ssps.GetPInternalStart(), ssps.GetPInternalEnd(), rhs_view_p);

        subm.SetSize(pIBEnd - pIBStart, nrhs_ext);
        PDc->GetSubMatrix(finePDof, coarseUDof_on_Bdr, subm);
        MultiVector localDerivative(subm.Data(), nrhs_ext, pIBEnd - pIBStart);
        MatrixTimesMultiVector(1., *Wloc_iA, localDerivative, rhs_view_p);

        // extend the PV vectors into the interior (see Lashuk-Vassilevski (6.43)
        // and (7.18))
        if (ssps.GetLocalOffsets(1) != 0)
        {
            ssps.Solve(rhs_view_u, rhs_view_p); // extension

#ifdef ELAG_DEBUG
            {
                // it is not clear what we are actually checking here,
                // (but these warnings do correspond to failures later in code)
                // check if extension satisfies constraint?
                MultiVector tmp(
                    nrhs_ext, ssps.GetLocalOffsets(2) - ssps.GetLocalOffsets(1));
                tmp = 0.;
                MatrixTimesMultiVector(1., ssps.GetmCloc(), rhs_view_p, tmp);
                if (tmp.Normlinf() > LOOSE_EQUALITY_TOL)
                {
                    if (ridge_stuff)
                        std::cerr << "* Warning hRidge: Form ";
                    else
                        std::cerr << "* Warning hPeak: Form ";
                    std::cerr << jform << " Agglomerated Element "
                              << iAE << " of codim " << codim_dom
                              << ": || C u ||_inf = " << tmp.Normlinf()
                              << " (expect 0), extension solve singular?"
                              << std::endl;
                    // // for next lines see build/testsuite/check.py
                    // std::stringstream tag;
                    // tag << "rp" << iAE;
                    // ssps.DumpDetails(tag.str());
                }
            }
#endif
        }
        subm.SetSize(ssps.GetLocalOffsets(1), nrhs_ext);
        rhs_view_u.CopyToDenseMatrix(
            0, nrhs_ext, ssps.GetLocalOffsets(0), ssps.GetLocalOffsets(1), subm);
        P_[jform]->SetSubMatrix(fineUDof_Internal, coarseUDof_on_Bdr, subm);

        // (4) Solve for RangeT Dofs (aka bubbles)
        nrhs_RangeT = i_AE_NullSpaceDof[iAE+1] - i_AE_NullSpaceDof[iAE];
        nRangeTDofs += nrhs_RangeT;
        coarseUDof_InternalRangeT.SetSize(nrhs_RangeT);
        uCDofHandler->SetNumberOfInteriorDofsRangeTSpace(
            codim_dom, iAE, nrhs_RangeT);
        if (nrhs_RangeT)
        {
            for (int i(0); i<nrhs_RangeT; ++i)
                coarseUDof_InternalRangeT[i] = coarseDofCounter++;

            coarsePDof_Internal.MakeRef(
                const_cast<int *>(j_AE_NullSpaceDof+i_AE_NullSpaceDof[iAE]),
                nrhs_RangeT);

            subm.SetSize(ssps.GetLocalOffsets(2)-ssps.GetLocalOffsets(1),
                         nrhs_RangeT);
            P_[jform+1]->GetSubMatrix(
                finePDof_Internal, coarsePDof_Internal, subm);

            // set up local rhs
            ssps.SetUpRhs(nrhs_RangeT, rhs_view_u, rhs_view_p);
            MultiVector subm_as_mv(subm.Data(), nrhs_RangeT,
                                   ssps.GetLocalOffsets(2)-ssps.GetLocalOffsets(1));
            MatrixTimesMultiVector(*Wloc, subm_as_mv, rhs_view_p);
            if (ssps.GetLocalOffsets(1) != 0)
            {
                ssps.Solve(rhs_view_u, rhs_view_p); // solve for bubbles
            }
            subm.SetSize(ssps.GetLocalOffsets(1), nrhs_RangeT);
            rhs_view_u.CopyToDenseMatrix(0, nrhs_RangeT, ssps.GetLocalOffsets(0),
                                         ssps.GetLocalOffsets(1), subm);
            P_[jform]->SetSubMatrix(fineUDof_Internal, coarseUDof_InternalRangeT,
                                    subm);

            for (int i(0); i<nrhs_RangeT; ++i)
            {
                coarser_sequence->D_[jform]->Set(
                    coarsePDof_Internal[i], coarseUDof_InternalRangeT[i], 1.);
                uCDofHandler->SetDofType(
                    coarseUDof_InternalRangeT[i], DofHandlerALG::RangeTSpace);
            }
        }

        // (5) Solve for Null Dofs (extension of target traces)
        //     these do not exist in the hPeakExtension version of this,
        //     that is, there are no Null Dofs in H1
        const int nlocaltargets = (LocalTargets_[idx].size() > 0) ?
            LocalTargets_[idx][iAE]->NumberOfVectors() : 0;
        if (ridge_stuff && ssps.GetLocalOffsets(1) > nrhs_RangeT &&
            (nTargets + nlocaltargets))
        {
            nrhs_Null = nTargets + nlocaltargets;
            MultiVector localTargets_Interior(nrhs_Null,
                                              fineUDof_Internal.Size());
            MultiVector localTargets_Bdr(nrhs_Null, fineUDof_Bdr.Size());
            Targets_[jform]->GetSubMultiVector(fineUDof_Internal,
                                               localTargets_Interior);
            Targets_[jform]->GetSubMultiVector(fineUDof_Bdr, localTargets_Bdr);

            if (nlocaltargets > 0)
            {
                PartitionLocalTargets(
                    nTargets, nlocaltargets, *LocalTargets_[idx][iAE],
                    fineUDof_Bdr.Size(), fineUDof_Internal.Size(),
                    localTargets_Interior, localTargets_Bdr);
            }

            // set up local rhs
            ssps.SetUpRhs(nrhs_Null, rhs_view_u, rhs_view_p);
            MatrixTimesMultiVector(-1., *Mloc_ib, localTargets_Bdr, rhs_view_u);
            MatrixTimesMultiVector(ssps.GetBloc(), localTargets_Interior,
                                   rhs_view_p);

            if (ssps.GetLocalOffsets(1) != 0)
            {
                ssps.Solve(rhs_view_u, rhs_view_p); // extend target traces
            }
            add(1., localTargets_Interior, -1., rhs_view_u,
                localTargets_Interior);

#ifdef ELAG_DEBUG
            {
                MultiVector dummy(
                    nrhs_Null,ssps.GetLocalOffsets(2) - ssps.GetLocalOffsets(1));
                MatrixTimesMultiVector(ssps.GetBloc(), localTargets_Interior,
                                       dummy);
                double norminf = dummy.Normlinf();

                if (norminf > LOOSE_EQUALITY_TOL)
                {
                    std::cerr << "Adding divergence-free bubble, iAE = "
                              << iAE
                              << ", || D u_loc ||_inf = " << norminf
                              << std::endl;
                    PARELAG_ASSERT(norminf <= LOOSE_EQUALITY_TOL);
                }
            }
#endif
            svd.ComputeON(localTargets_Interior, sv);
            const double s_max_tol = SVD_Tolerance_;

            nrhs_Null = 0;
            coarseUDof_InternalNull.SetSize(nrhs_Null );
            for (; nrhs_Null < sv.Size(); ++nrhs_Null)
            {
                if (sv(nrhs_Null) < s_max_tol)
                    break;
                uCDofHandler->SetDofType(coarseDofCounter,
                                         DofHandlerALG::NullSpace);
                coarseUDof_InternalNull.Append(coarseDofCounter++);
            }
            nNullSpaceDofs += nrhs_Null;
            uCDofHandler->SetNumberOfInteriorDofsNullSpace(codim_dom, iAE,
                                                           nrhs_Null);
            subm.SetSize(ssps.GetLocalOffsets(1) - ssps.GetLocalOffsets(0),
                         nrhs_Null);
            localTargets_Interior.CopyToDenseMatrix(
                0, nrhs_Null, ssps.GetLocalOffsets(0), ssps.GetLocalOffsets(1), subm);

            P_[jform]->SetSubMatrix(fineUDof_Internal,coarseUDof_InternalNull,
                                    subm);
        }
        else
        {
            nrhs_Null = 0;
            uCDofHandler->SetNumberOfInteriorDofsNullSpace(codim_dom, iAE,
                                                           nrhs_Null);
            coarseUDof_InternalNull.SetSize(nrhs_Null );
        }

        // create cochain projector Pi from fine to coarse
        {
            unique_ptr<MultiVector> myVect = BuildCochainProjector(
                nrhs_RangeT, nrhs_Null, coarseUDof_InternalRangeT,
                coarseUDof_InternalNull, *P_[jform], fineUDof_Internal,
                ssps.GetLocalOffsets(1) - ssps.GetLocalOffsets(0));
            Pi_[jform]->CreateDofFunctional(codim_dom, iAE, *myVect,
                                            ssps.GetMloc());
        }

        // (6) Coarsen part of the mass matrix
        {
            int nlocalbasis = nrhs_ext + nrhs_RangeT + nrhs_Null;
            int uAllStart, uAllEnd;
            DofAgg_[jform]->GetAgglomerateDofRange(
                codim_dom, iAE, uAllStart, uAllEnd);
            DofAgg_[jform]->GetViewAgglomerateDofGlobalNumering(
                codim_dom, iAE, fineUDof);
            auto M_aa = ExtractSubMatrix(*M_d, uAllStart, uAllEnd,
                                         uAllStart, uAllEnd);
            auto cElemMass = CoarsenMassMatrixPart(
                nlocalbasis, coarseUDof_on_Bdr,
                coarseUDof_InternalRangeT, coarseUDof_InternalNull,
                *P_[jform], fineUDof, *M_aa);
            mass.SetElementalMatrix(iAE, std::move(cElemMass));
        }
    }
    AE_NullSpaceDof.reset();

    DeRhamSequence_os << "*** Number of dofs that have been extended "
                      << R_t->Size() << std::endl
                      << "*** Number of RangeTSpace Dofs             "
                      << nRangeTDofs << std::endl
                      << "*** Number of NullSpace Dofs               "
                      << nNullSpaceDofs << std::endl;

    uCDofHandler->BuildEntityDofTable(codim_dom);
    coarser_sequence->M_[idx] = mass.GetAsSparseMatrix();

    PARELAG_TEST_FOR_EXCEPTION(
        uCDofHandler->GetNumberInteriorDofs(codim_dom) != coarseDofCounter-R_t->Size(),
        std::runtime_error,
        "DeRhamSequence::hRidgePeakExtension(): "
        "Number of Interior Dof according to DofHandler: "
        << uCDofHandler->GetNumberInteriorDofs(codim_dom ) << std::endl
        << "Actual of Interior Dof: "
        << coarseDofCounter - R_t->Size() << std::endl);
}

/**
  Trying to modularize hRidgeExtension, hPeakExtension means that
  we don't have access to the matrices etc to do these debug checks.
  So here we extract the data structures again and separate them out,
  which is slower but arguably makes the code prettier.
*/
void DeRhamSequence::CheckLocalExactnessCommuting(
    int jform, AgglomeratedTopology::Entity codim_dom)
{
    auto coarser_sequence = CoarserSequence_.lock();
    PARELAG_ASSERT(coarser_sequence);

    // check exactness
    {
        auto D_d = DistributeAgglomerateMatrix(
            codim_dom, *(D_[jform]), DofAgg_[jform+1].get(),
            DofAgg_[jform].get());
        auto D2_d = DistributeAgglomerateMatrix(
            codim_dom, *(D_[jform+1]), DofAgg_[jform+2].get(),
            DofAgg_[jform+1].get());
        unique_ptr<SparseMatrix> DD(Mult(*D2_d,*D_d));
        const double DDnorm = DD->MaxNorm();
        if (DDnorm > DEFAULT_EQUALITY_TOL)
            std::cerr << "DeRhamSequence not locally exact! "
                      << "DDnorm " << DDnorm << "\n";
    }

    // check that DeRhamSequence (partially) commutes
    // in hRidgeExtension, this tolerance was quite loose (1.e-4)
    // while in hPeakExtension it was tight (DEFAULT_EQUALITY_TOL = 1.e-9)
    constexpr double LOCAL_COMMUTING_TOL = 1.e-4;
    {
        unique_ptr<SparseMatrix> PDc{
            MultAbstractSparseMatrix(*(P_[jform+1]),
                                     *(coarser_sequence->D_[jform])) };
        unique_ptr<SparseMatrix> D2PDc(Mult(*(D_[jform+1]),*PDc));
        const double norm = D2PDc->MaxNorm();
        if (norm > LOCAL_COMMUTING_TOL)
        {
            if (false)
            {
                std::string fname = AppendProcessId(Comm_, "Ridge_D2PDc", "mtx");
                std::ofstream fid(fname.c_str());
                D2PDc->PrintMatlab(fid);
                fid.close();
                fname = AppendProcessId(Comm_, "deRS", "log");
                fid.open(fname.c_str());
                fid << DeRhamSequence_os.str();
                fid.close();
            }
            std::cerr << "DeRhamSequence does not locally commute! "
                      << "matrix norm for derivative for target: NORM: = "
                      << norm << std::endl;
        }
    }
}

unique_ptr<SparseMatrix>
DeRhamSequence::getUnextendedInterpolator(int jform)
{
    auto coarser_sequence = CoarserSequence_.lock();
    PARELAG_ASSERT(coarser_sequence);

    const int nFineDofs = P_[jform]->Size();
    const int nCoarseDofs = P_[jform]->Width();

    auto out = make_unique<SparseMatrix>(nFineDofs,nCoarseDofs);

    Array<int> internalAggDofs, internalCDofs;

    const int baseCodim = Dof_[jform]->GetMaxCodimensionBaseForDof();

    Array<double> subm_data;
    DenseMatrix subm;
    for (int codim(baseCodim); codim >= 0; --codim)
    {
        int nentities = coarser_sequence->Topo_->GetNumberLocalEntities(
            static_cast<AgglomeratedTopology::Entity>(codim));
        for (int ientity(0); ientity < nentities; ++ientity)
        {
            DofAgg_[jform]->GetViewAgglomerateInternalDofGlobalNumering(
                static_cast<AgglomeratedTopology::Entity>(codim), ientity,
                internalAggDofs);
            coarser_sequence->Dof_[jform]->GetInteriorDofs(
                static_cast<AgglomeratedTopology::Entity>(codim), ientity,
                internalCDofs);
            subm_data.SetSize(internalAggDofs.Size() * internalCDofs.Size(), 0.0);
            subm.UseExternalData(subm_data, internalAggDofs.Size(),
                                 internalCDofs.Size());
            P_[jform]->GetSubMatrix(internalAggDofs, internalCDofs, subm);
            out->SetSubMatrix(internalAggDofs, internalCDofs, subm );
            subm.ClearExternalData();
        }
    }
    out->Finalize();
    return out;
}

void DeRhamSequence::ShowProjector(int jform)
{
    if(jform > nForms_-1)
        mfem_error("ShowProjector has a too high jform");

    showP(jform, *P_[jform], Topo_->Partitioning());
}

void DeRhamSequence::ShowDerProjector(int jform)
{
    if(jform >= nForms_-1)
        mfem_error("ShowDerProjector has a too high jform");

    unique_ptr<SparseMatrix> dP{
        Mult(*D_[jform],*P_[jform]) };
    showP(jform+1, *dP, Topo_->Partitioning());
}

unique_ptr<ElementalMatricesContainer>
DeRhamSequence::GetElementalMassMatrices(
    int iform,AgglomeratedTopology::Entity icodim)
{
    const SparseMatrix & entity_dof = Dof_[iform]->GetEntityRDofTable(icodim);
    const int nEntities = entity_dof.Height();
    const int * i_entity_dof = entity_dof.GetI();

    auto out = make_unique<ElementalMatricesContainer>(nEntities);

    const int nDim = Topo_->Dimensions();
    const int idx = (nDim-iform)*(nForms_-iform)/2 + icodim;

    for(int ie(0); ie < nEntities; ++ie)
    {
        const int start    = i_entity_dof[ie];
        const int end      = i_entity_dof[ie+1];
        auto Mloc = ExtractSubMatrix(*M_[idx],start,end,start,end);
        auto Mloc_dense = make_unique<DenseMatrix>(Mloc->Size());
        Full(*Mloc,*Mloc_dense);
        out->SetElementalMatrix(ie,std::move(Mloc_dense));
    }
    return out;
}

/*-------------------------------------------*/

DeRhamSequenceAlg::DeRhamSequenceAlg(
    const std::shared_ptr<AgglomeratedTopology>& topo, int nforms)
    : DeRhamSequence(topo, nforms) {}

unique_ptr<SparseMatrix>
DeRhamSequenceAlg::ComputeProjectorFromH1ConformingSpace(int jform) const
{
    auto finer_sequence = FinerSequence_.lock();
    PARELAG_ASSERT(finer_sequence);

    auto pi_fine = finer_sequence->ComputeProjectorFromH1ConformingSpace(jform);

    // NOTE (trb 12/30/15): This is probably overkill, but now the
    // pointer is managed when it is created and not managed when it
    // is only viewed.
    unique_ptr<SparseMatrix> conformingSpaceInterp_unique;
    SparseMatrix * conformingSpaceInterpolator;

    if(jform == nForms_-1)
    {
        conformingSpaceInterpolator = finer_sequence->GetP(0);
    }
    else
    {
        unique_ptr<SparseMatrix> id{
            createSparseIdentityMatrix(nForms_-1) };
        conformingSpaceInterp_unique = Kron(*(finer_sequence->GetP(0)), *id);
        conformingSpaceInterpolator = conformingSpaceInterp_unique.get();
    }

    finer_sequence->GetPi(jform)->ComputeProjector();
    const SparseMatrix & jspaceProjector
        = finer_sequence->GetPi(jform)->GetProjectorMatrix();

    unique_ptr<SparseMatrix> tmp{
        Mult(*pi_fine, *conformingSpaceInterpolator) };
    unique_ptr<SparseMatrix> pi{
        Mult(const_cast<SparseMatrix &>(jspaceProjector), *tmp)};

    return pi;
}

void DeRhamSequenceAlg::computePVTraces(
    AgglomeratedTopology::Entity icodim,
    Vector & PVinAgg)
{
    const int jform = nForms_-1-icodim;
    const int nDofs = Dof_[jform]->GetNDofs();
    const int nAE = DofAgg_[jform]->GetNumberCoarseEntities(icodim);

    PARELAG_TEST_FOR_EXCEPTION(
        PVinAgg.Size() != nDofs,
        std::invalid_argument,
        "DeRhamSequenceAlg::computePVTraces(): PVinAgg is wrong size.");

    // j_entity_dof[ i_entity_dof[i] ] is the PV dof on entity i
    const SparseMatrix & entity_dof = Dof_[jform]->GetEntityDofTable(icodim);
    const int * const i_entity_dof = entity_dof.GetI();
    const int * const j_entity_dof = entity_dof.GetJ();

    const SparseMatrix & AE_e = Topo_->AEntityEntity(icodim);
    const int * j_AE_e = AE_e.GetJ();
    const double * v_AE_e = AE_e.GetData();
    const int nnz_AE_e = AE_e.NumNonZeroElems();

    PARELAG_TEST_FOR_EXCEPTION(
        AE_e.Size() != nAE,
        std::runtime_error,
        "DeRhamSequenceAlg::computePVTraces(): AEntityEntity has wrong size.");

    PARELAG_TEST_FOR_EXCEPTION(
        AE_e.Width() != entity_dof.Size(),
        std::runtime_error,
        "DeRhamSequenceAlg::computePVTraces(): AEntityEntity has wrong size.");

    PVinAgg = 0.0;
    for (const int * const j_AE_e_end = j_AE_e+nnz_AE_e;
         j_AE_e != j_AE_e_end;
         ++j_AE_e, ++v_AE_e)
    {
        // put +/- 1 in PVinAgg
        PVinAgg(j_entity_dof[i_entity_dof[*j_AE_e]]) = *v_AE_e;
    }
}

void DeRhamSequenceAlg::showP(int jform,
                              SparseMatrix & Pc,
                              Array<int> & parts_c)
{
    auto finer_sequence = FinerSequence_.lock();
    PARELAG_ASSERT(finer_sequence);

    unique_ptr<SparseMatrix> PP{
        Mult(*(finer_sequence->GetP(jform)), Pc)};
    Array<int> & parts_f = Topo_->FinerTopology()->Partitioning();
    const int nfpart = parts_f.Size();
    Array<int> fParts(nfpart);

    for(int i(0); i <nfpart; ++i)
        fParts[i] = parts_c[parts_f[i]];

    finer_sequence->showP(jform,*PP,fParts);
}

void DeRhamSequenceAlg::show(int jform, MultiVector & v)
{
    auto finer_sequence = FinerSequence_.lock();
    PARELAG_ASSERT(finer_sequence);

    MultiVector vFine(v.NumberOfVectors(), finer_sequence->GetP(jform)->Size() );
    MatrixTimesMultiVector(*(finer_sequence->GetP(jform) ), v, vFine);
    finer_sequence->show(jform, vFine);
}

void DeRhamSequenceAlg::ShowTrueData(int jform, MultiVector & true_v)
{
   auto finer_sequence = FinerSequence_.lock();
   PARELAG_ASSERT(finer_sequence);

   MultiVector vFine(true_v.NumberOfVectors(), finer_sequence->GetNumTrueDofs(jform));
   Mult(finer_sequence->GetTrueP(jform), true_v, vFine);
   finer_sequence->ShowTrueData(jform, vFine);
}

void DeRhamSequenceAlg::ExportGLVis(int jform, Vector & v, std::ostream & os)
{
    auto finer_sequence = FinerSequence_.lock();
    PARELAG_ASSERT(finer_sequence);

    elag_assert(v.Size() == this->GetNumberOfDofs(jform));
    Vector vFine(finer_sequence->GetNumberOfDofs(jform));
    finer_sequence->GetP(jform)->Mult(v, vFine);
    finer_sequence->ExportGLVis(jform, vFine, os);
}

void DeRhamSequenceAlg::ProjectCoefficient(int jform,
                                           Coefficient & c,
                                           Vector & v)
{
    auto finer_sequence = FinerSequence_.lock();
    PARELAG_ASSERT(finer_sequence);

    PARELAG_TEST_FOR_EXCEPTION(
        jform != 0 && jform != nForms_-1,
        std::invalid_argument,
        "DeRhamSequenceAlg::ProjectCoefficient(): "
        "Invalid jform");

    v.SetSize(finer_sequence->GetP(jform)->Width());

    Vector vf(finer_sequence->GetP(jform)->Size());
    finer_sequence->ProjectCoefficient(jform, c, vf);
    finer_sequence->GetPi(jform)->GetProjectorMatrix().Mult(vf, v);
}

void DeRhamSequenceAlg::ProjectVectorCoefficient(int jform,
                                                 VectorCoefficient & c,
                                                 Vector & v)
{
    auto finer_sequence = FinerSequence_.lock();
    PARELAG_ASSERT(finer_sequence);

    PARELAG_TEST_FOR_EXCEPTION(
        jform == 0 || jform == nForms_-1,
        std::invalid_argument,
        "DeRhamSequenceAlg::ProjectVectorCoefficient(): "
        "Invalid jform");

    v.SetSize(finer_sequence->GetP(jform)->Width());

    Vector vf(finer_sequence->GetP(jform)->Size());
    finer_sequence->ProjectVectorCoefficient(jform, c, vf);
    finer_sequence->GetPi(jform)->GetProjectorMatrix().Mult(vf, v);
}

std::stringstream DeRhamSequence::DeRhamSequence_os;
}//namespace parelag
