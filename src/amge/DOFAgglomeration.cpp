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

#include <memory>
#include <numeric>

#include <mfem.hpp>

#include "DOFAgglomeration.hpp"

#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "linalg/utilities/ParELAG_SubMatrixExtraction.hpp"
#include "linalg/dense/ParELAG_MultiVector.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{

using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;

DofAgglomeration::DofAgglomeration(
    const shared_ptr<AgglomeratedTopology>& topo, DofHandler * dof)
    : FineTopology_(topo),
      CoarseTopology_(topo->CoarserTopology()),
      Dof_(dof),
      AEntity_Dof_(Dof_->GetMaxCodimensionBaseForDof()+1),
      ADof_Dof_I_(Dof_->GetMaxCodimensionBaseForDof()+1),
      ADof_Dof_(Dof_->GetMaxCodimensionBaseForDof()+1),
      ADof_rDof_(Dof_->GetMaxCodimensionBaseForDof()+1),
      AE_nInternalDof_(Dof_->GetMaxCodimensionBaseForDof()+1),
      dofMapper_{}
{
    /*
     * dof_separatorType[i] = 0
     *    --> i belongs to the interior of an agglomerated element
     * dof_separatorType[i] = 1
     *    --> i belongs to the interior of an agglomerated facet
     * dof_separatorType[i] = 2
     *    --> i belongs to the interior of an agglomerated ridge
     * dof_separatorType[i] = 3
     *    --> i belongs to an agglomerated peak
     */
    std::vector<int> dof_separatorType(dof->GetNDofs());

    /* Compute preliminary AEntity_Dof_ by multiplying AEntityEntity
       by (fine) EntityDof. This should give the nonzero structure but
       still need to reorder interior/exterior and give correct values
       (orientations) */
    const auto num_entity_types = AEntity_Dof_.size();
    for (auto i = decltype(num_entity_types){0}; i < num_entity_types; ++i)
    {
        AEntity_Dof_[i].reset(
            Mult(topo->AEntityEntity(i),
                 Dof_->GetEntityDofTable(static_cast<DofHandler::entity>(i))) );
    }

    // (2) Compute the dof_separatorType:
    std::fill(dof_separatorType.begin(),dof_separatorType.end(),0);
    for (auto codim = decltype(num_entity_types){1}; codim < num_entity_types;
         ++codim)
    {
        const int nAE = AEntity_Dof_[codim]->Size();
        const int * const i = AEntity_Dof_[codim]->GetI();
        const int * const j = AEntity_Dof_[codim]->GetJ();
        const int * end = j;
        const int * j_it = j;
        for (const int * i_it = i; i_it != i+nAE; ++i_it)
        {
            for (end = j + *(i_it+1); j_it != end; ++j_it)
                dof_separatorType[*j_it] = codim;
        }
    }

    // loop over entity types = {elements, facets, ridges, peaks}
    for (auto i = decltype(num_entity_types){0}; i < num_entity_types; ++i)
    {
        const int nAGGdof = AEntity_Dof_[i]->NumNonZeroElems();
        const int nDof = AEntity_Dof_[i]->Width();
#ifdef ELAG_DEBUG
        const int nRDof
            = Dof_->GetEntityDofTable(
                static_cast<DofHandler::entity>(i)).NumNonZeroElems();
#endif

        // (3) Change the order of the entries in each row of AEntity_Dof_
        //     so that internal dofs appear first, fix the values of
        //     AEntity_Dof_.
        const int nAE = AEntity_Dof_[i]->Size();
        const int * i_AEntity_dof = AEntity_Dof_[i]->GetI();
        int * j_AEntity_dof = AEntity_Dof_[i]->GetJ(); //
        double * a_AEntity_dof = AEntity_Dof_[i]->GetData();

        AE_nInternalDof_[i].resize(nAE);
        int * tmp = AE_nInternalDof_[i].data();
        if (decltype(i){Dof_->GetMaxCodimensionBaseForDof()} > i)
        {
            // sort each row based on dof_separatorType, put count of dofs
            // of type i (that is, interior) in AE_nInternalDof_
            for (int iAE(0); iAE < nAE; ++iAE)
            {
                tmp[iAE] = reorderRow(
                    i_AEntity_dof[iAE+1] - i_AEntity_dof[iAE],
                    j_AEntity_dof+i_AEntity_dof[iAE],
                    a_AEntity_dof+i_AEntity_dof[iAE],
                    i,
                    Dof_->GetMaxCodimensionBaseForDof()+1,
                    dof_separatorType);
            }
        }
        else
        {
            // no sorting, but still count interior
            for (int iAE(0); iAE < nAE; ++iAE)
                tmp[iAE] = i_AEntity_dof[iAE+1] - i_AEntity_dof[iAE];
        }

        // Agglomerate dofs always have same orientation as global dofs
        // formerly we considered other choices.
        std::fill(a_AEntity_dof, a_AEntity_dof+nAGGdof, 1.0);

        // (4) Compute ADof_Dof (basically a reshaping of AEntity_Dof_)
        ADof_Dof_I_[i].resize(nAGGdof+1);
        std::iota(ADof_Dof_I_[i].begin(),ADof_Dof_I_[i].end(),0);
        ADof_Dof_[i] = make_unique<SparseMatrix>(
            ADof_Dof_I_[i].data(), AEntity_Dof_[i]->GetJ(),
            AEntity_Dof_[i]->GetData(), nAGGdof, nDof,
            false,false,false);
        auto Dof_rDof = ToUnique(
            Transpose(Dof_->GetrDofDofTable(static_cast<DofHandler::entity>(i))));

#ifdef ELAG_DEBUG
        PARELAG_TEST_FOR_EXCEPTION(
            Dof_rDof->Width() != Dof_rDof->NumNonZeroElems() ||
            nRDof != Dof_rDof->Width(),
            std::runtime_error,
            "DofAgglomeration::DofAgglomeration():\n"
            "nRDof " << nRDof << "\nDof_rDof->Width() " << Dof_rDof->Width() <<
            "\nDof_rDof->NumNonZeroElems() " << Dof_rDof->NumNonZeroElems() <<
            "\nrDof_Dof table from DofHandler is not sane!");
#endif

        // (5) build ADof_rDof_ (the hardest, longest part)
        auto AE_rDof = ToUnique(
            Mult(topo->AEntityEntity(i),
                 Dof_->GetEntityRDofTable(static_cast<DofHandler::entity>(i))));
        auto rDof_AE = ToUnique(Transpose(*AE_rDof));

        PARELAG_TEST_FOR_EXCEPTION(
            AE_rDof->Width() < AE_rDof->NumNonZeroElems(),
            std::runtime_error,
            "DofAgglomeration::DofAgglomeration():\n"
            "AE_rDof->Width() = " << AE_rDof->Width() << "\n"
            "AE_rDof->NumNonZeroElems() = " << AE_rDof->NumNonZeroElems() <<
            "\nSome rDof belongs to more than 1 AE.");
        PARELAG_TEST_FOR_EXCEPTION(
            AE_rDof->Width() != Dof_rDof->Width(),
            std::runtime_error,
            "DofAgglomeration::DofAgglomeration():\n"
            "AE_rDof->Width() " << AE_rDof->Width() << "\n"
            "Dof_rDof->Width() " << Dof_rDof->Width() <<
            "\nTwo ways of counting rDofs disagree!");

        int nnz = 0;
        // claim below is that AE_rDof and ADof_rDof should have same number of
        // nonzeros, ie, an rDof appears in exactly one AE, and exactly one ADof
        const int estimated_nnz = AE_rDof->NumNonZeroElems();
        int * i_ADof_rDof = new int[nAGGdof+1]; i_ADof_rDof[0] = 0;
        int * j_ADof_rDof = new int[estimated_nnz];
        double * a_ADof_rDof = new double[estimated_nnz];
        ADof_rDof_[i] = make_unique<SparseMatrix>(
            i_ADof_rDof,j_ADof_rDof,a_ADof_rDof,nAGGdof,Dof_rDof->Width());

        const int * i_Dof_rDof = Dof_rDof->GetI();
        int * j_Dof_rDof = Dof_rDof->GetJ();
        double * a_Dof_rDof = Dof_rDof->GetData();

        const int * i_rDof_AE = rDof_AE->GetI();
        const int * j_rDof_AE = rDof_AE->GetJ();

        const int * i_ADof_dof = ADof_Dof_[i]->GetI();
        const int * j_ADof_dof = ADof_Dof_[i]->GetJ();
        const double * a_ADof_dof = ADof_Dof_[i]->GetData();
        int gdof;
        double sign_Adof;

        // The loop below is sort of like a multiplication of
        //   ADof_Dof \times Dof_rDof
        // but done somehow AE by AE, and with special care taken for ordering
        // internal and boundary dofs, and correcting orientation.
        for (int iAE(0); iAE < nAE; ++iAE)
        {
            // first for internal dofs
            auto range = GetAgglomerateInternalDofRange(
                static_cast<entity_type>(i), iAE);
            for (int it(range.first); it != range.second; ++it)
            {
                gdof = j_ADof_dof[i_ADof_dof[it]];
                sign_Adof = a_ADof_dof[ i_ADof_dof[it] ];
                int * rdof_p;
                double * val_rdof_p = a_Dof_rDof+i_Dof_rDof[gdof];
                // loop over row of Dof_rDof corresponding to gdof
                for (rdof_p = j_Dof_rDof+i_Dof_rDof[gdof];
                     rdof_p != j_Dof_rDof+i_Dof_rDof[gdof+1];
                     ++rdof_p, ++val_rdof_p)
                {
                    if (i_rDof_AE[*rdof_p+1] - i_rDof_AE[*rdof_p] == 0)
                    {
                        // Skip this rdof since it does not belong to
                        // any agglomerate!!
                        continue;
                    }
                    *(j_ADof_rDof+nnz) = *rdof_p;
                    *(a_ADof_rDof+nnz) = *val_rdof_p * sign_Adof;
                    nnz++;
                }
                i_ADof_rDof[it+1] = nnz;
            }

            // then for boundary dofs
            range = GetAgglomerateBdrDofRange(
                static_cast<entity_type>(i), iAE);
            for (int it(range.first); it != range.second; ++it)
            {
                gdof = j_ADof_dof[i_ADof_dof[it]];
                sign_Adof = a_ADof_dof[ i_ADof_dof[it] ];
                int * rdof_p;
                double * val_rdof_p = a_Dof_rDof+i_Dof_rDof[gdof];
                // loop over row of Dof_rDof corresponding to gdof
                for (rdof_p = j_Dof_rDof+i_Dof_rDof[gdof];
                     rdof_p != j_Dof_rDof+i_Dof_rDof[gdof+1];
                     ++rdof_p, ++val_rdof_p)
                {
                    // *(a_ADof_rDof+i_ADof_rDof[it]) = rDof
                    // for all rDof associated to gdof that belongs to iAE.
#ifdef ELAG_DEBUG
                    PARELAG_TEST_FOR_EXCEPTION(
                        *rdof_p >= Dof_rDof->Width(),
                        std::runtime_error,
                        "DofAgglomeration::DofAgglomeration(): "
                        "FIXME: too large *rdof_p");
#endif
                    if (i_rDof_AE[*rdof_p+1] - i_rDof_AE[*rdof_p] == 0)
                    {
                        // Skip rdof since it does not belong to any
                        // agglomerate
                        continue;
                    }
#ifdef ELAG_DEBUG
                    PARELAG_TEST_FOR_EXCEPTION(
                        i_rDof_AE[*rdof_p+1] - i_rDof_AE[*rdof_p] != 1,
                        std::runtime_error,
                        "DofAgglomeration::DofAgglomeration():\n"
                        "FIXME: i_rDof_AE[*rdof_p+1] - i_rDof_AE[*rdof_p] != 1");
#endif
                    if ( j_rDof_AE[ i_rDof_AE[*rdof_p] ] == iAE )
                    {
                        PARELAG_TEST_FOR_EXCEPTION(
                            nnz >= estimated_nnz, std::runtime_error,
                            "Incorrect matrix structure, probably related to "
                            "topological issues.");
                        *(j_ADof_rDof+nnz) = *rdof_p;
                        *(a_ADof_rDof+nnz) = *val_rdof_p * sign_Adof;
                        nnz++;
                    }
                }
                i_ADof_rDof[it+1] = nnz;

#ifdef ELAG_DEBUG
                if (i_ADof_rDof[it+1] - i_ADof_rDof[it] == 0)
                {
                    std::ostringstream oss;
                    oss << "i_ADof_rDof[it+1] - i_ADof_rDof[it] == 0\n";
                    for (rdof_p = j_Dof_rDof+i_Dof_rDof[gdof];
                         rdof_p != j_Dof_rDof+i_Dof_rDof[gdof+1];
                         ++rdof_p)
                        oss << *rdof_p << "\t" << i_rDof_AE[*rdof_p] << "\t"
                            << j_rDof_AE[ i_rDof_AE[*rdof_p] ] << "\t"
                            << iAE << '\n';

                    for (rdof_p = j_Dof_rDof+i_Dof_rDof[gdof];
                         rdof_p != j_Dof_rDof+i_Dof_rDof[gdof+1];
                         ++rdof_p)
                        oss << i_rDof_AE[*rdof_p+1] - i_rDof_AE[*rdof_p] << '\n';

                    PARELAG_TEST_FOR_EXCEPTION(
                        i_ADof_rDof[it+1] - i_ADof_rDof[it] == 0,
                        std::runtime_error,
                        "DofAgglomeration::DofAgglomeration():\n"
                        "FIXME: Something has gone wrong.\n" << oss.str());
                }
#endif
            }
        }

        PARELAG_TEST_FOR_EXCEPTION(
            nnz != estimated_nnz,
            std::runtime_error,
            "DofAgglomeration::DofAgglomeration():"
            " nnz is " << nnz << ". estimated nnz is " << estimated_nnz << "\n"
            "Unstructured agglomerated topology is probably bad.");
    }
}

auto DofAgglomeration::GetAgglomerateInternalDofRange(
    entity_type entity, index_type entity_id) const -> range_type
{
    auto ret = std::make_pair(AEntity_Dof_[entity]->GetI()[entity_id],
                              AEntity_Dof_[entity]->GetI()[entity_id]);
    ret.second += AE_nInternalDof_[entity][entity_id];
    return ret;
}

void DofAgglomeration::GetAgglomerateInternalDofRange(
    entity_type entity, int entity_id,
    int & begin, int & end) const
{
    begin = AEntity_Dof_[entity]->GetI()[entity_id];
    end = begin + AE_nInternalDof_[entity][entity_id];
}

auto DofAgglomeration::GetAgglomerateBdrDofRange(
    entity_type entity, index_type entity_id) const -> range_type
{
    return std::make_pair(AEntity_Dof_[entity]->GetI()[entity_id] +
                          AE_nInternalDof_[entity][entity_id],
                          AEntity_Dof_[entity]->GetI()[entity_id+1]);
}

void DofAgglomeration::GetAgglomerateBdrDofRange(
    entity_type entity, int entity_id, int & begin, int & end) const
{
    begin = AEntity_Dof_[entity]->GetI()[entity_id] +
        AE_nInternalDof_[entity][entity_id];
    end = AEntity_Dof_[entity]->GetI()[entity_id+1];
}

auto DofAgglomeration::GetAgglomerateDofRange(
    entity_type entity, index_type entity_id) const -> range_type
{
    return std::make_pair(AEntity_Dof_[entity]->GetI()[entity_id],
                          AEntity_Dof_[entity]->GetI()[entity_id+1]);
}

void DofAgglomeration::GetAgglomerateDofRange(
    entity_type entity, int entity_id, int & begin, int & end) const
{
    begin = AEntity_Dof_[entity]->GetI()[entity_id];
    end = AEntity_Dof_[entity]->GetI()[entity_id+1];
}

int * DofAgglomeration::GetAgglomerateInternalDofGlobalNumering(
    entity_type entity, int entity_id, int * gdofs)
{
    int * begin = AEntity_Dof_[entity]->GetJ() +
        AEntity_Dof_[entity]->GetI()[entity_id];
    int * end   = begin+AE_nInternalDof_[entity][entity_id];

    for (int * it = begin; it != end; ++it, ++gdofs)
        *gdofs =*it;

    return gdofs;
}

unique_ptr<MultiVector> DofAgglomeration::DistributeGlobalMultiVector(
    entity_type entity, const MultiVector & vg)
{
    PARELAG_TEST_FOR_EXCEPTION(
        vg.Size() != Dof_->GetNDofs(),
        std::runtime_error,
        "DofAgglomeration::DistributeGlobalMultiVector(): "
        "vg has the wrong size!");

    const int nv = vg.NumberOfVectors();
    const int nDistDof = AEntity_Dof_[entity]->NumNonZeroElems();
    auto out = make_unique<MultiVector>(nv, nDistDof);
    Array<int> dofs(AEntity_Dof_[entity]->GetJ(), nDistDof);
    vg.GetSubMultiVector(dofs, *out);
    return out;
}

unique_ptr<Vector> DofAgglomeration::DistributeGlobalVector(
    entity_type entity, Vector & vg)
{
    PARELAG_TEST_FOR_EXCEPTION(
        vg.Size() != Dof_->GetNDofs(),
        std::runtime_error,
        "DofAgglomeration::DistributeGlobalVector():\n"
        "vg.Size() = " << vg.Size() << "\n"
        "Dof_->GetNDofs() = " << Dof_->GetNDofs() << "\n"
        "vg has the wrong size!");

    const int nDistDof = AEntity_Dof_[entity]->NumNonZeroElems();
    auto out = make_unique<Vector>(nDistDof);
    Array<int> dofs(AEntity_Dof_[entity]->GetJ(), nDistDof);
    vg.GetSubVector(dofs, *out);
    return out;
}

void DofAgglomeration::GetViewAgglomerateInternalDofGlobalNumering(
    entity_type entity, int entity_id, Array<int> & gdofs) const
{
    const int begin = AEntity_Dof_[entity]->GetI()[entity_id];
    const int size  = AE_nInternalDof_[entity][entity_id];
    gdofs.MakeRef(AEntity_Dof_[entity]->GetJ()+begin, size);
}

void DofAgglomeration::GetViewAgglomerateBdrDofGlobalNumering(
    entity_type entity, int entity_id, Array<int> & gdofs) const
{
    const int begin = AEntity_Dof_[entity]->GetI()[entity_id] +
        AE_nInternalDof_[entity][entity_id];
    const int end   = AEntity_Dof_[entity]->GetI()[entity_id+1];
    const int size  = end - begin;
    gdofs.MakeRef(AEntity_Dof_[entity]->GetJ()+begin, size);
}

void DofAgglomeration::GetViewAgglomerateDofGlobalNumering(
    entity_type entity, int entity_id, Array<int> & gdofs) const
{
    const int begin = AEntity_Dof_[entity]->GetI()[entity_id] ;
    const int end   = AEntity_Dof_[entity]->GetI()[entity_id+1];
    const int size  = end - begin;
    gdofs.MakeRef(AEntity_Dof_[entity]->GetJ()+begin, size);
}

void DofAgglomeration::CheckAdof()
{
    bool good;
    const auto size = AEntity_Dof_.size();
    for (auto i = decltype(size){0}; i < size; ++i)
    {
        double * end = ADof_rDof_[i]->GetData()+ADof_rDof_[i]->NumNonZeroElems();
        for (const double * it = ADof_rDof_[i]->GetData(); it != end; ++it)
        {
            PARELAG_TEST_FOR_EXCEPTION(
                fabs(*it-1.0) < 1e-6 && fabs(*it-1.0) > 1e-6,
                std::runtime_error,
                "DofAgglomeration::CheckAdof(): "
                "ADof_rDof_[" << i << "] has an entry equal to " << *it << "\n"
                "FIXME: Something has gone wrong.");
        }

        end = ADof_Dof_[i]->GetData()+ADof_Dof_[i]->NumNonZeroElems();
        for (const double * it = ADof_Dof_[i]->GetData(); it != end; ++it)
        {
            PARELAG_TEST_FOR_EXCEPTION(
                fabs(*it-1.0) < 1e-6 && fabs(*it-1.0) > 1e-6,
                std::runtime_error,
                "DofAgglomeration::CheckAdof(): "
                "ADof_Dof_[" << i << "] has an entry equal to " << *it << "\n"
                "FIXME: Something has gone wrong.");
        }

        auto rDof_ADof = ToUnique(Transpose(*ADof_rDof_[i]));
        auto rdof_dof = ToUnique(Mult(*rDof_ADof, *ADof_Dof_[i]));

        end = rdof_dof->GetData()+rdof_dof->NumNonZeroElems();
        for (double * it = rdof_dof->GetData(); it != end; ++it)
        {
             PARELAG_TEST_FOR_EXCEPTION(
                 fabs(*it) < .5,
                 std::runtime_error,
                 "DofAgglomeration::CheckAdof(): "
                 "FIXME: Something has gone wrong.");
        }

        const SparseMatrix & rdof_dof2 =
            Dof_->GetrDofDofTable(static_cast<entity_type>(i));

        // std::cout << "entity codimension: " << i << std::endl;
        good = AreAlmostEqual(*rdof_dof, rdof_dof2, *rdof_dof,
                              "(rDof_aDof*aDof_dof)", "rDof_dof",
                              "(rDof_aDof*aDof_dof)", 1e-6);

        if (!good)
        {
            std::cout << "Num of non-zeros per row rDof_ADof"
                      << rDof_ADof->MaxRowSize() << std::endl
                      << "Num of non-zeros per row ASof_Dof"
                      << ADof_Dof_[i]->MaxRowSize() << std::endl
                      << "Num of non-zeros per row rdof_dof"
                      << rdof_dof->MaxRowSize() << std::endl;
            rdof_dof->PrintMatlab(std::cout<<"Rdof_dof_computed\n");

            // FIXME (trb 06/28/16): Ummmmmm WHAT?!
            exit(1);
        }
    }
}

int DofAgglomeration::reorderRow(
    int nentries, int * j, double * a,
    int minWeight, int, const std::vector<int>& weights)
{
    int nMinWeight(0);

    // SAFER IMPLEMENTATION
    std::vector<Triple<int,int,double>> triples(nentries);

    for (int ii(0); ii < nentries; ++ii)
    {
        triples[ii].one = weights[j[ii]];
        if (triples[ii].one == minWeight)
            ++nMinWeight;
        triples[ii].two = j[ii];
        triples[ii].three = a[ii];
    }

    SortTriple(triples.data(), nentries);

    for (int ii(0); ii < nentries; ++ii)
    {
        j[ii] = triples[ii].two;
        a[ii] = triples[ii].three;
    }

    return nMinWeight;
}


unique_ptr<SparseMatrix> AssembleAgglomerateMatrix(
    DofAgglomeration::entity_type entity,
    SparseMatrix & M_e,
    DofAgglomeration * range,
    DofAgglomeration * domain)
{
    const SparseMatrix * Rt = range->ADof_rDof_[entity].get();
    const SparseMatrix * Pt = domain->ADof_rDof_[entity].get();

    auto P = ToUnique(Transpose(*Pt));
    auto RtM_e = ToUnique(Mult(*Rt, M_e));

    return ToUnique(Mult(*RtM_e, *P));
}

unique_ptr<SparseMatrix> AssembleAgglomerateRowsGlobalColsMatrix(
    DofAgglomeration::entity_type entity, SparseMatrix & M_e,
    DofAgglomeration * range, DofHandler * domain)
{
    const SparseMatrix & Rt = *(range->ADof_rDof_[entity]);
    auto tmp = ToUnique(Mult(Rt, M_e));
    const SparseMatrix & P = domain->GetrDofDofTable(entity);
    return ToUnique(Mult(*tmp,P));
}

unique_ptr<SparseMatrix> Assemble(
    DofAgglomeration::entity_type entity, const SparseMatrix & M_a,
    DofAgglomeration * range, DofAgglomeration * domain)
{
    PARELAG_TEST_FOR_EXCEPTION(
        range == nullptr && domain == nullptr,
        std::runtime_error,
        "Assemble(...): Range and domain can't both be null!");

    int flag(0);
    flag += (domain == nullptr) ? 0 : 1;
    flag += (range == nullptr) ? 0 : 2;

    switch(flag)
    {
    case 1: /*domain != NULL && range == NULL*/
    {
        //std::cout << "Case 1: domain != NULL && range == NULL" << std::endl;
        const SparseMatrix & P = *(domain->ADof_Dof_[entity]);
        return ToUnique(Mult(M_a, P));
    }
    case 2: /*domain == NULL && range != NULL*/
    {
        //std::cout << "Case 2: domain == NULL && range != NULL" << std::endl;
        const SparseMatrix & R = *(range->ADof_Dof_[entity]);
        auto Rt = ToUnique(Transpose(R));
        return ToUnique(Mult(*Rt, M_a));
    }
    case 3: /*domain != NULL && range != NULL*/
    {
        //std::cout << "Case 3: domain != NULL && range != NULL" << std::endl;
        const SparseMatrix & R = *range->ADof_Dof_[entity];
        const SparseMatrix & P = *domain->ADof_Dof_[entity];

        auto Rt = ToUnique(Transpose(R));
        auto RtM_a = ToUnique(Mult(*Rt, M_a));
        return ToUnique(Mult(*RtM_a, P));
    }
    default:
    {
        PARELAG_TEST_FOR_EXCEPTION(
            true,
            std::runtime_error,
            "Assemble(): Impossible value of flag.");
    }
    }//end switch
}

unique_ptr<SparseMatrix> DistributeAgglomerateMatrix(
    DofAgglomeration::entity_type entity,
    const SparseMatrix & D_g,
    DofAgglomeration * range,
    DofAgglomeration * domain)
{

    PARELAG_TEST_FOR_EXCEPTION(
        range == nullptr && domain == nullptr,
        std::runtime_error,
        "Assemble(): Range and domain can't both be null.");

    int flag(0);
    flag += (domain == nullptr) ? 0 : 1;
    flag += (range == nullptr) ? 0 : 2;

    switch(flag)
    {
    case 1: /*domain != NULL && range == NULL*/
    {
        auto P = ToUnique(Transpose(*(domain->ADof_Dof_[entity])));
        return ToUnique(Mult(D_g, *P));
    }
    case 2: /*domain == NULL && range != NULL*/
    {
        return ToUnique(Mult(*(range->ADof_Dof_[entity]), D_g));
    }
    case 3: /*domain != NULL && range != NULL*/
    {
        return Distribute(
            D_g,*(range->AEntity_Dof_[entity]),
            *(domain->AEntity_Dof_[entity]));
    }
    default:
    {
        PARELAG_TEST_FOR_EXCEPTION(
            true, std::runtime_error, "Assemble(): Impossible value of flag");
    }
    }// end switch
}

unique_ptr<SparseMatrix> DistributeProjector(
    DofAgglomeration::entity_type entity,
    const SparseMatrix & P_t,
    DofHandler * coarseRange,
    DofAgglomeration * domain)
{
    return Distribute(
            P_t, coarseRange->GetEntityDofTable(entity),
            *(domain->AEntity_Dof_[entity]));
}
}//namespace parelag
