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

#include "Topology.hpp"

#include <numeric>

#include "structures/connectedComponents.hpp"
#include "structures/minimalIntersectionSet.hpp"
#include "topology/AgglomeratedTopologyCheck.hpp"
#include "partitioning/MFEMRefinedMeshPartitioner.hpp"
#include "utilities/MemoryUtils.hpp"
#include "utilities/mpiUtils.hpp"
#include "utilities/Trace.hpp"

// for book's algorithm in topology coarsening
#include "linalg/utilities/ParELAG_SubMatrixExtraction.hpp"
#include "linalg/utilities/ParELAG_MatrixUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::shared_ptr;
using std::make_shared;
using std::unique_ptr;
using array_t = AgglomeratedTopology::array_t;

AgglomeratedTopology::AgglomeratedTopology(MPI_Comm comm, int ncodim)
    : Comm_(comm),
      nDim_(-1),
      nCodim_(ncodim),
      pMesh_(nullptr),
      B_(nCodim_),
      Weights_(nCodim_+1),
      element_attribute(0),
      facet_bdrAttribute(nullptr),
      entityTrueEntity(nCodim_+1),
      globalAgglomeration(false),
      AEntity_entity(0),
      ATrueEntity_trueEntity(0),
      workspace(nCodim_)
{
    for (auto& vec : Weights_)
        vec = make_unique<array_t>();

    for (auto& map : entityTrueEntity)
        map = make_unique<SharingMap>(Comm_);
}

AgglomeratedTopology::AgglomeratedTopology(
    const shared_ptr<mfem::ParMesh>& pMesh,int nCodim)
    : Comm_(pMesh->GetComm()),
      nDim_(pMesh->Dimension()),
      nCodim_(nCodim),
      pMesh_(pMesh),
      B_(nCodim_),
      Weights_(nDim_),
      element_attribute(0),
      facet_bdrAttribute(nullptr),
      entityTrueEntity(nCodim_+1),
      globalAgglomeration(false),
      AEntity_entity(0),
      ATrueEntity_trueEntity(0),
      workspace(nCodim_)
{
    for (auto& vec : Weights_)
        vec = make_unique<array_t>();

    for (auto& map : entityTrueEntity)
        map = make_unique<SharingMap>(Comm_);

    generateTopology(*pMesh_);
}

void AgglomeratedTopology::generateTopology(mfem::ParMesh& pmesh)
{
    {
        std::vector<unique_ptr<FiniteElementCollection>> fecColl(nDim_+1);

        fecColl[0] = make_unique<L2_FECollection>(0, nDim_);
        fecColl[1] = make_unique<RT_FECollection>(0, nDim_);
        if(nDim_ == 2)
        {
            fecColl[2] = make_unique<H1_FECollection>(1, nDim_);
        }
        else
        {
            fecColl[2] = make_unique<ND_FECollection>(1, nDim_);
            fecColl[3] = make_unique<H1_FECollection>(1, nDim_);
        }

        std::vector<unique_ptr<FiniteElementSpace>> feSpace(nCodim_+1);
        for(int i(0); i < nCodim_+1; ++ i)
            feSpace[i] = make_unique<FiniteElementSpace>(&pmesh,
                                                         fecColl[i].get());

        for(int i(0); i<nCodim_; ++i)
        {
            mfem::DiscreteLinearOperator Bvarf(feSpace[i+1].get(),
                                               feSpace[i].get());

            // NOTE: This DiscreteInterpolator will be deallocated by the
            // destructor of Bvarf
            unique_ptr<DiscreteInterpolator> Bop;
            switch (i){
            case 0:
                Bop = make_unique<DivergenceInterpolator>();
                break;
            case 1:
                if (nDim_ == 2)
                    Bop = make_unique<GradientInterpolator>();
                else
                    Bop = make_unique<CurlInterpolator>();
                break;
            case 2:
                Bop = make_unique<GradientInterpolator>();
                break;
            default:
                PARELAG_TEST_FOR_EXCEPTION(
                    true,
                    std::logic_error,
                    "nCodim_ = " << nCodim_ << "; i = " << i << ".");
            }
            Bvarf.AddDomainInterpolator(Bop.release());
            Bvarf.Assemble();
            Bvarf.Finalize();

            B_[i] = make_unique<TopologyTable>(ToUnique(Bvarf.LoseMat()));
            B_[i]->OrientationTransform();
        }

        // Build the SharingMap for each codimension.
        for(int i(0); i < nCodim_+1; ++i)
            entityTrueEntity[i]->SetUp(pmesh,i);
    }

    initializeWeights(pmesh);

    element_attribute.SetSize(pmesh.GetNE());
    for(int i = 0; i < pmesh.GetNE(); ++i)
        element_attribute[i] = pmesh.GetAttribute(i);

    if(nCodim_ >= 1)
        generateFacetBdrAttributeTable(pmesh);

    BuildConnectivity();
}

void AgglomeratedTopology::initializeWeights(const mfem::Mesh& mesh)
{
    Weights_[ELEMENT] = make_unique<array_t>(mesh.GetNE());

    if(nCodim_ > 0)
    {
        if(nDim_ == 3)
        {
            Weights_[FACET] = make_unique<array_t>(mesh.GetNFaces());
            if(nCodim_ > 1)
                Weights_[RIDGE] = make_unique<array_t>(mesh.GetNEdges());
        }
        else
            Weights_[FACET] = make_unique<array_t>(mesh.GetNEdges());
    }

    for(int i = 0; i < std::min(nCodim_+1, nDim_); ++i)
        *Weights_[i] = DataTraits<data_type>::one();

}

void AgglomeratedTopology::generateFacetBdrAttributeTable(const mfem::Mesh& mesh)
{
    const int nFacets = Weights_[FACET]->Size();
    const int nnz = mesh.GetNBE();

    int * i_mat = new int[nFacets+1];
    std::fill(i_mat, i_mat+nFacets+1, 0);

    if(mesh.bdr_attributes.Size() == 0)
    {
        int * j_mat = new int[1];
        data_type * a_mat = new data_type[1];
        facet_bdrAttribute
            = make_unique<TopologyTable>(i_mat,j_mat,a_mat,nFacets,0);
        return;
    }

    int * j_mat = new int[nnz];
    data_type * a_mat = new data_type[nnz];
    const int nbdrattribs = mesh.bdr_attributes.Max();

    // NOTE: in 2D, MFEM does not orient boundary facets in a
    // consistent way.  Therefore we need to use the sign of
    // facet_element to make sure that the orientation is correct on
    // each boundary surface.
    auto  facet_element = ToUnique(Transpose(*(B_[0])));

    const int * i_facet_element = facet_element->GetI();
    const double * a_facet_element = facet_element->GetData();

    //mark which facet is a boundary facet.
    for(int i = 0; i < nnz; ++i)
    {
        const int index = mesh.GetBdrElementEdgeIndex(i);
        (i_mat[index+1])++;
    }

    // do a partial sum to compute the i array of the CSRmatrix
    std::partial_sum(i_mat, i_mat+nFacets+1, i_mat);

    // Fill in the CSR matrix
    for(int i = 0; i < nnz; ++i)
    {
        // Get the facet index corresponding to boundary element i
        const int index = mesh.GetBdrElementEdgeIndex(i);
        // fill the j array corresponding to facet index to the
        // respective boundary attribute
        j_mat[i_mat[index]] = mesh.GetBdrAttribute(i)-1;
        // This is a redundant check. If index is a bdr facets that it
        // is adjacent to only 1 element.
        PARELAG_ASSERT(i_facet_element[index+1] - i_facet_element[index] == 1);
        // We orient the bdr facet as the opposite of its local
        // orientation in the adjacent element.
        a_mat[i_mat[index]] = -a_facet_element[i_facet_element[index]];
    }

    facet_bdrAttribute
            = make_unique<TopologyTable>(i_mat,j_mat,a_mat,nFacets,nbdrattribs);
}

void AgglomeratedTopology::BuildConnectivity()
{
    for(int codim = 0; codim < nCodim_; ++codim)
    {
        // I'm exploiting that j = i+1 to compute this idx
        const int idx = codim*(7-codim)/2;
        Conn_[idx] = make_unique<BooleanMatrix>(GetB(codim));
    }

    if(nCodim_ > 1)
        // ELEMENT,RIDGE = 1, per Conn_ doc in Topology.hpp
        Conn_[1] = BoolMult(GetB(ELEMENT), GetB(FACET));
    if(nCodim_ > 2)
    {
        // ELEMENT,PEAK = 2, per Conn_ doc in Topology.hpp
        Conn_[2] = BoolMult(*Conn_[1], GetB(RIDGE));
        // FACET,PEAK = 4, per Conn_ doc in Topology.hpp
        Conn_[4] = BoolMult(GetB(FACET), GetB(RIDGE));
    }
}

const BooleanMatrix &
AgglomeratedTopology::GetConnectivity(int range,int domain) const
{
    const int idx = range*(5-range)/2 + domain - 1;

    PARELAG_ASSERT(Conn_[idx]);
    return *Conn_[idx];
}

void
AgglomeratedTopology::GetBoundaryOfEntity(
    Entity type,Entity type_bdr,
    int ientity,Array<int> & bdr_entity) const
{
    const int idx = type*(5-type)/2 + type_bdr - 1;

    PARELAG_ASSERT(Conn_[idx]);
    Conn_[idx]->GetRow(ientity, bdr_entity);
}

mfem::SparseMatrix * AgglomeratedTopology::LocalElementElementTable()
{
    if(workspace.elem_elem_local)
        return workspace.elem_elem_local.get();

    auto Bt = GetB(0).Transpose();

    Bt->ScaleRows(Weight(1));

    workspace.elem_elem_local.reset(Mult(GetB(0),*Bt));

    return workspace.elem_elem_local.get();
}

AgglomeratedTopology::par_table_t *
AgglomeratedTopology::GlobalElementElementTable()
{
    if(workspace.elem_elem_global)
        return workspace.elem_elem_global.get();

    auto Bt = ToUnique(TrueB(0).Transpose());

    Bt->ScaleRows(*(TrueWeight(1)));

    workspace.elem_elem_global.reset(ParMult(&TrueB(0), Bt.get()));

    return workspace.elem_elem_global.get();
}

void AgglomeratedTopology::ShowMe(std::ostream & os)
{
    os << "  N_elements = "
       << std::setw(10) <<  GetNumberLocalEntities(ELEMENT)
       << std::setw(10) <<  GetNumberGlobalTrueEntities(ELEMENT)
       << std::endl;
    os << "  N_facets   = "
       << std::setw(10) <<  GetNumberLocalEntities(FACET)
       << std::setw(10) <<  GetNumberGlobalTrueEntities(FACET)
       << std::endl;
    os << "  N_ridges   = "
       << std::setw(10) <<  GetNumberLocalEntities(RIDGE)
       << std::setw(10) <<  GetNumberGlobalTrueEntities(RIDGE)
       << std::endl;
    if (nDim_ == 3)
        os << "  N_peaks    = "
           << std::setw(10) <<  GetNumberLocalEntities(PEAK)
           << std::setw(10) <<  GetNumberGlobalTrueEntities(PEAK)
           << std::endl;

    if (nDim_ == nCodim_)
    {
        int eulerCharacteristicLocal  = 0;
        int eulerCharacteristicGlobal = 0;
        for (int codim(nDim_); codim >= 0; codim-=2)
        {
            eulerCharacteristicLocal +=
                GetNumberLocalEntities(static_cast<Entity>(codim));
            eulerCharacteristicGlobal +=
                GetNumberGlobalTrueEntities(static_cast<Entity>(codim));
        }
        for (int codim(nDim_-1); codim >= 0; codim-=2)
        {
            eulerCharacteristicLocal -=
                GetNumberLocalEntities(static_cast<Entity>(codim));
            eulerCharacteristicGlobal -=
                GetNumberGlobalTrueEntities(static_cast<Entity>(codim));
        }

        // an Euler characterstic of 1 is "good", a non-1 Euler characterstic
        // on the coarse mesh is associated (always?) with failure later on, but
        // some failing cases also have Euler characteristic 1.
        os << "Euler Characteristic = "
           << std::setw(10) << eulerCharacteristicLocal
           << std::setw(10) << eulerCharacteristicGlobal
           << std::endl;
    }

    std::vector<std::string> entityName;
    entityName.emplace_back("Element");
    entityName.emplace_back("Facet");
    if (nCodim_ == 3)
        entityName.emplace_back("Ridge");

    Array<int> minW(nCodim_), maxW(nCodim_);
    Array<double> meanW(nCodim_);

    for (int codim(0); codim < nCodim_; ++codim)
    {
        array_t & wc(*Weights_[codim]);
        data_type minw(wc[0]);
        data_type maxw(wc[0]);
        data_type sumw(static_cast<data_type>(0));

        for(data_type * it = wc.GetData(); it != wc.GetData()+wc.Size(); ++it)
        {
            minw = (minw < *it) ? minw : *it;
            maxw = (maxw > *it) ? maxw : *it;
            sumw += *it;
        }

        minW[codim] = minw;
        maxW[codim] = maxw;
        meanW[codim] = static_cast<double>(sumw) / static_cast<double>(wc.Size());
    }

    os << "Entity   MIN  MAX AVERAGE (weight)" << std::endl;
    for (int codim(0); codim < nCodim_; ++codim)
        os << entityName[codim] << " " << minW[codim] << " "
           << maxW[codim] << " " << meanW[codim] << std::endl;
}

AgglomeratedTopology * AgglomeratedTopology::FinestTopology()
{
    AgglomeratedTopology * topo = this;

    while (!(topo->FinerTopology_.expired()))
        topo = topo->FinerTopology().get();

    return topo;
}

AgglomeratedTopology::par_table_t & AgglomeratedTopology::TrueB(int i) const
{
    if(workspace.trueB_[i] == nullptr)
        workspace.trueB_[i] = IgnoreNonLocalRange(
            *(entityTrueEntity[i]), *(B_[i]), *(entityTrueEntity[i+1]));

    return *(workspace.trueB_[i]);
}

unique_ptr<array_t> AgglomeratedTopology::TrueWeight(int i) const
{
    auto truew = make_unique<array_t>(
        entityTrueEntity[i]->GetTrueLocalSize());
    entityTrueEntity[i]->IgnoreNonLocal(*(Weights_[i]),*truew);

    return truew;
}

//This function will check the topology of codimension icodim and fix the AEntity_entity table (and its transpose fc_AF) if needed.
void AgglomeratedTopology::CheckHFacetsTopology(int icodim, std::unique_ptr<TopologyTable> & fc_AF)
{
    using ATC = AgglomeratedTopologyCheck;
    std::stringstream os;
    ATC::ShowBadAgglomeratedEntities(icodim, *this, os);
    SerializedOutput(Comm_, std::cout, os.str());
    Array<int> isbad;
    bool anybad = ATC::MarkBadAgglomeratedEntities(icodim,*this,isbad);
    if (anybad)
    {
        DeAgglomerateBadAgglomeratedEntities(isbad,icodim);
        fc_AF = AEntity_entity[icodim]->Transpose();
    }
}

int augment_ia_AF_f_shared(int * ia_AF_face_startshd, SparseMatrix& AE_AEoffd, int * cmapAEAE,
                            SparseMatrix& fc_AEoffd, int * cmapfcAE, TopologyTable& AE_fc );

int augment_ia_AF_f_boundary(int * ia_AF_face_startbnd, TopologyTable& facet_bdrAttribute,
                                                  SparseMatrix& AE_bnd, TopologyTable& AE_fc );

int augment_ia_ja_data_AF_f_boundary2(int * ia_AF_face_startbnd, int * ja_AF_face, double * data_AF_face,
                std::map<int,int>& AEs_withbnd, TopologyTable& fc_AE, SparseMatrix& fc_AEoffd, TopologyTable& AE_fc );

int augment_ja_data_AF_f_shared(int * ia_AF_face_startshd, int * ja_AF_face, double * data_AF_face,
                                int localAEs_startindex, SparseMatrix& AE_AEoffd, int * cmapAEAE,
                                SparseMatrix& fc_AEoffd, int * cmapfcAE, TopologyTable& AE_fc );

int augment_ja_data_AF_f_boundary(int * ia_AF_face_startbnd, int * ja_AF_face, double * data_AF_face,
                                  TopologyTable& facet_bdrAttribute, SparseMatrix& AE_bnd, TopologyTable& AE_fc );

int augment_ja_data_AF_f_inner(int * ia_AF_face, int * ja_AF_face, double * data_AF_face,
            int nAF_inner, SparseMatrix& AFinner_AE_face, SparseMatrix& AFinner_AE, TopologyTable& AE_fc);
  
std::unique_ptr<TopologyTable> AgglomeratedTopology::ComputeCoarseFacets(
        std::shared_ptr<AgglomeratedTopology> CoarseTopology,
        TopologyTable& AE_fc)
{
    int nAF_inner = 0;    // number of connected pairs (AE, AE) where both AEs are local!
    int nAF_shared = 0;   // number of connected pairs (AE,AE) where one of the AEs belong to another process
    int nAF_boundary = 0; // number of connected pairs (AE, bndattribute)

    auto fc_Truefc = entityTrueEntity[1]->get_entity_trueEntity();

    const int nfacets = fc_Truefc->Height(); // used in the very end for output TopologyTable

    int * AErow_starts = CoarseTopology->entityTrueEntity[0]->get_entity_trueEntity()->GetRowStarts();
    int nAE = CoarseTopology->entityTrueEntity[0]->get_entity_trueEntity()->M();
    int * fc_starts = fc_Truefc->GetRowStarts();
    HypreParMatrix AE_fc_bd(Comm_, nAE, fc_Truefc->M(), AErow_starts, fc_starts, &AE_fc);
    auto AE_Truefc = mfem::ParMult(&AE_fc_bd, fc_Truefc);

    auto Truefc_AE = AE_Truefc->Transpose();

    auto globalAE_AE = ParMult(AE_Truefc, Truefc_AE);
    int * row_starts = globalAE_AE->GetRowStarts();
    int localAEs_startindex = row_starts[0];

    SparseMatrix AE_AEdiag; // localAE_to_localAE
    globalAE_AE->GetDiag(AE_AEdiag);

    // computing nAF_inner
    nAF_inner = (AE_AEdiag.NumNonZeroElems() - AE_AEdiag.Height()) / 2;

    SparseMatrix AE_AEoffd; // localAE_to_nonlocalAE (with hypre compression)
    int * cmapAEAE;
    globalAE_AE->GetOffd(AE_AEoffd, cmapAEAE);

    int * offd_i = AE_AEoffd.GetI();
    // computing nAF_shared
    // there is another way to compute it - just count the number of ShareAttributes
    nAF_shared = offd_i[AE_AEoffd.Size()];

    auto fc_Truefc_AE = ParMult(fc_Truefc, Truefc_AE);

    delete Truefc_AE;
    delete AE_Truefc;

    // equivalent to the facet_ShareAttribute which is used the same way as facet_bdrAttribute later
    // special attribute for each facet is the index of nonlocal AE to which this face belongs
    SparseMatrix fc_AEoffd;
    int * cmapfcAE;
    fc_Truefc_AE->GetOffd(fc_AEoffd, cmapfcAE);

    // creating AF_AE for the inner AFs
    int * ia_AFinner_AE = new int [nAF_inner + 1];
    ia_AFinner_AE[0] = 0;
    // each inner AF belongs to a pair of AEs;
    for ( int row = 0; row < nAF_inner; ++row)
        ia_AFinner_AE[row+1] = ia_AFinner_AE[row] + 2;

    int * ja_AFinner_AE = new int [ia_AFinner_AE[nAF_inner]];
    double * data_AFinner_AE = new double [ia_AFinner_AE[nAF_inner]];

    // setting ja and data for AFinner_AE
    int nnz = 0;
    // loop over all local AEs
    for ( int row = 0; row < AE_AEdiag.Height(); ++row)
    {
        int ncols = AE_AEdiag.RowSize(row);
        int * cols = AE_AEdiag.GetRowColumns(row);
        for ( int colno = 0; colno < ncols; ++colno)
            if (cols[colno] > row) // looking for the connected pair (AE1,AE2) with AE1 < AE2
            {
                // we are saying that this AF belongs to the chosen pair of local AEs
                ja_AFinner_AE[ia_AFinner_AE[nnz]] = row;
                ja_AFinner_AE[ia_AFinner_AE[nnz] + 1] = cols[colno];
                data_AFinner_AE[ia_AFinner_AE[nnz]] = 1.0;
                // we set opposite orientation for this AF in the second AE with larger index
                data_AFinner_AE[ia_AFinner_AE[nnz] + 1] = -1.0;
                nnz++;
            }
    }

    auto AFinner_AE = make_unique<TopologyTable>(ia_AFinner_AE, ja_AFinner_AE, data_AFinner_AE, nAF_inner, AE_AEdiag.Height());

    // creating AF_AE_face which has nonzero entries equal to 1 or 2
    // see the description of book's algo
    // for the book's algo we need to forget about orientation at this multiplication
    auto AFinner_AE_face = MultAbs(*AFinner_AE, AE_fc);

    // counting number of boundary AFs and creating necessary structures
    std::unique_ptr<SparseMatrix> AE_bnd;
    // additional structures used in case when facet_bdrAttribute is not available
    std::unique_ptr<TopologyTable> fc_AE;
    // stores pairs (AE, number of boundary facets which belong to this AE)
    std::map<int,int> AEs_withbnd;
    int nfc_boundary = 0;

    if (facet_bdrAttribute) // if boundary is specified
    {
        AE_bnd = ToUnique(Mult(AE_fc, *facet_bdrAttribute)); // AE_fc_boundary
        // computing nAF_boundary
        nAF_boundary = AE_bnd->NumNonZeroElems();
    }
    else // even if we don't have facet_bdrAttribute, we need to identify boundary AFs
    {
        fc_AE = AE_fc.Transpose();

        // counting number of AEs with boundary faces
        nAF_boundary = 0;
        for ( int fc = 0; fc < nfacets; ++fc)
        {
            // if facet belongs to exactly 1 local AE and 0 nonlocal AEs
            if ( fc_AE->RowSize(fc) == 1 && fc_AEoffd.RowSize(fc) == 0 )
            {
                // increasing the map entry for the key = AE index
                AEs_withbnd[fc_AE->GetRowColumns(fc)[0]]++;
                nfc_boundary++;
            }
        }
        nAF_boundary = AEs_withbnd.size();
    }

    // Finally, we are ready to define the total local number of AFs
    int nAF = nAF_inner + nAF_shared + nAF_boundary;

    /*
    if (myid == 0)
        std::cout << "debugging: nAF_inner = " << nAF_inner
              << ", nAF_boundary = " << nAF_boundary
              << ", nAF_shared = " << nAF_shared << "\n";
    */

    // creating AF_face for inner AFs by deleting the entries equal 1
    // in order to get the final AF_face for inner AE faces.
    // The corrected arrays are called ia_new, ja_new and data_new
    // TODO: make it in a smarter way, by writing a special Mult() function
    // because we know that in AFinner_AE there are exactly 2 elements in each row
    // and know how to determine whether the value in the product will be 1 or 2

    int * ia_old = AFinner_AE_face->GetI();
    double * data_old = AFinner_AE_face->GetData();

    // creating ia for AF_face and filling its part related to inner AFs

    // taking part of ia related inner AFs
    int * ia_AF_face = new int[nAF + 1];
    ia_AF_face[0] = 0;
    for ( int i = 0; i < nAF_inner; ++i)
    {
        ia_AF_face[i+1] = ia_AF_face[i];
        for (int j = 0; j < ia_old[i+1] - ia_old[i]; ++j)
        {
            if (data_old[ia_old[i] + j] > 1.5)
                ia_AF_face[i+1]++;
        }
    }

    // augment the inner AF_face with shared & boundary AFs
    // TODO: make it in a smarter way

    // augmenting ia with shared AFs
    int row_shift = nAF_inner;

    int count_augmented_rows = augment_ia_AF_f_shared(ia_AF_face + row_shift,
                                                      AE_AEoffd, cmapAEAE, fc_AEoffd, cmapfcAE, AE_fc );

    // checking that two ways of counting shared AFs give the same number
    PARELAG_ASSERT(count_augmented_rows == nAF_shared);

    // augmenting ia with boundary AFs
    row_shift = nAF_inner + nAF_shared;

    if (facet_bdrAttribute) // if boundary is specified
    {
        int count_augmented_rows = augment_ia_AF_f_boundary(ia_AF_face + row_shift,
                                                            *facet_bdrAttribute, *AE_bnd, AE_fc );
        // checking that two ways of counting boundary AFs give the same number
        PARELAG_ASSERT(count_augmented_rows == nAF_boundary);
    }

    int * ja_AF_face;
    double * data_AF_face;


    if (facet_bdrAttribute)
    {
        ja_AF_face = new int [ia_AF_face[nAF]];
        data_AF_face = new double [ia_AF_face[nAF]];
    }
    else
    {
        ja_AF_face = new int [ia_AF_face[nAF_inner + nAF_shared] + nfc_boundary];
        data_AF_face = new double [ia_AF_face[nAF_inner + nAF_shared] + nfc_boundary];
        augment_ia_ja_data_AF_f_boundary2(ia_AF_face + row_shift, ja_AF_face, data_AF_face,
                                          AEs_withbnd, *fc_AE, fc_AEoffd, AE_fc );

        fc_AE.reset();
    }

    // taking parts of ja and data related to inner AE faces
    augment_ja_data_AF_f_inner(ia_AF_face, ja_AF_face, data_AF_face,
                               nAF_inner, *AFinner_AE_face, *AFinner_AE, AE_fc);

    AFinner_AE.reset();
    AFinner_AE_face.reset();

    // loop over all AE elements, computing ja and data augmentation from the shared AFs
    row_shift = nAF_inner;

    int count_nnz = augment_ja_data_AF_f_shared(ia_AF_face + row_shift, ja_AF_face, data_AF_face,
                              localAEs_startindex, AE_AEoffd, cmapAEAE, fc_AEoffd, cmapfcAE, AE_fc );

    if (nAF_shared > 0)
        PARELAG_ASSERT(count_nnz == ia_AF_face[row_shift + count_augmented_rows] - ia_AF_face[row_shift] );

    delete fc_Truefc_AE;
    delete globalAE_AE;

    if (facet_bdrAttribute) // if boundary is specified
    {
        row_shift = nAF_inner + nAF_shared;
        augment_ja_data_AF_f_boundary(ia_AF_face + row_shift, ja_AF_face, data_AF_face,
                                                    *facet_bdrAttribute, *AE_bnd, AE_fc);
    }
    // if not specified, we have already completed ja and data by calling augment_ia_ja_data_AF_f_boundary2()

    // finally,
    auto AF_face = make_unique<TopologyTable>(ia_AF_face, ja_AF_face, data_AF_face, nAF, nfacets);

    return AF_face;
}

std::shared_ptr<AgglomeratedTopology>
AgglomeratedTopology::CoarsenLocalPartitioning(
    Array<int>& partitioning, bool check_topology,
    bool preserve_material_interfaces, int coarsefaces_algo)
{
    PARELAG_ASSERT(coarsefaces_algo == 0 || coarsefaces_algo == 2);

    using ATC = AgglomeratedTopologyCheck;
    {
        const int * part_data = partitioning.GetData();
        Partition_ = std::vector<int>(part_data,part_data+partitioning.Size());
    }

    // This just views the data as an mfem::Array
    mfem::Array<int> part_tmp(Partition_.data(),Partition_.size());

    auto CoarseTopology = make_shared<AgglomeratedTopology>(Comm_,nCodim_);

    // Set the weak ptr lest we forget
    CoarserTopology_ = CoarseTopology;

    // Link the fine-grid back to us
    CoarseTopology->FinerTopology_ = shared_from_this();
    CoarseTopology->nDim_ = nDim_;

    // Check that all components are connected and material interfaces
    // are treated properly, and remove empty partitions.
    size_t nAE;
    if (preserve_material_interfaces)
    {
        nAE = connectedComponents(
            part_tmp, *LocalElementElementTable(), element_attribute);
    }
    else
    {
        nAE = connectedComponents(part_tmp, *LocalElementElementTable());
    }
    PARELAG_ASSERT(AEntity_entity.size() == 0);

    AEntity_entity.resize(nCodim_+1);
    AEntity_entity[0] = TransposeOrientation(part_tmp, nAE);

    if (check_topology)
    {
        std::stringstream os;
        ATC::ShowBadAgglomeratedEntities(0, *this, os);
        SerializedOutput(Comm_, std::cout, os.str());
        Array<int> isbad;
        bool anybad = ATC::MarkBadAgglomeratedEntities(0, *this, isbad);
        if (anybad)
        {
            DeAgglomerateBadAgglomeratedEntities(isbad,0);
            nAE = AEntity_entity[0]->Height();
        }
    }
    CoarseTopology->entityTrueEntity[0]->SetUp(nAE);

    // creating coarse faces from elements (icodim = 0)
    auto AE_fc = MultOrientation(*(AEntity_entity[0]),*(B_[0]));
    std::unique_ptr<TopologyTable> fc_AF;


    if (coarsefaces_algo == 2)
    {
        // algorithm from Panayots' book
        AEntity_entity[1] = ComputeCoarseFacets(CoarseTopology, *AE_fc);
        fc_AF = AEntity_entity[1]->Transpose();
    }
    else
    {
        // old topology coarsening algorithm using Minimal Intersection Sets
        // may lead to large temporarly memory allocation when boundary labels need to be preserved
        auto fc_AE_fc = ToUnique(Mult(*(AE_fc->Transpose()), *AE_fc));

        // by the way, the old algorithm is working correctly when facet_bdrAttribute is absent
        // it is finding boundary AFs but doesn't adjust them to the bdr attributes
        if (facet_bdrAttribute)
        {
            auto fc_bdrAttr_fc= ToUnique(
                Mult(*facet_bdrAttribute,*(facet_bdrAttribute->Transpose())));
            fc_AE_fc = ToUnique(Add(*fc_AE_fc, *fc_bdrAttr_fc));
        }

        auto Z = AssembleNonLocal(*(entityTrueEntity[1]),
                                  *fc_AE_fc,
                                  *(entityTrueEntity[1]));

        fc_AF = make_unique<TopologyTable>(
            findMinimalIntersectionSets(*Z, .5));

        AEntity_entity[1] = fc_AF->Transpose();
    }

    if (check_topology)
        CheckHFacetsTopology(1, fc_AF);

    CoarseTopology->B_[0] = MultOrientation(*AE_fc, *fc_AF);
    CoarseTopology->entityTrueEntity[1]->SetUp(
        *fc_AF, *(entityTrueEntity[1]) );

    // loop from coarse faces, to edges, to vertices
    for (int icodim = 1; icodim < nCodim_; ++icodim)
    {
        auto AE_fc = MultOrientation(*(AEntity_entity[icodim]),*(B_[icodim]));

        std::unique_ptr<TopologyTable> fc_AF;

        // icodim > 0, thus old topology coarsening algo is used
        auto fc_AE_fc = ToUnique(Mult(*(AE_fc->Transpose()), *AE_fc));

        auto Z = AssembleNonLocal(*(entityTrueEntity[icodim+1]),
                                  *fc_AE_fc,
                                  *(entityTrueEntity[icodim+1]));
        fc_AF = make_unique<TopologyTable>(
            findMinimalIntersectionSets(*Z, .5));

        AEntity_entity[icodim+1] = fc_AF->Transpose();

        if (check_topology)
            CheckHFacetsTopology(icodim+1, fc_AF);

        CoarseTopology->B_[icodim] = MultOrientation(*AE_fc, *fc_AF);
        CoarseTopology->entityTrueEntity[icodim+1]->SetUp(
            *fc_AF, *(entityTrueEntity[icodim+1]) );
    } // end of loop over icodim > 0

    if (facet_bdrAttribute)
        CoarseTopology->facet_bdrAttribute =
            MultOrientation(*(AEntity_entity[1]), *facet_bdrAttribute);

    for (int icodim = 0; icodim < std::min(nCodim_+1,nDim_); ++icodim)
    {
        CoarseTopology->Weights_[icodim]->SetSize(AEntity_entity[icodim]->Size());
        AEntity_entity[icodim]->WedgeMult(
            *(Weights_[icodim]),*(CoarseTopology->Weights_[icodim]));
    }

    this->setCoarseElementAttributes(element_attribute,
                                     CoarseTopology->element_attribute);
    CoarseTopology->BuildConnectivity();


    return CoarseTopology;
}


#if 0
// Input: fec = coarse level Finite-element collection
// Output: R = array of restriction operators -- owned by caller
std::shared_ptr<AgglomeratedTopology>
AgglomeratedTopology::UniformRefinement(
    Array<FiniteElementCollection *> & fecs,
    std::vector<std::unique_ptr<mfem::SparseMatrix>>& Pg)
{

    // TODO: I could check that sizes are right, etc
    PARELAG_ASSERT(pMesh_);
    PARELAG_ASSERT(this == FinestTopology());

    {
        using FES_Ptr = unique_ptr<FiniteElementSpace>;
        using FES_Vector = std::vector<FES_Ptr>;

        const auto nSpaces = fecs.Size();

        FES_Vector fespaces_coarse(nSpaces);
        for (auto ii = 0; ii < nSpaces; ++ii)
            fespaces_coarse[ii] = make_unique<FiniteElementSpace>(pMesh_.get(),
                                                                  fecs[ii]);

        pMesh_->UseTwoLevelState(1);

        pMesh_->UniformRefinement();

        for (auto ii = 0; ii < nSpaces; ++ii)
        {
            pMesh_->SetState(Mesh::TWO_LEVEL_FINE);

            auto fes_fine = make_unique<FiniteElementSpace>(pMesh_.get(),
                                                            fecs[ii]);

            auto R = ToUnique(
                fes_fine->GlobalRestrictionMatrix(fespaces_coarse[ii].get()));
            R->Finalize();

            // Now, if Pg[ii] points to NULL, then I'm on the coarsest
            // mesh and should set Pg[ii]=R^T.
            if (Pg[ii] == nullptr)
            {
                Pg[ii] = ToUnique(Transpose(*R));
            }
            else
            {
                // In this case, the P_G created on the coarser grid is
                // here. I need to multiply this on the LEFT by R^T.
                auto Rt = ToUnique(Transpose(*R));
                R.reset();

                Pg[ii] = ToUnique(Mult(*Rt,*Pg[ii]));
                // Now, Pg[ii] should be the interpolator that
                // interpolates a vector from the coarsest grid to the
                // level that has just been created.
            }

        }

        pMesh_->SetState(Mesh::NORMAL);
    }

    const int nAE = pMesh_->GetNE();

    auto FineTopology = make_shared<AgglomeratedTopology>(pMesh_, nCodim_);

    // Set the weak ptr lest we forget...
    FinerTopology_ = FineTopology;

    FineTopology->CoarserTopology_ = shared_from_this();
    FineTopology->globalAgglomeration = false;
    FineTopology->Partition_.resize(pMesh_->GetNE());

    mfem::Array<int> partitioning(
        FineTopology->Partition_.data(),FineTopology->Partition_.size());

    MFEMRefinedMeshPartitioner partitioner(nDim_);
    partitioner.Partition(pMesh_->GetNE(), nAE, partitioning);

    elag_assert(FineTopology->AEntity_entity.size() == 0);
    FineTopology->AEntity_entity.resize(nCodim_+1);
    FineTopology->AEntity_entity[0] =
        TransposeOrientation(partitioning, nAE);

    switch(nCodim_)
    {
    case 0:
        break;
    case 1:
        FineTopology->AEntity_entity[1] =
            generate_AF_f_ForUniformRefinement();
        break;
    case 2:
        FineTopology->AEntity_entity[1] =
            generate_AF_f_ForUniformRefinement();
        FineTopology->AEntity_entity[2] =
            generate_AR_r_ForUniformRefinement();
        break;
    case 3:
        FineTopology->AEntity_entity[1] =
            generate_AF_f_ForUniformRefinement();
        FineTopology->AEntity_entity[2] =
            generate_AR_r_ForUniformRefinement();
        FineTopology->AEntity_entity[3] =
            generate_AP_p_ForUniformRefinement();
        break;
    default:
        PARELAG_TEST_FOR_EXCEPTION( true,
                                    std::invalid_argument,
                                    "Invalid nCodim_ = " << nCodim_ );
    }

    pMesh_.reset();
    return FineTopology;
}
#endif

std::shared_ptr<AgglomeratedTopology>
AgglomeratedTopology::UniformRefinement()
{
    elag_assert(pMesh_);
    elag_assert(this == FinestTopology());

    const int nAE = pMesh_->GetNE();
    pMesh_->UniformRefinement();

    auto FineTopology = make_shared<AgglomeratedTopology>(pMesh_, nCodim_);
    FinerTopology_ = FineTopology;
    FineTopology->CoarserTopology_ = shared_from_this();
    FineTopology->globalAgglomeration = false;
    FineTopology->Partition_.resize(pMesh_->GetNE());

    mfem::Array<int> partitioning(
        FineTopology->Partition_.data(),FineTopology->Partition_.size());
    MFEMRefinedMeshPartitioner partitioner(nDim_);
    partitioner.Partition(pMesh_->GetNE(),nAE,partitioning);

    elag_assert(FineTopology->AEntity_entity.size() == 0);
    FineTopology->AEntity_entity.resize(nCodim_+1);

    FineTopology->AEntity_entity[0] =
        TransposeOrientation(partitioning, nAE);

    switch(nCodim_)
    {
    case 0:
        break;
    case 1:
        FineTopology->AEntity_entity[1] =
            generate_AF_f_ForUniformRefinement();
        break;
    case 2:
        FineTopology->AEntity_entity[1] =
            generate_AF_f_ForUniformRefinement();
        FineTopology->AEntity_entity[2] =
            generate_AR_r_ForUniformRefinement();
        break;
    case 3:
        FineTopology->AEntity_entity[1] =
            generate_AF_f_ForUniformRefinement();
        FineTopology->AEntity_entity[2] =
            generate_AR_r_ForUniformRefinement();
        FineTopology->AEntity_entity[3] =
            generate_AP_p_ForUniformRefinement();
        break;
    default:
        elag_error(1);
    }

    pMesh_.reset();
    return FineTopology;
}

unique_ptr<TopologyTable> FindIntersectionsAF(const mfem::SparseMatrix& Z)
{
    const int nAF = Z.Size();
    const int nf = Z.Width();

    const int * const i_Z = Z.GetI();
    const int * const j_Z = Z.GetJ();
    const double * const a_Z = Z.GetData();

    int nnz = 0;
    int * i_AF_f = new int[nAF+1];

    constexpr double tol = 1e-9;
    const double * it = a_Z, *end;
    for(int i(0); i < nAF; ++i)
    {
        i_AF_f[i] = nnz;
        for(end = a_Z + i_Z[i+1]; it != end; ++it)
        {
            elag_assert(fabs(*it) < 2.+tol);
            if(fabs(*it) > 2. - tol)
                ++nnz;
        }
    }
    i_AF_f[nAF] = nnz;

    int * j_AF_f = new int[nnz];
    double * o_AF_f = new double[nnz];
    it = a_Z;
    const int * j_it = j_Z;
    nnz = 0;
    for(int i(0); i < nAF; ++i)
    {
        for(end = a_Z + i_Z[i+1]; it != end; ++it, ++j_it)
        {
            elag_assert(fabs(*it) < 2.+tol);
            if( fabs(*it) > 2. - tol )
            {
                j_AF_f[nnz] = *j_it;
                o_AF_f[nnz] = (*it > 0.) ? 1.0:-1.0;
                ++nnz;
            }
        }
    }

    return make_unique<TopologyTable>(i_AF_f,j_AF_f,o_AF_f,nAF,nf);
}

unique_ptr<TopologyTable>
AgglomeratedTopology::generate_AF_f_ForUniformRefinement()
{
    const TopologyTable & AF_bdr = *facet_bdrAttribute;

    auto finerTopology = FinerTopology_.lock();

    auto AE_f = MultOrientation(*(finerTopology->AEntity_entity[0]),
                                *(finerTopology->B_[0]));
    auto bdr_f = finerTopology->FacetBdrAttribute().Transpose();

    auto Z1 = ToUnique(Mult(*(B_[0]->Transpose()), *AE_f));
    auto Z2 = ToUnique(Mult(AF_bdr, *bdr_f));
    auto Z3 = AssembleNonLocal(*(entityTrueEntity[FACET]),*Z1,
                               *(finerTopology->entityTrueEntity[FACET]));

    auto Z = ToUnique(Add(*Z2, *Z3));

    return FindIntersectionsAF(*Z);
}

unique_ptr<TopologyTable>
FindIntersectionsAR(
    const mfem::SparseMatrix& AR_AF,const mfem::SparseMatrix& Z)
{
    const int nAR = Z.Size();
    const int nr = Z.Width();

    const int * const i_Z = Z.GetI();
    const int * const j_Z = Z.GetJ();
    const double * const a_Z = Z.GetData();

    const int * const i_AR_AF = AR_AF.GetI();

    int nnz = 0;
    int * i_AR_r = new int[nAR+1];

    constexpr double tol = 1e-9;
    const double * it = a_Z, *end;
    for(int i(0); i < nAR; ++i)
    {
        i_AR_r[i] = nnz;
        const int nAdiacentFaces = i_AR_AF[i+1] - i_AR_AF[i];
        for(end = a_Z + i_Z[i+1]; it != end; ++it)
        {
            elag_assert(fabs(*it) < static_cast<double>(nAdiacentFaces)+tol);
            if(fabs(*it) > static_cast<double>(nAdiacentFaces) - tol)
                ++nnz;
        }
    }
    i_AR_r[nAR] = nnz;

    int * j_AR_r = new int[nnz];
    double * o_AR_r = new double[nnz];
    it = a_Z;
    const int * j_it = j_Z;
    nnz = 0;
    for(int i(0); i < nAR; ++i)
    {
        const int nAdiacentFaces = i_AR_AF[i+1] - i_AR_AF[i];
        for(end = a_Z + i_Z[i+1]; it != end; ++it, ++j_it)
        {
            elag_assert(fabs(*it) < static_cast<double>(nAdiacentFaces)+tol);
            if(fabs(*it) > static_cast<double>(nAdiacentFaces) - tol)
            {
                j_AR_r[nnz] = *j_it;
                o_AR_r[nnz] = (*it > 0.) ? 1.0:-1.0;
                ++nnz;
            }
        }
    }

    return make_unique<TopologyTable>(i_AR_r,j_AR_r,o_AR_r,nAR,nr);
}

unique_ptr<TopologyTable>
AgglomeratedTopology::generate_AR_r_ForUniformRefinement()
{
    auto finerTopology = FinerTopology_.lock();
    auto AR_AF = B_[1]->Transpose();
    auto AF_r = MultOrientation(*(finerTopology->AEntity_entity[1]),
                                *(finerTopology->B_[1]));
    auto  Z = ToUnique(Mult(*AR_AF, *AF_r));
    return FindIntersectionsAR(*AR_AF, *Z);
}

unique_ptr<TopologyTable>
AgglomeratedTopology::generate_AP_p_ForUniformRefinement()
{
    auto finerTopology = FinerTopology_.lock();
    auto AR_AF = B_[2]->Transpose();
    auto AF_r = MultOrientation(*(finerTopology->AEntity_entity[2]),
                                *(finerTopology->B_[2]));
    auto Z = ToUnique(Mult(*AR_AF, *AF_r));
    return FindIntersectionsAR(*AR_AF, *Z);
}


void AgglomeratedTopology::DeAgglomerateBadAgglomeratedEntities(
    Array<int> & isbad, int icodim)
{
    const int numBadAE = isbad.Sum();
    if (numBadAE == 0) return;

    const TopologyTable * AEE = AEntity_entity[icodim].get();
    const int * AEE_I = AEE->GetI();
    const int * AEE_J = AEE->GetJ();
    const double * AEE_data = AEE->GetData();

    const int nAE = AEE->Height(); // Original number of agglomerated entities

    // Find number of new agglomerated entities
    int new_nAE = 0;
    for (int iAE = 0; iAE < nAE; ++iAE)
    {
        if (isbad[iAE])
            new_nAE += AEE_I[iAE+1] - AEE_I[iAE];
        else
            new_nAE += 1;
    }

    // Create array for new I indices
    int * new_I = new int[new_nAE+1];
    new_I[new_nAE] = AEE_I[AEE->Height()];

    int row_counter(0);
    for (int iAE(0); iAE < nAE; ++iAE)
    {
        if (isbad[iAE] == 1)
        {
            for (int row(AEE_I[iAE]); row < AEE_I[iAE+1]; ++row)
            {
                new_I[row_counter] = row;
                row_counter++;
            }
        }
        else
        {
            new_I[row_counter] = AEE_I[iAE];
            row_counter++;
        }
    }

    // Copy J and data from AEE
    int * new_J = new int[AEE->NumNonZeroElems()];
    std::copy(AEE_J,AEE_J+AEE->NumNonZeroElems(),new_J);
    double * new_data = new double[AEE->NumNonZeroElems()];
    std::copy(AEE_data,AEE_data+AEE->NumNonZeroElems(),new_data);

    // Replace AEE with new TopologyTable
    auto new_AEE = make_unique<TopologyTable>(
        new_I,new_J,new_data,new_nAE,AEE->Width());

    std::cout << "Correcting agglomerated topology for icodim: " << icodim
              << std::endl << "  original number of agglomerates: "
              << AEE->Height() << std::endl << "  number which were bad: "
              << numBadAE << std::endl
              << "  number of new agglomerates after de-agglomeration: "
              << new_AEE->Height() - AEE->Height() << std::endl;

    AEntity_entity[icodim] = std::move(new_AEE);
}

void AgglomeratedTopology::setCoarseElementAttributes(
    const Array<int> & fine_attr, Array<int> & coarse_attr)
{
    TopologyTable & AE_el = *(this->AEntity_entity[ELEMENT]);

    const int nAE = AE_el.Height();
    const int ne  = AE_el.Width();

    const int * i_AE_el = AE_el.GetI();
    const int * j_AE_el = AE_el.GetJ();

    PARELAG_ASSERT(fine_attr.Size() == ne);
    coarse_attr.SetSize(nAE);

    for(int iAE = 0; iAE < nAE; ++iAE)
    {
        const int start = i_AE_el[iAE];
#ifdef PARELAG_ASSERTING
        const int end = i_AE_el[iAE+1];
        const int nnz = end - start;
#endif
        elag_assert(nnz); // do not allow empty agglomerates;
        int current_attr = fine_attr[j_AE_el[start]];
        // ATB 17 August 2015: in general we want to allow multiple attributes in a single AE.
        //                     the logical partitioner enforces one attribute only per AE,
        //                     but we do not always use logical partitioner
        //                     if we want to enforce this for some specific cases, it should be optional
        /*
          for(int i = 0; i < nnz; ++i)
          elag_assert(fine_attr[ j_AE_el[ start+i ] ] == current_attr);
        */
        coarse_attr[iAE] = current_attr;
    }
}

// Should be extended to store for facet_element(:,2) -1 if boundary and -2 if processor boundary
void AgglomeratedTopology::buildFacetElementArray()
{
    if(facet_element.NumCols() > 0)
        return;

    auto face_el = B_[0]->Transpose();
    Array<int> elements;
    Vector row;
    facet_element.SetSize(face_el->Height(),2);
    int el1,el2;
    for(int iAF(0); iAF < face_el->Height(); ++iAF)
    {
        face_el->GetRow(iAF,elements,row);
        PARELAG_ASSERT(elements.Size() > 0 && elements.Size() <= 2);
        if(elements.Size() == 1)
        {
            el1 = elements[0];
            el2 = -1;
            PARELAG_ASSERT(row(0) > 0.0);
        }
        else
        {
            el1 = elements[0];
            el2 = elements[1];
            if(row(0) < 0.0)
                mfem::Swap(el1,el2);
            PARELAG_ASSERT(row(0)*row(1) < 0.0);
        }
        facet_element(iAF,0) = el1;
        facet_element(iAF,1) = el2;
    }
}

void AgglomeratedTopology::GetFacetNeighbors(int iAF, int & el1, int & el2)
{
    if (facet_element.NumCols() == 0)
        this->buildFacetElementArray();

    el1 = facet_element(iAF,0);
    el2 = facet_element(iAF,1);
}
  
// Used in ComputeCoarseFacets only. Fills ia array for shared AFs for future AF_face table
int augment_ia_AF_f_shared(int * ia_AF_face_startshd, SparseMatrix& AE_AEoffd, int * cmapAEAE,
                            SparseMatrix& fc_AEoffd, int * cmapfcAE, TopologyTable& AE_fc )
{
    int * offd_i = AE_AEoffd.GetI();
    int * offd_j = AE_AEoffd.GetJ();

    int * ia_fc_AEoffd = fc_AEoffd.GetI();
    int * ja_fc_AEoffd = fc_AEoffd.GetJ();
    int count_augmented_rows = 0;
    // loop over all AE elements, computing ia augmentation from the shared AFs
    for (int localAE = 0; localAE < AE_AEoffd.Height(); ++localAE)
    {
        int offd_ncols = offd_i[localAE+1] - offd_i[localAE];
        if (offd_ncols > 0) // if this local AE is connected to some AEs from the other process aka nonlocal AEs
        {

            for ( int col = 0; col < offd_ncols; ++col ) // loop over all nonlocal AES
            {
                int nonlocalAE = cmapAEAE[offd_j[offd_i[localAE] + col]]; // taking care of the off-diagonal compression in hypre

                int nAEfacets = AE_fc.RowSize(localAE);
                int * localAEfacets = AE_fc.GetRowColumns(localAE);

                // loop over all facets of given local AE, looking for the same special attribute
                // TODO: loops several times over AE facets, can be avoided by using additional array of size nAEfacets
                int temp_nfacets = 0; // number of facets in the intersection of local and nonlocal AEs
                for (int facetno = 0; facetno < nAEfacets; ++facetno)
                {
                    int facet = localAEfacets[facetno];
                    if (ia_fc_AEoffd[facet+1] - ia_fc_AEoffd[facet] > 0) // if the given facet is shared
                    {
                        int facet_specialattribute = cmapfcAE[ja_fc_AEoffd[ia_fc_AEoffd[facet]]];
                        if (facet_specialattribute == nonlocalAE) // if it is shared with the given nonlocal AE
                        {
                            temp_nfacets++;
                        }
                    }
                }

                // now we know, how many facets will belong to the considered shared AF
                ia_AF_face_startshd[count_augmented_rows + 1] = ia_AF_face_startshd[count_augmented_rows] + temp_nfacets;
                count_augmented_rows++;
            }
        }
    }

    return count_augmented_rows;
}

// Used in ComputeCoarseFacets only. Fills ia array for boundary AFs for future AF_face table
int augment_ia_AF_f_boundary(int * ia_AF_face_startbnd, TopologyTable& facet_bdrAttribute,
                                                  SparseMatrix& AE_bnd, TopologyTable& AE_fc )
{

    int * ia_facet_bdrAttr = facet_bdrAttribute.GetI();
    int * ja_facet_bdrAttr = facet_bdrAttribute.GetJ();

    int count_augmented_rows = 0;
    // loop over all AE elements, computing ia augmentation from the boundary AFs
    // a single AE can have facets belonging to different boundary parts, thus
    // generating several boundary AFs
    for (int row = 0; row < AE_bnd.Height(); ++row)
    {
        int ncols = AE_bnd.RowSize(row);
        if (ncols > 0) // choosing AEs which have smth on the boundary
        {
            int nAEfacets = AE_fc.RowSize(row);
            int * AEfacets = AE_fc.GetRowColumns(row);
            int * bdrattribs = AE_bnd.GetRowColumns(row);
            // loop over all boundary attributes where our AE has something
            for ( int colno = 0; colno < ncols; ++colno)
            {
                int bdrattrib = bdrattribs[colno];
                int nfacets_perattrib = 0;
                // looping over all facets which belong to given AE
                for ( int facetno = 0; facetno < nAEfacets; ++facetno )
                {
                    int facet = AEfacets[facetno];
                    // checking if one of the facets inside given AE is a boundary facet with given bdr attribute
                    if ( ia_facet_bdrAttr[facet + 1] - ia_facet_bdrAttr[facet] == 1 &&
                         ja_facet_bdrAttr[ia_facet_bdrAttr[facet]] == bdrattrib)
                    {
                        nfacets_perattrib++;
                    }
                }

                // must be always true, because we have a nonzero from the row of AE_bnd
                PARELAG_ASSERT(nfacets_perattrib > 0);

                ia_AF_face_startbnd[count_augmented_rows + 1] = ia_AF_face_startbnd[count_augmented_rows] + nfacets_perattrib;
                count_augmented_rows++;
            }
        } // end of if ncols > 0
    } // end of loop over AE elements

    return count_augmented_rows;
}

// Used in ComputeCoarseFacets only. Fills ia, ja and data array for boundary AFs for future AF_face table
// when no facet_bdrAttribute is available
int augment_ia_ja_data_AF_f_boundary2(int * ia_AF_face_startbnd, int * ja_AF_face, double * data_AF_face,
                std::map<int,int>& AEs_withbnd, TopologyTable& fc_AE, SparseMatrix& fc_AEoffd, TopologyTable& AE_fc )
{
    // loop over all AEs which have boundary AFs
    int bndAF_count = 0;
    for (const auto& pair : AEs_withbnd)
    {
        int AEind = pair.first;
        int nfacets_to_find = pair.second;

        ia_AF_face_startbnd[bndAF_count + 1] = ia_AF_face_startbnd[bndAF_count] + nfacets_to_find;

        int * AEfacets = AE_fc.GetRowColumns(AEind);
        int nAEfacets = AE_fc.RowSize(AEind);

        // loop over all facets of given AE, picking out only boundary facets
        int nfacets_found = 0;
        for ( int facetno = 0; facetno < nAEfacets; ++facetno)
        {
            int fc = AEfacets[facetno];

            // if facet belongs to exactly 1 local AE and 0 nonlocal AEs
            if ( fc_AE.RowSize(fc) == 1 && fc_AEoffd.RowSize(fc) == 0 )
            {
                ja_AF_face[ia_AF_face_startbnd[bndAF_count] + nfacets_found] = fc;
                data_AF_face[ia_AF_face_startbnd[bndAF_count] + nfacets_found] = 1.0; //what is the correct sign for bdr facets?
                nfacets_found++;
            }
        }
        PARELAG_ASSERT(nfacets_found == nfacets_to_find);
        bndAF_count++;
    }

    return AEs_withbnd.size();
}


// Used in ComputeCoarseFacets only. Fills ja and data array for shared AFs
// in the future AF_face table. Assumes that ia array is already filled
int augment_ja_data_AF_f_shared(int * ia_AF_face_startshd, int * ja_AF_face, double * data_AF_face,
                                int localAEs_startindex, SparseMatrix& AE_AEoffd, int * cmapAEAE,
                                SparseMatrix& fc_AEoffd, int * cmapfcAE, TopologyTable& AE_fc )
{
    int * offd_i = AE_AEoffd.GetI();
    int * offd_j = AE_AEoffd.GetJ();

    int * ia_fc_AEoffd = fc_AEoffd.GetI();
    int * ja_fc_AEoffd = fc_AEoffd.GetJ();

    int * ia_AE_fc = AE_fc.GetI();
    double * data_AE_fc = AE_fc.GetData();

    int count_nnz = 0;
    int count_augmented_rows = 0;

    for (int localAE = 0; localAE < AE_AEoffd.Height(); ++localAE)
    {
        int offd_ncols = offd_i[localAE+1] - offd_i[localAE];
        if (offd_ncols > 0) // if this local AE is connected to some AEs from the other process aka nonlocal AEs
        {
            for ( int col = 0; col < offd_ncols; ++col ) // loop over all nonlocalAES
            {

                int nonlocalAE = cmapAEAE[offd_j[offd_i[localAE] + col]]; // taking care of off-diagonal compression in hypre
                bool flip_orientation = ( nonlocalAE < localAEs_startindex ? true : false); // if nonlocal AE belongs to the proc with lower id in the comm

                int nAEfacets = AE_fc.RowSize(localAE);
                int * localAEfacets = AE_fc.GetRowColumns(localAE);

                // loop over all facets of given local AE, looking for the same special attribute
                // FIXME: loops several times over AE facets, can be avoided by using additional array of size nAEfacets
                int temp_nfacets = 0;
                for (int facetno = 0; facetno < nAEfacets; ++facetno)
                {
                    int facet = localAEfacets[facetno];
                    if (ia_fc_AEoffd[facet+1] - ia_fc_AEoffd[facet] > 0) // if the given facet is shared
                    {
                        int facet_specialattribute = cmapfcAE[ja_fc_AEoffd[ia_fc_AEoffd[facet]]];
                        if (facet_specialattribute == nonlocalAE) // if it is shared with the given nonlocal AE
                        {
                            ja_AF_face[ia_AF_face_startshd[count_augmented_rows] + temp_nfacets] = facet;

                            // taking facet orientation from local AE_fc
                            data_AF_face[ia_AF_face_startshd[count_augmented_rows] + temp_nfacets] = data_AE_fc[ia_AE_fc[localAE] + facetno];

                            // flipping irentation if needed
                            if (flip_orientation)
                                data_AF_face[ia_AF_face_startshd[count_augmented_rows] + temp_nfacets] *= -1;

                            temp_nfacets++;

                            count_nnz++;
                        }
                    }
                }

                //ia_AF_face[row_shift + count_augmented_rows + 1] = ia_AF_face[row_shift + count_augmented_rows] + temp_nfacets;
                count_augmented_rows++;
            }
        }
    }

    return count_nnz;
}

// Used in ComputeCoarseFacets only. Fills ja and data array for boundary AFs in the
// future AF_face table. Assumes that ia array is already filled
int augment_ja_data_AF_f_boundary(int * ia_AF_face_startbnd, int * ja_AF_face, double * data_AF_face,
                                  TopologyTable& facet_bdrAttribute, SparseMatrix& AE_bnd, TopologyTable& AE_fc )
{

    int * ia_facet_bdrAttr = facet_bdrAttribute.GetI();
    int * ja_facet_bdrAttr = facet_bdrAttribute.GetJ();

    // augmenting ja and data with boundary related stuff
    int count_augmented_rows = 0;
    // loop over all AE elements, again, but now for ja and aa, boundary AFs
    for (int row = 0; row < AE_bnd.Height(); ++row)
    {
        int ncols = AE_bnd.RowSize(row);
        if (ncols > 0) // choosing AEs which have smth on the boundary
        {
            int nAEfacets = AE_fc.RowSize(row);
            int * AEfacets = AE_fc.GetRowColumns(row);
            int * bdrattribs = AE_bnd.GetRowColumns(row);
            // loop over all boundary attributes where our AE has something
            for ( int colno = 0; colno < ncols; ++colno)
            {
                int bdrattrib = bdrattribs[colno];
                int nfacets_perattrib = 0;
                for ( int facetno = 0; facetno < nAEfacets; ++facetno )
                {
                    int facet = AEfacets[facetno];
                    // checking if one of the facets inside given AE is a boundary facet with given bdr attribute
                    if ( ia_facet_bdrAttr[facet + 1] - ia_facet_bdrAttr[facet] == 1 &&
                         ja_facet_bdrAttr[ia_facet_bdrAttr[facet]] == bdrattrib)
                    {
                        ja_AF_face[ia_AF_face_startbnd[count_augmented_rows] + nfacets_perattrib] = facet;
                        data_AF_face[ia_AF_face_startbnd[count_augmented_rows] + nfacets_perattrib] = 1.0;
                        nfacets_perattrib++;
                    }
                }

                // must be always true, because we have a nonzero from the row of AE_bnd
                PARELAG_ASSERT(nfacets_perattrib > 0);

                //ia_AF_face[row_shift + count_augmented_rows + 1] = ia_AF_face[row_shift + count_augmented_rows] + nfaces_perattrib;
                count_augmented_rows++;
            } // end of loop over boundary attributes where the AE has something
        } // end of if ncols > 0
    } // end of the loop over AE elements

    return 0;
}

// Used in ComputeCoarseFacets only. Fills ja and data array for inner AFs in the
// future AF_face table. Assumes that ia array is already filled
int augment_ja_data_AF_f_inner(int * ia_AF_face, int * ja_AF_face, double * data_AF_face,
            int nAF_inner, SparseMatrix& AFinner_AE_face, SparseMatrix& AFinner_AE, TopologyTable& AE_fc)
{
    int * ia_AE_fc = AE_fc.GetI();
    int * ja_AE_fc = AE_fc.GetJ();
    double * data_AE_fc = AE_fc.GetData();

    int * ia_old = AFinner_AE_face.GetI();
    int * ja_old = AFinner_AE_face.GetJ();
    double * data_old = AFinner_AE_face.GetData();

    int * ia_AFinner_AE = AFinner_AE.GetI();
    int * ja_AFinner_AE = AFinner_AE.GetJ();
    double * data_AFinner_AE = AFinner_AE.GetData();

    for ( int i = 0; i < nAF_inner; ++i)
    {
        int count = 0;
        for (int j = 0; j < ia_old[i+1] - ia_old[i]; ++j)
        {
            if (data_old[ia_old[i] + j] > 1.0 + 0.5)
            {
                ja_AF_face[ia_AF_face[i] + count] = ja_old[ia_old[i] + j];

                int iAE = ja_AFinner_AE[ia_AFinner_AE[i] + 0];

                PARELAG_ASSERT(data_AFinner_AE[ia_AFinner_AE[i] + 0] > 0.5); // must be 1.0 actually

                int facet_to_find = ja_AF_face[ia_AF_face[i] + count];
                int nnzrowstart = ia_AE_fc[iAE];
                int nnzrowend = ia_AE_fc[iAE + 1];
                // replace by while maybe?
                int ifacet_in_row = 0;
                for (int ifacet = 0; ifacet < nnzrowend - nnzrowstart; ++ifacet)
                {
                    if ( ja_AE_fc[nnzrowstart + ifacet] == facet_to_find )
                    {
                        ifacet_in_row = ifacet;
                        break;
                    }
                }

                data_AF_face[ia_AF_face[i] + count] = data_AE_fc[nnzrowstart + ifacet_in_row];
                count++;
            }

        }
    }

    return 0;
}

}//namespace parelag
