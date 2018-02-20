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

#include "AgglomeratedTopologyCheck.hpp"

#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "linalg/utilities/ParELAG_SubMatrixExtraction.hpp"
#include "utilities/mpiUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

void AgglomeratedTopologyCheck::additionalTopologyCheck(
    int codim, AgglomeratedTopology & topo, Array<int> & isbad, bool verbose,
    std::ostream& os)
{
    PARELAG_ASSERT(codim == 0 || codim == 1);

    // Orientation cancels out non agglomerate boundary faces, so abs everything
    unique_ptr<SparseMatrix> _AElement_bface{
        Mult(topo.AEntityEntity(codim),topo.GetB(codim))};
    DropZerosFromSparseMatrix dropzeros;
    // AElement_bface in codim 0, AFace_bedge in codim 1
    auto AElement_bface = DropEntriesFromSparseMatrix(*_AElement_bface,
                                                      dropzeros);
    _AElement_bface.reset();
    double * AElement_bface_data = AElement_bface->GetData();
    for (int i(0); i < AElement_bface->NumNonZeroElems(); ++i)
        AElement_bface_data[i] = std::fabs(AElement_bface_data[i]);
    TopologyTable & face_edge = topo.GetB(codim+1); // global fine connectivity
    auto abs_face_edge = DeepCopy(face_edge);
    double* data = abs_face_edge->GetData();
    for(int i(0); i < abs_face_edge->NumNonZeroElems(); ++i)
        data[i] = std::fabs(data[i]);

    // AElement_bedge in codim 0, AFace_bvertex in codim 1
    unique_ptr<SparseMatrix> AElement_bedge{
        Mult(*AElement_bface,*abs_face_edge)};

    int * i_AElement_bface = AElement_bface->GetI();
    int * j_AElement_bface = AElement_bface->GetJ();
    int * i_AElement_bedge = AElement_bedge->GetI();
    int * j_AElement_bedge = AElement_bedge->GetJ();
    for (int iAE(0); iAE < AElement_bface->Height(); ++iAE)
    {
        Array<int> rows(&(j_AElement_bface[i_AElement_bface[iAE]]),
                        i_AElement_bface[iAE+1]-i_AElement_bface[iAE]);
        Array<int> cols(&(j_AElement_bedge[i_AElement_bedge[iAE]]),
                        i_AElement_bedge[iAE+1]-i_AElement_bedge[iAE]);
        // local_bface_bedge in codim 0, local_bedge_bvertex in codim 1
        auto local_bface_bedge = ExtractRowAndColumns(*abs_face_edge,rows,cols);

        Vector ones(local_bface_bedge->Height());
        ones = 1.0;
        Vector twos(local_bface_bedge->Width());

        local_bface_bedge->MultTranspose(ones, twos);
        if (std::fabs(twos.Sum() - 2*twos.Size()) > 1e-10)
        {
            if (verbose)
            {
                os << "    codim " << codim << " iAE " << iAE
                   << " has bad connectivity (eg boundary edge adjacent to >2 "
                   << "boundary faces)." << std::endl;
                // twos.Print(os);
            }
            isbad[iAE] = 1;
        }
    }
}

bool AgglomeratedTopologyCheck::MarkBadAgglomeratedEntities(
    int codim, AgglomeratedTopology & topo, Array<int> & isbad)
{
    PARELAG_ASSERT(codim <= topo.Codimensions());

    DenseMatrix bettiNumbers;
    computeBettiNumbersAgglomeratedEntities(codim, topo, bettiNumbers);
    isbad.SetSize(bettiNumbers.Height());
    int nDim;
    for (int i(0); i < isbad.Size(); ++i)
        isbad[i] = 0;
    switch(codim)
    {
    case 0:
        nDim = topo.Dimensions();
        for (int iAE = 0; iAE < bettiNumbers.Height(); ++iAE)
        {
            if (bettiNumbers(iAE,0) != 1)
                isbad[iAE] = 1; // disconnected

            for (int i(1); i < nDim; ++i)
                if (bettiNumbers(iAE,i) != 0)
                {
                    if (nDim - 1 == i)
                        isbad[iAE] = 1; // hole
                    else
                        isbad[iAE] = 1; // tunnel
                }
        }
        break;
    case 1:
        nDim = bettiNumbers.Width();
        for(int iAE = 0; iAE < bettiNumbers.Height(); ++iAE)
        {
            if (bettiNumbers(iAE,0) != 1)
                isbad[iAE] = 1; // disconnected

            for (int i(1); i < nDim; ++i)
                if (bettiNumbers(iAE,i) != 0 )
                    isbad[iAE] = 1; // hole
        }
        break;
    case 2:
        for (int iAE = 0; iAE < bettiNumbers.Height(); ++iAE)
        {
            if (bettiNumbers(iAE,0) != 1)
                isbad[iAE] = 1; // disconnected
        }
        break;
    default:
        break;
    }

    // Additional topological checks
    if (topo.Dimensions() == 2 && codim == 0)
        additionalTopologyCheck(codim, topo, isbad);
    else if (topo.Dimensions() == 3 && (codim == 0 || codim == 1))
        additionalTopologyCheck(codim, topo, isbad);

    if (isbad.Sum() > 0)
        return true;
    return false;
}

void AgglomeratedTopologyCheck::ShowBadAgglomeratedEntities(
    int codim, AgglomeratedTopology & topo, std::ostream & os)
{
    elag_assert( codim <= topo.Codimensions() );

    DenseMatrix bettiNumbers;
    computeBettiNumbersAgglomeratedEntities(codim, topo, bettiNumbers);
    switch (codim)
    {
    case 0:
        showBadAgglomeratedElements(bettiNumbers, topo, os);
        break;
    case 1:
        showBadAgglomeratedFacets(bettiNumbers, topo, os);
        break;
    case 2:
        showBadAgglomeratedRidges(bettiNumbers, topo, os);
        break;
    default:
        break;
    }

    // the following may be too expensive
    Array<int> isbad_dummy;
    isbad_dummy.SetSize(bettiNumbers.Height());
    std::stringstream oss;
    if ((topo.Dimensions() == 2 && codim == 0) ||
        (topo.Dimensions() == 3 && (codim == 0 || codim == 1)))
    {
        additionalTopologyCheck(codim, topo, isbad_dummy, true, oss);
    }

    SerializedOutput(MPI_COMM_WORLD, std::cout, oss.str());
}

void AgglomeratedTopologyCheck::showBadAgglomeratedElements(
    DenseMatrix & bettiNumbers, AgglomeratedTopology & topo, std::ostream & os)
{
    int nDim = topo.Dimensions();

    for (int iAE = 0; iAE < bettiNumbers.Height(); ++iAE)
    {
        if (bettiNumbers(iAE,0) != 1)
            os << "    Element " << iAE << " is disconnected. "
               << "The number of connected components is "
               << bettiNumbers(iAE,0) << std::endl;

        for (int i(1); i < nDim; ++i)
        {
            if (bettiNumbers(iAE,i) != 0 )
            {
                if (nDim - 1 == i )
                    os << "    Element " << iAE << " has "
                       << bettiNumbers(iAE,i) << " holes." << std::endl;
                else
                    os << "    Element " << iAE << " has "
                       << bettiNumbers(iAE,i) << " tunnels." << std::endl;
            }
        }
    }
}

void AgglomeratedTopologyCheck::showBadAgglomeratedFacets(
    DenseMatrix & bettiNumbers, AgglomeratedTopology &, std::ostream & os)
{
    int nDim = bettiNumbers.Width();

    for(int iAE = 0; iAE < bettiNumbers.Height(); ++iAE)
    {
        if (bettiNumbers(iAE,0) != 1)
            os << "    Facet " << iAE << " is disconnected. "
               << "The number of connected components is "
               << bettiNumbers(iAE,0) << std::endl;

        for (int i(1); i < nDim; ++i)
        {
            if (bettiNumbers(iAE,i) != 0 )
                os << "    Facet " << iAE << " has "
                   << bettiNumbers(iAE,i) << " holes." << std::endl;
        }
    }
}

void AgglomeratedTopologyCheck::showBadAgglomeratedRidges(
    DenseMatrix & bettiNumbers, AgglomeratedTopology &, std::ostream & os)
{
    for (int iAE = 0; iAE < bettiNumbers.Height(); ++iAE)
    {
        if (bettiNumbers(iAE,0) != 1)
            os << "    Ridge " << iAE << " is disconnected. "
               << "The number of connected components is "
               << bettiNumbers(iAE,0) << std::endl;
    }
}

void AgglomeratedTopologyCheck::computeBettiNumbersAgglomeratedEntities(
    int codim, AgglomeratedTopology & topo, DenseMatrix & bettiNumbers)
{
    int nLowerDims = topo.Dimensions() - codim;
    if (nLowerDims == 0)
    {
        bettiNumbers.SetSize(0);
        return;
    }

    // FIXME (trb 12/09/15): I haven't thought all the way through
    // this yet, but I'd love to get rid of this Array<TopologyTable
    // *> in favor of something else -- perhaps even no array at
    // all. If we could simply pull the things we need when we need
    // them during the loop, I would be much happier. Happier still of
    // the [0] case could be separated out so the reference to a stack
    // variable could be avoided.
    Array<TopologyTable *> AE_entity(nLowerDims+1);
    AE_entity[0] = &topo.AEntityEntity(codim);
    for (int i(0); i < nLowerDims; ++i)
    {
        AE_entity[i+1] = MultBoolean(*(AE_entity[i]),
                                     topo.GetB(codim+i)).release();
    }
    int nAE = AE_entity[0]->Size();
    Array< Array<int> * > entity_in_AE(nLowerDims+1);

    for(int i(0); i < nLowerDims+1; ++i)
        entity_in_AE[i] = new Array<int>;

    Vector dummy; // don't need values, just i,j entries
    Array<int> dim_k(nLowerDims+1), rank_k(nLowerDims+1);
    rank_k[nLowerDims] = 0;
    bettiNumbers.SetSize(nAE,nLowerDims);
    for (int iAE(0); iAE < nAE; ++iAE)
    {
        for (int i(0); i < nLowerDims+1; ++i)
        {
            AE_entity[i]->GetRow(iAE, *(entity_in_AE[i]), dummy);
            dim_k[i] = entity_in_AE[i]->Size();
        }

        for (int i(0); i < nLowerDims; ++i)
        {
            if (dim_k[i] == 0 || dim_k[i+1] == 0)
                rank_k[i] = 0;
            else
            {
                DenseMatrix dloc(dim_k[i], dim_k[i+1]);
                topo.GetB(codim+i).GetSubMatrix(*(entity_in_AE[i]),
                                                *(entity_in_AE[i+1]),
                                                dloc);
                rank_k[i] = dloc.Rank(1.e-9);
#ifdef ELAG_DEBUG
                // should check ratios of smallest singular values?
                int check_rank = dloc.Rank(1.e-7);
                elag_assert(rank_k[i] == check_rank);
#endif
            }
        }

        int l_codim;
        for(int i(0); i < nLowerDims; ++i)
        {
            l_codim = nLowerDims - i - 1;
            // dim(kernel) = (dim_k[i+1] - rank_k[k]), dim(ran) = rank_k[i+1]
            bettiNumbers(iAE, l_codim) = dim_k[i+1] - rank_k[i] - rank_k[i+1];
        }
    }

    for(int i(0); i < nLowerDims+1; ++i)
        delete entity_in_AE[i];

    for (int i(1); i < nLowerDims+1; ++i)
        delete AE_entity[i];
}
}//namespace parelag
