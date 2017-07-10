/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#include "elag_topology.hpp"


void AgglomeratedTopologyCheck::ShowBadAgglomeratedEntities(int codim, AgglomeratedTopology & topo, std::ostream & os)
{
	elag_assert( codim <= topo.Codimensions() );

	DenseMatrix betas;
	computeBettiNumbersAgglomeratedEntities(codim, topo, betas);
	switch(codim)
	{
	case 0:
		showBadAgglomeratedElements(betas, topo, os);
		break;
	case 1:
		showBadAgglomeratedFacets(betas, topo, os);
		break;
	case 2:
		showBadAgglomeratedRidges(betas, topo, os);
		break;
	default:
		break;
	}
}

void AgglomeratedTopologyCheck::showBadAgglomeratedElements(DenseMatrix & betas, AgglomeratedTopology & topo, std::ostream & os)
{
	int nDim = topo.Dimensions();

	for(int iAE = 0; iAE < betas.Height(); ++iAE)
	{
		if(betas(iAE,0) != 1)
			os << " Element " << iAE << " is disconnected. The number of connected components is " << betas(iAE,0) << "\n";

		for(int i(1); i < nDim; ++i)
			if(betas(iAE,i) != 0 )
			{
				if(nDim - 1 == i )
					os << " Element " << iAE << " has " << betas(iAE,i) << "holes. \n";
				else
					os << " Element " << iAE << " has " << betas(iAE,i) << "tunnels. \n";
			}
	}
}

void AgglomeratedTopologyCheck::showBadAgglomeratedFacets(DenseMatrix & betas, AgglomeratedTopology & topo, std::ostream & os)
{
	int nDim = betas.Width();

	for(int iAE = 0; iAE < betas.Height(); ++iAE)
	{
		if(betas(iAE,0) != 1)
			os << " Facet " << iAE << " is disconnected. The number of connected components is " << betas(iAE,0) << "\n";

		for(int i(1); i < nDim; ++i)
			if(betas(iAE,i) != 0 )
					os << " Facet " << iAE << " has " << betas(iAE,i) << "holes. \n";
	}
}

void AgglomeratedTopologyCheck::showBadAgglomeratedRidges(DenseMatrix & betas,AgglomeratedTopology & topo, std::ostream & os)
{
	for(int iAE = 0; iAE < betas.Height(); ++iAE)
	{
		if(betas(iAE,0) != 1)
			os << " Ridge " << iAE << " is disconnected. The number of connected components is " << betas(iAE,0) << "\n";
	}
}

void AgglomeratedTopologyCheck::computeBettiNumbersAgglomeratedEntities(int codim, AgglomeratedTopology & topo, DenseMatrix & betas)
{
	int nLowerDims = topo.Dimensions() - codim;

	if(nLowerDims == 0)
	{
		betas.SetSize(0);
		return;
	}

	Array< TopologyTable *>  AE_entity(nLowerDims+1);
	AE_entity[0] = &topo.AEntityEntity(codim);

	for(int i(0); i < nLowerDims; ++i)
		AE_entity[i+1] = MultBoolean(*(AE_entity[i]), topo.B(codim+i) );

	int nAE = AE_entity[0]->Size();
	Array< Array<int> * > entity_in_AE(nLowerDims+1);

	for(int i(0); i < nLowerDims+1; ++i)
		entity_in_AE[i] = new Array<int>;

	Vector dummy;

	Array<int> dim_k(nLowerDims+1), rank_k(nLowerDims+1);

	rank_k[nLowerDims] = 0;

	betas.SetSize(nAE,nLowerDims);
	for( int iAE(0); iAE < nAE; ++iAE)
	{
		for(int i(0); i < nLowerDims+1; ++i)
		{
			AE_entity[i]->GetRow(iAE, *(entity_in_AE[i]), dummy);
			dim_k[i] = entity_in_AE[i]->Size();
		}

		for(int i(0); i < nLowerDims; ++i)
		{
			DenseMatrix dloc(dim_k[i], dim_k[i+1]);
			topo.B(codim+i).GetSubMatrix(*(entity_in_AE[i]), *(entity_in_AE[i+1]), dloc);
			rank_k[i] = dloc.Rank(1.e-9);
		}

		int codim;

		for(int i(0); i < nLowerDims; ++i)
		{
			codim = nLowerDims - i - 1;
			betas(iAE, codim) = dim_k[i+1] - rank_k[i] - rank_k[i+1];
		}
	}

	for(int i(0); i < nLowerDims+1; ++i)
		delete entity_in_AE[i];

	for(int i(0); i < nLowerDims; ++i)
		delete AE_entity[i+1];
}

#if 0
void AgglomeratedTopologyCheck::computeBettiNumbersAgglomeratedElements(AgglomeratedTopology & topo, DenseMatrix & betas)
{
	int nDim = topo.Dimensions();

	Array< TopologyTable *>  AE_entity(nDim+1);
	AE_entity[0] = &(topo.AEntityEntity(0));

	for(int i(0); i < nDim; ++i)
		AE_entity[i+1] = MultBoolean(*(AE_entity[i]), topo.B(i) );

	int nAE = AE_entity[0]->Size();
	Array< Array<int> * > entity_in_AE(nDim+1);

	for(int i(0); i < nDim+1; ++i)
		entity_in_AE[i] = new Array<int>;

	Vector dummy;

	Array<int> dim_k(nDim+1), rank_k(nDim+1);

	rank_k[nDim] = 0;

	betas.SetSize(nAE, nDim);

	for( int iAE(0); iAE < nAE; ++iAE)
	{
		for(int i(0); i < nDim+1; ++i)
		{
			AE_entity[i]->GetRow(iAE, *(entity_in_AE[i]), dummy);
			dim_k[i] = entity_in_AE[i]->Size();
		}

		for(int i(0); i < nDim; ++i)
		{
			DenseMatrix dloc(dim_k[i], dim_k[i+1]);
			topo.B(i).GetSubMatrix(*(entity_in_AE[i]), *(entity_in_AE[i+1]), dloc);
			rank_k[i] = dloc.Rank(1.e-9);
		}

		int codim;
		for(int i(0); i < nDim; ++i)
		{
			codim = nDim - i - 1;
			betas(iAE, codim) = dim_k[i+1] - rank_k[i] - rank_k[i+1];
		}
	}

	for(int i(0); i < nDim+1; ++i)
		delete entity_in_AE[i];

	for(int i(0); i < nDim; ++i)
		delete AE_entity[i+1];
}

void AgglomeratedTopologyCheck::computeBettiNumbersOfAgglomeratedFacets(AgglomeratedTopology & topo, DenseMatrix & betas)
{
	int AEntity_codim = 1;
	int nLowerDims = topo.Dimensions() - AEntity_codim;

	Array< TopologyTable *>  AE_entity(nLowerDims+1);
	AE_entity[0] = &topo.AEntityEntity(AEntity_codim);

	for(int i(0); i < nLowerDims; ++i)
		AE_entity[i+1] = MultBoolean(*(AE_entity[i]), topo.B(AEntity_codim+i) );

	int nAE = AE_entity[0]->Size();
	Array< Array<int> * > entity_in_AE(nLowerDims+1);

	for(int i(0); i < nLowerDims+1; ++i)
		entity_in_AE[i] = new Array<int>;

	Vector dummy;

	Array<int> dim_k(nLowerDims+1), rank_k(nLowerDims+1);

	rank_k[nLowerDims] = 0;

	betas.SetSize(nAE,nLowerDims);
	for( int iAE(0); iAE < nAE; ++iAE)
	{
		for(int i(0); i < nLowerDims+1; ++i)
		{
			AE_entity[i]->GetRow(iAE, *(entity_in_AE[i]), dummy);
			dim_k[i] = entity_in_AE[i]->Size();
		}

		for(int i(0); i < nLowerDims; ++i)
		{
			DenseMatrix dloc(dim_k[i], dim_k[i+1]);
			topo.B(AEntity_codim+i).GetSubMatrix(*(entity_in_AE[i]), *(entity_in_AE[i+1]), dloc);
			rank_k[i] = dloc.Rank(1.e-9);
		}

		int codim;

		for(int i(0); i < nLowerDims; ++i)
		{
			codim = nLowerDims - i - 1;
			betas(iAE, codim) = dim_k[i+1] - rank_k[i] - rank_k[i+1];
		}
	}

	for(int i(0); i < nLowerDims+1; ++i)
		delete entity_in_AE[i];

	for(int i(0); i < nLowerDims; ++i)
		delete AE_entity[i+1];
}
#endif
