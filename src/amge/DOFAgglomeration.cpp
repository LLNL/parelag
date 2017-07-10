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

#include "elag_amge.hpp"

#include "general/sort_pairs.hpp"
#include "../linalg/MatrixUtils.hpp"

#include "../linalg/SubMatrixExtraction.hpp"

DofAgglomeration::DofAgglomeration(AgglomeratedTopology * topo, DofHandler * dof_):
	fineTopo(topo),
	coarseTopo(topo->coarserTopology),
	dof(dof_),
	AEntity_dof(dof->GetMaxCodimensionBaseForDof()+1),
	ADof_Dof(dof->GetMaxCodimensionBaseForDof()+1),
	ADof_rDof(dof->GetMaxCodimensionBaseForDof()+1),
	AE_nInternalDof(dof->GetMaxCodimensionBaseForDof()+1),
	dof_separatorType(dof_->GetNDofs()),
	dofMapper(0)
{

	AEntity_dof = static_cast<SparseMatrix *>(0);
	ADof_Dof = static_cast<SparseMatrix *>(0);
	ADof_rDof = static_cast<SparseMatrix *>(0);
	AE_nInternalDof = static_cast<Array<int> *>(0);

	//(1) Compute the preliminary AEntity_dof matrix (still need to reorder entries and to set the correct values)
	for(int i(0); i < AEntity_dof.Size(); ++i)
		AEntity_dof[i] = Mult( topo->AEntityEntity(i), const_cast<SparseMatrix &>(dof->GetEntityDofTable(static_cast<DofHandler::entity>(i))) );


	//(2) Compute the dof_seperatorType:
	dof_separatorType = 0;
	for(int codim(1); codim < dof->GetMaxCodimensionBaseForDof()+1; ++codim)
	{
		const int nAE     = AEntity_dof[codim]->Size();
		const int * const i = AEntity_dof[codim]->GetI();
		const int * const j = AEntity_dof[codim]->GetJ();
		const int * end = j;
		const int * j_it = j;
		for(const int * i_it = i; i_it != i+nAE; ++i_it)
			for(end = j + *(i_it+1); j_it != end; ++j_it)
				dof_separatorType[*j_it] = codim;
	}

	// (3) Change the order of the entries in each row of AEntity_dof so that internal dofs appear first, fix the values of AEntity_dof.
	// (4) Compute ADof_Dof and ADof_rDof matrices
	for(int i(0); i < AEntity_dof.Size(); ++i)
	{
//		std::cout << "Space Type " << dof->GetMaxCodimensionBaseForDof() << " entity type " << i << "\n";
		const int nAGGdof = AEntity_dof[i]->NumNonZeroElems();
		const int nDof    = AEntity_dof[i]->Width();
		const int nRDof   = dof->GetEntityDofTable(static_cast<DofHandler::entity>(i)).NumNonZeroElems();
		const int nAE     = AEntity_dof[i]->Size();

		const int * i_AEntity_dof = AEntity_dof[i]->GetI();
		int * j_AEntity_dof = AEntity_dof[i]->GetJ();
		double * a_AEntity_dof = AEntity_dof[i]->GetData();

		//(2)
		if(dof->GetMaxCodimensionBaseForDof() > i)
		{
			AE_nInternalDof[i] = new Array<int>(nAE);
			Array<int> &tmp = *AE_nInternalDof[i];
			for(int iAE(0); iAE < nAE; ++iAE)
				tmp[iAE] = reorderRow(i_AEntity_dof[iAE+1] - i_AEntity_dof[iAE], j_AEntity_dof+i_AEntity_dof[iAE], a_AEntity_dof+i_AEntity_dof[iAE],
								  i, dof->GetMaxCodimensionBaseForDof()+1, dof_separatorType );
		}
		else
		{
//			std::cout<<"All dofs are internal \n";
			AE_nInternalDof[i] = new Array<int>(nAE);
			Array<int> &tmp = *AE_nInternalDof[i];
			for(int iAE(0); iAE < nAE; ++iAE)
				tmp[iAE] = i_AEntity_dof[iAE+1] - i_AEntity_dof[iAE];
		}

		int * i_ADof_dof = new int[nAGGdof+1];
		for(int kk(0); kk < nAGGdof+1; ++kk)
			i_ADof_dof[kk] = kk;

#if 0
		// Agglomerate dofs have the same orientation as global dofs in the interior of the agglomerate and depending on the orientation of the face outside
		for(double * end = a_AEntity_dof+nAGGdof; a_AEntity_dof != end; ++a_AEntity_dof)
		{
			if(fabs(*a_AEntity_dof) < 1e-5) // if it is an internal dof Adof and dof are oriented in the same way
				*a_AEntity_dof = 1.0;
			else
				(*a_AEntity_dof) = (*a_AEntity_dof < 0.0) ? -1.0 : 1.0;
		}
#else
		// Agglomerate dofs always have same orientation as global dofs
		std::fill(a_AEntity_dof, a_AEntity_dof+nAGGdof, 1.0);
#endif

		ADof_Dof[i] = new SparseMatrix(i_ADof_dof, AEntity_dof[i]->GetJ(), AEntity_dof[i]->GetData(), nAGGdof, nDof);
		SparseMatrix * Dof_rDof = Transpose( const_cast<SparseMatrix &>(dof->GetrDofDofTable(static_cast<DofHandler::entity>(i))) );

#ifdef MFEM_DEBUG
		if(Dof_rDof->Width() != Dof_rDof->NumNonZeroElems() || nRDof != Dof_rDof->Width())
		{
			std::cout << "nRDof " << nRDof << std::endl;
			std::cout << "Dof_rDof->Width() " << Dof_rDof->Width() << std::endl;
			std::cout << "Dof_rDof->NumNonZeroElems() " << Dof_rDof->NumNonZeroElems() << std::endl;
			mfem_error("caput\n");
		}
#endif

		SparseMatrix * AE_rDof = Mult( topo->AEntityEntity(i), const_cast<SparseMatrix &>(dof->GetEntityRDofTable(static_cast<DofHandler::entity>(i))));
		SparseMatrix * rDof_AE = Transpose(*AE_rDof);

		if(AE_rDof->Width() < AE_rDof->NumNonZeroElems())
		{
			std::cout << "AE_rDof->Width() " << AE_rDof->Width() << std::endl;
			std::cout << "AE_rDof->NumNonZeroElems() " << AE_rDof->NumNonZeroElems() << std::endl;
			mfem_error("caput\n");
		}

		if(AE_rDof->Width() != Dof_rDof->Width())
		{
			std::cout << "AE_rDof->Width() " << AE_rDof->Width() << std::endl;
			std::cout << "Dof_rDof->Width() " << Dof_rDof->Width() << std::endl;
			mfem_error("caput\n");
		}


//		AE_rDof->PrintMatlab(std::cout<<"AE_rDof\n");

		int nnz = 0, estimated_nnz = AE_rDof->NumNonZeroElems();
		int * i_ADof_rDof = new int[nAGGdof+1]; i_ADof_rDof[0] = 0;
		int * j_ADof_rDof = new int[estimated_nnz];
		double * a_ADof_rDof = new double[estimated_nnz];
		ADof_rDof[i] = new SparseMatrix(i_ADof_rDof, j_ADof_rDof, a_ADof_rDof, nAGGdof, Dof_rDof->Width());

		int * i_Dof_rDof = Dof_rDof->GetI();
		int * j_Dof_rDof = Dof_rDof->GetJ();
		double * a_Dof_rDof = Dof_rDof->GetData();

		int * i_rDof_AE = rDof_AE->GetI();
		int * j_rDof_AE = rDof_AE->GetJ();

		i_ADof_dof = ADof_Dof[i]->GetI();
		int * j_ADof_dof = ADof_Dof[i]->GetJ();
		double * a_ADof_dof = ADof_Dof[i]->GetData();
		int gdof;
		double sign_Adof;

		for(int iAE(0); iAE < nAE; ++iAE)
		{
			//Deal with internal dofs
			int begin, end;
			GetAgglomerateInternalDofRange(static_cast<AgglomeratedTopology::EntityByCodim>(i), iAE, begin, end);
			for(int it(begin); it != end; ++it)
			{

				gdof = j_ADof_dof[ i_ADof_dof[it] ];
				sign_Adof = a_ADof_dof[ i_ADof_dof[it] ];
				int * rdof_p;
				double * val_rodf_p;
				for(rdof_p = j_Dof_rDof+i_Dof_rDof[gdof], val_rodf_p = a_Dof_rDof+i_Dof_rDof[gdof]; rdof_p != j_Dof_rDof+i_Dof_rDof[gdof+1]; ++rdof_p, ++val_rodf_p)
				{
					if(i_rDof_AE[*rdof_p+1] - i_rDof_AE[*rdof_p] == 0)
					{
//						std::cout << "Skip rdof " << *rdof_p << " since it does not belong to any agglomerate!!\n";
						continue;
					}
					*(j_ADof_rDof+nnz) = *rdof_p;
					*(a_ADof_rDof+nnz) = *val_rodf_p * sign_Adof;
					nnz++;
				}
				i_ADof_rDof[it+1] = nnz;
//OLD AND BUGGED :(
//				nnz += i_Dof_rDof[gdof+1] - i_Dof_rDof[gdof];
//				i_ADof_rDof[it+1] = nnz;
//				std::cout << it << "\t" << gdof << "\t" << nnz << std::endl;
//				std::copy(j_Dof_rDof+i_Dof_rDof[gdof], j_Dof_rDof+i_Dof_rDof[gdof+1], j_ADof_rDof+i_ADof_rDof[it]);
//				std::copy(a_Dof_rDof+i_Dof_rDof[gdof], a_Dof_rDof+i_Dof_rDof[gdof+1], a_ADof_rDof+i_ADof_rDof[it]);
			}
			GetAgglomerateBdrDofRange(static_cast<AgglomeratedTopology::EntityByCodim>(i), iAE, begin, end);
			for(int it(begin); it != end; ++it)
			{
				gdof = j_ADof_dof[i_ADof_dof[it]];
				sign_Adof = a_ADof_dof[ i_ADof_dof[it] ];
				int * rdof_p;
				double * val_rdof_p = a_Dof_rDof+i_Dof_rDof[gdof];
				for(rdof_p = j_Dof_rDof+i_Dof_rDof[gdof]; rdof_p != j_Dof_rDof+i_Dof_rDof[gdof+1]; ++rdof_p, ++val_rdof_p)
				{
					//*(a_ADof_rDof+i_ADof_rDof[it]) = rDof for all rDof associated to gdof that belongs to iAE.
#ifdef MFEM_DEBUG
					if(*rdof_p >= Dof_rDof->Width())
						mfem_error("too large *rdof_p");
#endif
					if(i_rDof_AE[*rdof_p+1] - i_rDof_AE[*rdof_p] == 0)
					{
//						std::cout << "Skip rdof " << *rdof_p << " since it does not belong to any agglomerate!!\n";
						continue;
					}
#ifdef MFEM_DEBUG
					if(i_rDof_AE[*rdof_p+1] - i_rDof_AE[*rdof_p] != 1)
					{
						mfem_error("i_rDof_AE[*rdof_p+1] - i_rDof_AE[*rdof_p] != 1");
					}
#endif
					if( j_rDof_AE[ i_rDof_AE[*rdof_p] ] == iAE )
					{
						*(j_ADof_rDof+nnz) = *rdof_p;
						*(a_ADof_rDof+nnz) = *val_rdof_p * sign_Adof;
						nnz++;
					}
				}
				i_ADof_rDof[it+1] = nnz;

#ifdef MFEM_DEBUG
				if(i_ADof_rDof[it+1] - i_ADof_rDof[it] == 0)
				{
					std::cout << "error: i_ADof_rDof[it+1] - i_ADof_rDof[it] == 0\n";
					for(rdof_p = j_Dof_rDof+i_Dof_rDof[gdof]; rdof_p != j_Dof_rDof+i_Dof_rDof[gdof+1]; ++rdof_p)
						std::cout<< *rdof_p << "\t" << i_rDof_AE[*rdof_p] << "\t" << j_rDof_AE[ i_rDof_AE[*rdof_p] ] << "\t" << iAE <<std::endl;

					for(rdof_p = j_Dof_rDof+i_Dof_rDof[gdof]; rdof_p != j_Dof_rDof+i_Dof_rDof[gdof+1]; ++rdof_p)
						std::cout<<i_rDof_AE[*rdof_p+1] - i_rDof_AE[*rdof_p] << std::endl;

					mfem_error("caput");
				}
#endif

//				std::cout << it << "\t" << gdof << "\t" << nnz << std::endl;
			}
		}

		if(nnz != estimated_nnz )
		{
			std::cout<<"nnz is " << nnz << "#estimated nnz is " << estimated_nnz << "\n";
			mfem_error("wrong nnz!");
		}

		delete rDof_AE;
		delete AE_rDof;
		delete Dof_rDof;
	}

}

void DofAgglomeration::GetAgglomerateInternalDofRange(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, int & begin, int & end)
{
	begin = AEntity_dof[entity_type]->GetI()[entity_id];
	end = begin + AE_nInternalDof[entity_type]->operator [](entity_id);
}

void DofAgglomeration::GetAgglomerateBdrDofRange(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, int & begin, int & end)
{
	begin = AEntity_dof[entity_type]->GetI()[entity_id] + AE_nInternalDof[entity_type]->operator [](entity_id);
	end = AEntity_dof[entity_type]->GetI()[entity_id+1];
}

void DofAgglomeration::GetAgglomerateDofRange(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, int & begin, int & end)
{
	begin = AEntity_dof[entity_type]->GetI()[entity_id];
	end = AEntity_dof[entity_type]->GetI()[entity_id+1];
}

int * DofAgglomeration::GetAgglomerateInternalDofGlobalNumering(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, int * gdofs)
{
	int * begin = AEntity_dof[entity_type]->GetJ() + AEntity_dof[entity_type]->GetI()[entity_id];
	int * end   = begin+AE_nInternalDof[entity_type]->operator [](entity_id);

	for(int * it = begin; it != end; ++it, ++gdofs)
		*gdofs =*it;

	return gdofs;
}

MultiVector * DofAgglomeration::DistributeGlobalMultiVector(AgglomeratedTopology::EntityByCodim entity_type, const MultiVector & vg)
{

	if( vg.Size() != dof->GetNDofs() )
		mfem_error("vg has the wrong size!");

	int nv = vg.NumberOfVectors();
	int nDistDof = AEntity_dof[entity_type]->NumNonZeroElems();
	MultiVector * out = new MultiVector(nv, nDistDof );
	Array<int> dofs(AEntity_dof[entity_type]->GetJ(), nDistDof );
	vg.GetSubMultiVector(dofs, *out);
	return out;
}

Vector * DofAgglomeration::DistributeGlobalVector(AgglomeratedTopology::EntityByCodim entity_type, Vector & vg)
{
	if( vg.Size() != dof->GetNDofs() )
	{
		std::cout << "vg.Size() = " << vg.Size() << "\n";
		std::cout << "dof->GetNDofs() = " << dof->GetNDofs() << "\n";
		mfem_error("vg has the wrong size!");
	}

	int nDistDof = AEntity_dof[entity_type]->NumNonZeroElems();
	Vector * out = new Vector(nDistDof );
	Array<int> dofs(AEntity_dof[entity_type]->GetJ(), nDistDof );
	vg.GetSubVector(dofs, *out);
	return out;
}

#if 0
void DofAgglomeration::MapDofFromLocalNumberingAEtoLocalNumberingBoundaryAE(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, Table & map)
{
	//never tested :(
	mfem_error("DofAgglomeration::MapDofFromLocalNumberingAEtoLocalNumberingBoundaryAE(Topology::entity entity_type, int entity_id, Table & map)");
	dofMapper.SetSize( dof->GetNDofs() );

	//(1) loop on the boundary of AE and write the Agglomerate local numbering in the opportune dof mapper.
	Array<int> globalIdsBdrDofs;
	int aggBdrLocalStart, aggBdrLocalEnd;
	DofAgglomeration::GetAgglomerateBdrDofRange(entity_type, entity_id, aggBdrLocalStart, aggBdrLocalEnd);
	GetViewAgglomerateBdrDofGlobalNumering(entity_type, entity_id, globalIdsBdrDofs);

	for( int i(0);  i < globalIdsBdrDofs.Size(); ++i )
		dofMapper[ globalIdsBdrDofs[i] ] = aggBdrLocalStart+i;

	// All Bdr Dof are mapped once and only once.
	int nnz_map = aggBdrLocalEnd - aggBdrLocalStart;

	//(2) Determine the number of bdr entities (== # rows in map)
	int nBdrEntities(0);

	for(int bdr_type(dof->GetMaxCodimensionBaseForDof()); bdr_type > entity_type; --bdr_type)
		nBdrEntities += coarseTopo->GetConnectivity(entity_type, bdr_type).RowSize(entity_id);

	//(3) We can build map.
	map.SetDims(nBdrEntities, nnz_map);

	int * i_map = map.GetI();
	int * j_map = map.GetJ();

	i_map[0] = 0;
	nnz_map = 0;
	Vector discard;
	Array<int> Af, internalDofsAf_globalNumbering;
	for(int bdr_type(dof->GetMaxCodimensionBaseForDof()); bdr_type > entity_type; --bdr_type)
	{
		SparseMatrix & Ae_Af = const_cast<SparseMatrix &>( coarseTopo->GetConnectivity(entity_type, bdr_type) );
		Ae_Af.GetRow(entity_id, Af, discard);
		int * Af_it = Af.GetData();
		for(int * end = Af_it+Af.Size(); Af_it != end; ++Af_it)
		{
			GetViewAgglomerateInternalDofGlobalNumering(static_cast<AgglomeratedTopology::EntityByCodim>(bdr_type), *Af_it, internalDofsAf_globalNumbering);
			*(i_map++) = (nnz_map += GetNumberAgglomerateInternalDofs(static_cast<AgglomeratedTopology::EntityByCodim>(bdr_type), *Af_it) );
			int * iDa_it = internalDofsAf_globalNumbering.GetData();
			for(int * end = j_map+nnz_map; j_map != end; ++j_map, ++iDa_it)
				*j_map = dofMapper[*iDa_it];
		}
	}


}
#endif

void DofAgglomeration::GetViewAgglomerateInternalDofGlobalNumering(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, Array<int> & gdofs)
{
	int begin = AEntity_dof[entity_type]->GetI()[entity_id];
	int size  = AE_nInternalDof[entity_type]->operator [](entity_id);
	gdofs.MakeRef(AEntity_dof[entity_type]->GetJ()+begin, size);
}

void DofAgglomeration::GetViewAgglomerateBdrDofGlobalNumering(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, Array<int> & gdofs)
{
	int begin = AEntity_dof[entity_type]->GetI()[entity_id] + AE_nInternalDof[entity_type]->operator [](entity_id);
	int end   = AEntity_dof[entity_type]->GetI()[entity_id+1];
	int size  = end - begin;
	gdofs.MakeRef(AEntity_dof[entity_type]->GetJ()+begin, size);
}

void DofAgglomeration::GetViewAgglomerateDofGlobalNumering(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, Array<int> & gdofs)
{
	int begin = AEntity_dof[entity_type]->GetI()[entity_id] ;
	int end   = AEntity_dof[entity_type]->GetI()[entity_id+1];
	int size  = end - begin;
	gdofs.MakeRef(AEntity_dof[entity_type]->GetJ()+begin, size);
}

void DofAgglomeration::CheckAdof()
{
	bool good;
	for(int i(0); i < AEntity_dof.Size(); ++i)
	{

		for(const double * it = ADof_rDof[i]->GetData(); it != ADof_rDof[i]->GetData()+ADof_rDof[i]->NumNonZeroElems(); ++it)
		{
			if(fabs(*it-1.0) < 1e-6 && fabs(*it-1.0) > 1e-6)
			{
				std::cout<<"ADof_rDof["<<i<<"] has an entry equal to " << *it <<std::endl;
				mfem_error("kill!!");
			}
		}

		for(const double * it = ADof_Dof[i]->GetData(); it != ADof_Dof[i]->GetData()+ADof_Dof[i]->NumNonZeroElems(); ++it)
		{
			if(fabs(*it-1.0) < 1e-6 && fabs(*it-1.0) > 1e-6)
			{
				std::cout<<"ADof_Dof["<<i<<"] has an entry equal to " << *it <<std::endl;
				mfem_error("kill!!");
			}
		}


		SparseMatrix * rDof_ADof = Transpose(*ADof_rDof[i]);
		SparseMatrix * rdof_dof = Mult(*rDof_ADof, *ADof_Dof[i]);

		for(double * it = rdof_dof->GetData(); it != rdof_dof->GetData()+rdof_dof->NumNonZeroElems(); ++it)
		{
			if(fabs(*it) < .5)
				mfem_error("Something strange happened!");
		}

		const SparseMatrix & rdof_dof2 = dof->GetrDofDofTable(static_cast<AgglomeratedTopology::EntityByCodim>(i) );

//		std::cout << "entity codimension: " << i << std::endl;
		good = AreAlmostEqual(*rdof_dof, rdof_dof2, *rdof_dof, "(rDof_aDof*aDof_dof)", "rDof_dof", "(rDof_aDof*aDof_dof)", 1e-6);

		if(!good)
		{
			std::cout<<"Num of non-zeros per row rDof_ADof" << rDof_ADof->MaxRowSize()<<std::endl;
			std::cout<<"Num of non-zeros per row ASof_Dof" << ADof_Dof[i]->MaxRowSize()<<std::endl;
			std::cout<<"Num of non-zeros per row rdof_dof" << rdof_dof->MaxRowSize()<<std::endl;
			rdof_dof->PrintMatlab(std::cout<<"Rdof_dof_computed\n");
//			rdof_dof2.PrintMatlab(std::cout<<"Rdof_dof_original\n");
			exit(1);
//			std::cout<<"rdof_dof_computed: " << rdof_dof->Size() << " " << rdof_dof->NumNonZeroElems() << std::endl;
//			ADof_rDof[i]->PrintMatlab(std::cout << "ADof_rDof \n");
//			ADof_Dof[i]->PrintMatlab(std::cout << "ADof_Dof \n");
		}

		delete rdof_dof;
		delete rDof_ADof;

	}
}


DofAgglomeration::~DofAgglomeration()
{
	for(int i(0); i < dof->GetMaxCodimensionBaseForDof()+1; ++i)
	{
		delete AE_nInternalDof[i];
		delete ADof_rDof[i];
		delete[] ADof_Dof[i]->GetI();
		ADof_Dof[i]->LoseData();
		delete ADof_Dof[i];
		delete AEntity_dof[i];
	}
}

int DofAgglomeration::reorderRow(int nentries, int * j, double * a, int minWeight, int maxWeight, const Array<int> & weights)
{

	int nMinWeight(0);
	//SAFER IMPLEMENTATION
	Triple<int,int, double> * triples = new Triple<int, int, double>[nentries];

	for(int ii(0); ii < nentries; ++ii)
	{
		if( (triples[ii].one = weights[ j[ii] ]) == minWeight)
			++nMinWeight;
		triples[ii].two = j[ii];
		triples[ii].three = a[ii];
	}

	SortTriple (triples, nentries);

	for(int ii(0); ii < nentries; ++ii)
	{
		j[ii] = triples[ii].two;
		a[ii] = triples[ii].three;
	}

	delete[] triples;


	return nMinWeight;



#if 0
	int tmpj;
	int tmpa;

	int f_it(0), b_it(nentries-1);

	//partition: store first all the entries with weight == minWeight
	while(f_it < b_it)
	{
		while( weights[ j[f_it] ] == minWeight && f_it < nentries)
			++f_it;


		while( weights[ j[b_it] ] != minWeight && b_it > -1 )
			--b_it;

		if(f_it == nentries || b_it == -1)
		{
			std::stringstream errorMSG;
			errorMSG << "I'm in the wrong place! minWeight = " << minWeight << " nentries = " << nentries << "\n";
			errorMSG << "j \t w[j] \n";
			for(int i(0); i < nentries; ++i)
				errorMSG << j[i] << "\t" << weights[ j[i] ] << "\n";

			std::cout<< errorMSG.str();
			break;
		}


		tmpj = j[b_it];
		tmpa = a[b_it];
		j[b_it] = j[f_it];
		a[b_it] = a[f_it];
		j[f_it] = tmpj;
		a[f_it] = tmpa;
	}

	//recursion: we reorder the last part of the vector by increasing of 1 minWeight.
	if(++minWeight < maxWeight && f_it < nentries)
		reorderRow(nentries - f_it, j+f_it, a + f_it, minWeight, maxWeight, weights);

	return f_it;
#endif
}

//------------------------------------------------------------------------------------//



SparseMatrix * AssembleAgglomerateMatrix(AgglomeratedTopology::EntityByCodim entity_type, SparseMatrix & M_e, DofAgglomeration * range, DofAgglomeration * domain)
{
	SparseMatrix * Rt = range->ADof_rDof[entity_type];
	SparseMatrix * Pt = domain->ADof_rDof[entity_type];

	SparseMatrix * P = Transpose(*Pt);
	SparseMatrix * RtM_e = Mult(*Rt, M_e);

	SparseMatrix * RtM_eP = Mult(*RtM_e, *P);

	delete RtM_e;
	delete P;
	return RtM_eP;
}

SparseMatrix * AssembleAgglomerateRowsGlobalColsMatrix(AgglomeratedTopology::EntityByCodim entity_type, SparseMatrix & M_e, DofAgglomeration * range, DofHandler * domain)
{
	SparseMatrix * Rt  = range->ADof_rDof[entity_type];
	SparseMatrix * tmp = Mult(*Rt, M_e);
	SparseMatrix & P   = const_cast<SparseMatrix &>( domain->GetrDofDofTable(entity_type) );
	SparseMatrix * out = Mult(*tmp, P);

	delete tmp;

	return out;
}

SparseMatrix * Assemble(AgglomeratedTopology::EntityByCodim entity_type, SparseMatrix & M_a, DofAgglomeration * range, DofAgglomeration * domain)
{
	int flag(0);
	flag += (domain == NULL) ? 0 : 1;
	flag += (range == NULL) ? 0 : 2;

	SparseMatrix * R, *P, *Rt, *RtM_a, *M_g;

	switch(flag)
	{
	case 0: /*domain == NULL && range == NULL*/
		mfem_error("SparseMatrix * Assemble(Topology::entity entity_type, SparseMatrix & M_a, "
				"DofAgglomeration * range, DofAgglomeration * domain): range and domain can't both be null");
		return static_cast<SparseMatrix *>(NULL);

	case 1: /*domain != NULL && range == NULL*/
//		std::cout << "Case 1: domain != NULL && range == NULL" << std::endl;
		P = domain->ADof_Dof[entity_type];
		M_g = Mult(M_a, *P);
		return M_g;

	case 2: /*domain == NULL && range != NULL*/
//		std::cout << "Case 2: domain == NULL && range != NULL" << std::endl;
		R = range->ADof_Dof[entity_type];
		Rt = Transpose(*R);
		M_g = Mult(*Rt, M_a);
		delete Rt;
		return M_g;

	case 3: /*domain != NULL && range != NULL*/
//		std::cout << "Case 3: domain != NULL && range != NULL" << std::endl;
		R = range->ADof_Dof[entity_type];
		P = domain->ADof_Dof[entity_type];

		Rt = Transpose(*R);
		RtM_a = Mult(*Rt, M_a);
		M_g = Mult(*RtM_a, *P);

		delete RtM_a;
		delete Rt;
		return M_g;

	default:
		mfem_error("SparseMatrix * Assemble(Topology::entity entity_type, SparseMatrix & M_a, "
				"DofAgglomeration * range, DofAgglomeration * domain): Impossible value of flag");
		return static_cast<SparseMatrix *>(NULL);
	}
}



SparseMatrix * DistributeAgglomerateMatrix(AgglomeratedTopology::EntityByCodim entity_type, SparseMatrix & D_g, DofAgglomeration * range, DofAgglomeration * domain)
{

	int flag(0);
	flag += (domain == NULL) ? 0 : 1;
	flag += (range == NULL) ? 0 : 2;

	SparseMatrix * RtD_gP, *P;

	switch(flag)
	{
	case 0: /*domain == NULL && range == NULL*/
		mfem_error("SparseMatrix * Assemble(Topology::entity entity_type, SparseMatrix & M_a, "
				"DofAgglomeration * range, DofAgglomeration * domain): range and domain can't both be null");
		return static_cast<SparseMatrix *>(NULL);

	case 1: /*domain != NULL && range == NULL*/
//		std::cout << "Case 1: domain != NULL && range == NULL" << std::endl;
		P = Transpose(* (domain->ADof_Dof[entity_type]) );
		RtD_gP = Mult(D_g, *P);
		delete P;
		return RtD_gP;

	case 2: /*domain == NULL && range != NULL*/
//		std::cout << "Case 2: domain == NULL && range != NULL" << std::endl;
		return Mult(*(range->ADof_Dof[entity_type]), D_g);

	case 3: /*domain != NULL && range != NULL*/
//		std::cout << "Case 3: domain != NULL && range != NULL" << std::endl;
		return Distribute(D_g, *(range->AEntity_dof[entity_type]), *(domain->AEntity_dof[entity_type]) );

	default:
		mfem_error("SparseMatrix * Assemble(Topology::entity entity_type, SparseMatrix & M_a, "
				"DofAgglomeration * range, DofAgglomeration * domain): Impossible value of flag");
		return static_cast<SparseMatrix *>(NULL);
	}
}

SparseMatrix * DistributeProjector(AgglomeratedTopology::EntityByCodim entity_type, SparseMatrix & P_t, DofHandler * coarseRange, DofAgglomeration * domain)
{

	return Distribute(P_t, coarseRange->GetEntityDofTable(entity_type), *(domain->AEntity_dof[entity_type]) );

/*
	const SparseMatrix & CRDof_CDof = coarseRange->GetrDofDofTable(entity_type);
	SparseMatrix * dg_PT = Mult(const_cast<SparseMatrix &>(CRDof_CDof), P_t);
	SparseMatrix * aFDof_FDof  = domain->ADof_Dof[entity_type];
	SparseMatrix * FDof_aFDof  = Transpose(*aFDof_FDof);
	SparseMatrix * dg_PT_loc = Mult(*dg_PT, *FDof_aFDof);

	const SparseMatrix & AE_CRDof = coarseRange->GetEntityRDofTable(entity_type);
	int nAE = AE_CRDof.Size();
	const int * i_AE_CRDof = AE_CRDof.GetI();
	const int * j_AE_CRDof = AE_CRDof.GetJ();
//	const double * a_AE_CRDof = AE_CRDof.GetData();

	const int * i = dg_PT_loc->GetI();
	const int * j = dg_PT_loc->GetJ();
	const double * a = dg_PT_loc->GetData();

	int * i_trimmed = dg_PT_loc->GetI();
	int * j_trimmed = dg_PT_loc->GetJ();
	double * a_trimmed = dg_PT_loc->GetData();
	int nnz_trimmed = 0;

	const int * i_AE_FDof = domain->AEntity_dof[entity_type]->GetI();

	int AEfDof_start, AEfDof_end;
	for(int iAE(0); iAE < nAE; ++iAE)
	{
		AEfDof_start = i_AE_FDof[iAE];
		AEfDof_end = i_AE_FDof[iAE+1];
		for(const int * iCRDof = j_AE_CRDof+i_AE_CRDof[iAE]; iCRDof != j_AE_CRDof+i_AE_CRDof[iAE+1]; ++iCRDof)
		{
			for(int jpos = i[*iCRDof]; jpos != i[*iCRDof+1]; ++jpos)
			{
				if(j[jpos] >=  AEfDof_start && j[jpos] < AEfDof_end)
				{
					j_trimmed[nnz_trimmed] = j[jpos];
					a_trimmed[nnz_trimmed] = a[jpos];
					++nnz_trimmed;
				}
			}
			i_trimmed[*iCRDof+1] = nnz_trimmed;
		}
	}

	delete dg_PT;
	delete FDof_aFDof;

	return dg_PT_loc;
*/
}
