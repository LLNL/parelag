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
#include "../linalg/MatrixUtils.hpp"

extern "C" {
    // LU decomposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // Solve
    void dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int* ldb, int *info);

}


CochainProjector::CochainProjector(AgglomeratedTopology * cTopo_, DofHandler * cDof_, DofAgglomeration * dofAgg_, SparseMatrix * P_):
		cTopo(cTopo_),
		cDof(dynamic_cast<DofHandlerALG*>(cDof_) ),
		dofAgg(dofAgg_),
		P(P_),
		dofLinearFunctional( cDof->GetMaxCodimensionBaseForDof() + 1),
		Pi(NULL)
{

	if(cDof == NULL)
		mfem_error("CochainProjector #1");

	for(int i(0); i < dofLinearFunctional.Size(); ++i)
		dofLinearFunctional[i] = new ElementalMatricesContainer(cDof->GetNumberEntities(static_cast<AgglomeratedTopology::EntityByCodim>(i) ));


}

void CochainProjector::CreateDofFunctional(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, const MultiVector & localProjector, const SparseMatrix & M_ii)
{
	if( entity_type > cDof->GetMaxCodimensionBaseForDof() )
		mfem_error("CochainProjector::CreateDofFunctional #1");

	if(entity_id >= cDof->GetNumberEntities(entity_type) )
		mfem_error("CochainProjector::CreateDofFunctional #2");

	if(M_ii.Width() != localProjector.Size() )
		mfem_error("CochainProjector::CreateDofFunctional #3");

	if(M_ii.Size() != localProjector.Size() )
		mfem_error("CochainProjector::CreateDofFunctional #3");


	int nFineDof = localProjector.Size();
	int nCoarseDof = localProjector.NumberOfVectors();
	DenseMatrix * dof_lin_func = new DenseMatrix(nCoarseDof, nFineDof);

	if(nCoarseDof != 0)
	{
		MultiVector MlP(nCoarseDof, nFineDof);
		MatrixTimesMultiVector(M_ii, localProjector, MlP);
		MlP.CopyToDenseMatrixT(0, nCoarseDof, 0, nFineDof, *dof_lin_func);

		StdInnerProduct dot( nFineDof );
		DenseMatrix cLocMass_ii(nCoarseDof, nCoarseDof);
		dot(localProjector, MlP, cLocMass_ii);

		if( cLocMass_ii.CheckFinite() )
			mfem_error("CochainProjector::CreateDofFunctional #4");

		double * data_cLocMass = cLocMass_ii.Data();
		double * data_MlP = dof_lin_func->Data();
		int * ipiv = new int[nCoarseDof];

		// Compute the LU P factorization of xLocMass and Solve cLocMass^-1 MlP'
		{
			char trans = 'N';
			int  n     = nCoarseDof;
			int  nrhs  = nFineDof;
			int  info;
			dgetrf_(&n, &n, data_cLocMass, &n, ipiv, &info);
			dgetrs_(&trans, &n, &nrhs, data_cLocMass, &n, ipiv, data_MlP, &n, &info);
		}

		delete[] ipiv;


		if( dof_lin_func->CheckFinite() )
		{
			std::cout << "Topology::entity " << entity_type << " entity_id " << entity_id << "\n";
			MatrixTimesMultiVector(M_ii, localProjector, MlP);
			dot(localProjector, MlP, cLocMass_ii);

			cLocMass_ii.PrintMatlab(std::cout);

			mfem_error("CochainProjector::CreateDofFunctional #5");
		}
	}

	dofLinearFunctional[entity_type]->SetElementalMatrix(entity_id, dof_lin_func);
}

void CochainProjector::SetDofFunctional(AgglomeratedTopology::EntityByCodim entity_type, int entity_id, DenseMatrix * dof_lin_func )
{
	dofLinearFunctional[entity_type]->SetElementalMatrix(entity_id, dof_lin_func);
}

/*
void CochainProjector::LocalResidualBdr(Topology::entity entity_type, int entity_id, MultiVector & res)
{
	MultiVector vFine(res.NumberOfVectors(), res.Size());
	vFine = res;

	int sizeVCoarse(0);

	Array<int> nFacets;
	for(int codim(cDof->GetMaxCodimensionBaseForDof()); codim > entity_type; --codim)
	{
		SparseMatrix & entity_nFacets = const_cast<SparseMatrix &>( cTopo->GetConnectivity(entity_type, codim) );
		entity_nFacets.GetRow(entity_id, nFacets);
		int * nfacets_it = nFacets.GetData();

		for(int * end = nfacets_it + nFacets.Size(); nfacets_it != end; ++nfacets_it)
			sizeVCoarse += cDof->GetNumberInteriorDofs(static_cast<Topology::entity>(codim), *nfacets_it);
	}

	MultiVector vCoarse(res.NumberOfVectors(), sizeVCoarse);
	vCoarse = 0.0;

	ProjectLocalBdr(entity_type, entity_id, vFine, vCoarse, res);
}

void CochainProjector::ProjectLocalBdr(Topology::entity entity_type, int entity_id, const MultiVector & vFine, MultiVector & vCoarse, MultiVector & res)
{
	mfem_error("CochainProjector::ProjectLocalBdr(Topology::entity entity_type, int entity_id, const MultiVector & vFine, MultiVector & vCoarse, MultiVector & res)");
	int nv = vFine.NumberOfVectors();

	int codim_max = cDof->GetMaxCodimensionBaseForDof();

	vCoarse = 0.;
	res = vFine;

	MultiVector vOnBdrEntity;

	int sizeVCoarse(0);
	Array<int> nFacets;

	Array<int> aggFineDofs, cDofs;
	dofAgg->GetViewAgglomerateDofGlobalNumering(entity_type, entity_id, aggFineDofs);
	cDof->GetDofs(entity_type, entity_id, cDofs);
	DenseMatrix Ploc;
	P->GetSubMatrix(aggFineDofs, cDofs, Ploc);

	Table bdr_map;
	dofAgg->MapDofFromLocalNumberingAEtoLocalNumberingBoundaryAE(entity_type, entity_id, bdr_map);

	for(int codim(codim_max); codim > entity_type; --codim)
	{
		SparseMatrix & entity_nFacets = const_cast<SparseMatrix &>( cTopo->GetConnectivity(entity_type, codim) );
		entity_nFacets.GetRow(entity_id, nFacets);
		int * nfacets_it = nFacets.GetData();

		for(int * end = nfacets_it + nFacets.Size(); nfacets_it != end; ++nfacets_it)
		{
			// Select the appropriate Agglomerate Dofs
			// Put in vOnBdrEntity the appropriate dofs of res.
			// Select the appropriate Coarse Dofs
			// Do the multiplication with dofLinearFunctional
			// Set the results of the multiplication in vCoarse
		}
//			dofAgg->GetViewAgglomerateInternalDofGlobalNumering(static_cast<Topology::entity>(codim), *nfacets_it, internalAggDofs);
//			cDof->GetInteriorDofs(static_cast<Topology::entity>(codim), ientity, internalCDofs);

//			res_local.SetSizeAndNumberOfVectors(internalAggDofs.Size(), nv);
//			vCoarse_local.SetSizeAndNumberOfVectors(internalCDofs.Size(), nv);

//			res.GetSubMultiVector(internalAggDofs, res_local);
//			Mult( dofLinearFunctional[codim]->GetElementalMatrix(ientity), res_local, vCoarse_local);

//			vCoarse.SetSubMultiVector(internalCDofs, vCoarse_local);
		}

		res = vFine;
		Mult(-1, Ploc, vCoarse, res);
}

void CochainProjector::ProjectLocalBdr(Topology::entity entity_type, int entity_id, const MultiVector & vFine, MultiVector & vCoarse)
{
	MultiVector res( vFine.NumberOfVectors(), vFine.Size() );
	ProjectLocalBdr(entity_type, entity_id, vFine, vCoarse, res);
}
*/
void CochainProjector::Project(const MultiVector & vFine, MultiVector & vCoarse)
{
	if(Pi)
		MatrixTimesMultiVector(*Pi, vFine, vCoarse);
	else
	{
		MultiVector res( vFine.NumberOfVectors(), vFine.Size() );
		Project(vFine, vCoarse, res);
	}

}

void CochainProjector::Project(const MultiVector & vFine, MultiVector & vCoarse, MultiVector & res)
{
	if(!Finalized())
		mfem_error("CochainProjector::Project(const MultiVector & vFine, MultiVector & vCoarse, MultiVector & res) #0");

	int nv = vFine.NumberOfVectors();

	int codim_max = cDof->GetMaxCodimensionBaseForDof();

	vCoarse = 0.;
	res = vFine;

	MultiVector vCoarse_d;

	MultiVector res_local, vCoarse_local;
	Array<int> internalAggDofs, internalCDofs;

	for(int codim(codim_max); codim >= 0; --codim)
	{
		for(int ientity(0); ientity < cDof->GetNumberEntities(static_cast<AgglomeratedTopology::EntityByCodim>(codim)); ++ientity)
		{
			dofAgg->GetViewAgglomerateInternalDofGlobalNumering(static_cast<AgglomeratedTopology::EntityByCodim>(codim), ientity, internalAggDofs);
			cDof->GetInteriorDofs(static_cast<AgglomeratedTopology::EntityByCodim>(codim), ientity, internalCDofs);

			res_local.SetSizeAndNumberOfVectors(internalAggDofs.Size(), nv);
			vCoarse_local.SetSizeAndNumberOfVectors(internalCDofs.Size(), nv);

#if 1
			vCoarse.GetSubMultiVector(internalCDofs, vCoarse_local);
			if(vCoarse_local.Norml2() > 1e-14)
				mfem_error("CochainProjector::Project(const MultiVector & vFine, MultiVector & vCoarse, MultiVector & res) #1");

#endif

			res.GetSubMultiVector(internalAggDofs, res_local);
			Mult( dofLinearFunctional[codim]->GetElementalMatrix(ientity), res_local, vCoarse_local);

			vCoarse.SetSubMultiVector(internalCDofs, vCoarse_local);
		}

		res = vFine;
		MatrixTimesMultiVector(-1, *P, vCoarse, res);
	}

}

void CochainProjector::ComputeProjector()
{
	/*
	 * ALGORITHM:
	 * Pi_{codim_base} = \hat{ Pi_{codim_base} }
	 * Pi_{codim - 1} = Pi_{codim} + hat{ Pi_{codim-1} } (I - P Pi_codim )
	 *
	 * where \hat{ Pi_codim } is assembled from dofLinearFunctional[codim]
	 */

	if(Pi)
		return;

	if( !Finalized() )
		mfem_error("Please finish the setup of the projector before assembling it as a SparseMatrix\n");

	if(P->Width() != cDof->GetNDofs() )
		mfem_error("P->Width() != cDof->GetNDofs()" );

	if(P->Size() != dofAgg->GetDofHandler()->GetNDofs() )
		mfem_error("P->Width() != cDof->GetNDofs()" );

//	int nFineDofs = P->Size();
//	int nCoarseDofs = P->Width();
	int codimBase = cDof->GetMaxCodimensionBaseForDof();

	SparseMatrix * Pi_next = NULL, * Pi_this = NULL;

	Pi_this = assembleInternalProjector(codimBase);

	for(int codim(codimBase-1); codim >= 0; --codim)
	{
		SparseMatrix * hatPi = assembleInternalProjector(codim);
		SparseMatrix * PPi_this = Mult(*P, *Pi_this);
		SparseMatrix * hatPiPPi_this = Mult(*hatPi, *PPi_this);
		Pi_next = Add(1., *Pi_this, 1., *hatPi, -1., *hatPiPPi_this);
		delete PPi_this;
		delete hatPiPPi_this;
		delete hatPi;
		delete Pi_this;
		Pi_this = Pi_next;
	}

	Pi = Pi_this;

}

const SparseMatrix & CochainProjector::GetProjectorMatrix()
{
	if(!Pi)
		mfem_error("Need to Call Compute Projector first\n");

	return *Pi;

}

SparseMatrix * CochainProjector::GetIncompleteProjector()
{
	int nFineDofs = P->Size();
	int nCoarseDofs = P->Width();
	SparseMatrix * out = new SparseMatrix(nCoarseDofs, nFineDofs);
	Array<int> internalAggDofs, internalCDofs;

	int baseCodim = cDof->GetMaxCodimensionBaseForDof();

	for(int codim(baseCodim); codim >= 0; --codim)
	{
		ElementalMatricesContainer & data = *(dofLinearFunctional[codim]);

		for(int ientity(0); ientity < cDof->GetNumberEntities(static_cast<AgglomeratedTopology::EntityByCodim>(codim)); ++ientity)
		{
			dofAgg->GetViewAgglomerateInternalDofGlobalNumering(static_cast<AgglomeratedTopology::EntityByCodim>(codim), ientity, internalAggDofs);
			cDof->GetInteriorDofs(static_cast<AgglomeratedTopology::EntityByCodim>(codim), ientity, internalCDofs);
			out->AddSubMatrix( internalCDofs, internalAggDofs, data.GetElementalMatrix(ientity) );
		}
	}

	out->Finalize();

	return out;

}

void CochainProjector::Check()
{
	if(!Finalized())
		mfem_error("Check() #0");

	int nFineDofs = P->Size();
	int nCoarseDofs = P->Width();
	int nv = 5;
	int seed = 1;

	MultiVector vc(nv, nCoarseDofs), vf(nv, nFineDofs), Piv(nv, nCoarseDofs);

	vc.Randomize(seed);
	vf = 0.0;

	MatrixTimesMultiVector(*P, vc, vf);

	if( vc.CheckFinite() )
		mfem_error("void CochainProjector::Check() #1");

	if( vf.CheckFinite() )
		mfem_error("void CochainProjector::Check() #2");

	Project(vf, Piv);

	if( Piv.CheckFinite() )
		mfem_error("void CochainProjector::Check() #3");

	Piv -= vc;

	if( Piv.Normlinf() > 1e-9 )
	{
		std::cout << "|| v_c - Pi P v_c ||_2 / || v_c ||_2 = " << Piv.Norml2() / vc.Norml2() << std::endl;
		std::cout << "|| v_c - Pi P v_c ||_inf / || v_c ||_inf = " << Piv.Normlinf() / vc.Normlinf() << std::endl;
	}

}

void CochainProjector::LongCheck()
{
	if(!Finalized())
		mfem_error("Check() #0");

#ifdef SLOW
	int nFineDofs = P->Size();
	int nCoarseDofs = P->Width();
	int seed = 1;

	MultiVector vc(nCoarseDofs, nCoarseDofs), vf(nCoarseDofs, nFineDofs), Piv(nCoarseDofs, nCoarseDofs);

	vc = 0.0;

	for(int i(0); i < nCoarseDofs; ++i)
		vc.GetDataFromVector(i)[i] = 1.;

	vf = 0.0;

	MatrixTimesMultiVector(*P, vc, vf);

	if( vc.CheckFinite() )
		mfem_error("void CochainProjector::Check() #1");

	if( vf.CheckFinite() )
		mfem_error("void CochainProjector::Check() #2");

	Project(vf, Piv);

	if( Piv.CheckFinite() )
		mfem_error("void CochainProjector::Check() #3");

	Piv -= vc;

	if( Piv.Normlinf() > 1e-9 )
	{
		std::cout << "|| v_c - Pi P v_c ||_2 / || v_c ||_2 = " << Piv.Norml2() / vc.Norml2() << std::endl;
		std::cout << "|| v_c - Pi P v_c ||_inf / || v_c ||_inf = " << Piv.Normlinf() / vc.Normlinf() << std::endl;
	}
#endif
}

void CochainProjector::CheckInvariants()
{
	Check();

#ifdef ELAG_DEBUG
	LongCheck();
#endif

	if(!Pi)
		ComputeProjector();

	SparseMatrix * PiP = Mult(*Pi, *P);

	bool passed = IsAlmostIdentity(*PiP, 1e-9);
	if(!passed)
		std::cout<<"Pi P is not the identity operator :( \n";

	delete PiP;

	int nFineDofs = P->Size();
	int nCoarseDofs = P->Width();
	int nv = 5;
	int seed = 1;

	MultiVector vf(nv, nFineDofs);
	vf.Randomize(seed);

	MultiVector vc1(nv, nCoarseDofs), vc2(nv, nCoarseDofs), vdiff(nv, nCoarseDofs);
	Project(vf, vc1);
	MatrixTimesMultiVector(*Pi, vf, vc2);
	subtract(vc1, vc2, vdiff);
	if(vdiff.Normlinf() > 1e-9)
	{
		std::cout<< " MatrixFree Projector and Matrix Projector are not the same \n";
		std::cout << vdiff.Normlinf() << std::setw(14) << vdiff.Norml1() << "\n";
	}

}

CochainProjector::~CochainProjector()
{

	for(int i(0); i < dofLinearFunctional.Size(); ++i )
		delete dofLinearFunctional[i];

	delete Pi;
}

int CochainProjector::Finalized()
{
	for(int i(0); i < dofLinearFunctional.Size(); ++i )
		if( ! dofLinearFunctional[i]->Finalized() )
			return 0;

	return P->Finalized() && cDof->Finalized();
}

SparseMatrix * CochainProjector::assembleInternalProjector(int codim)
{
	int nFineDofs = P->Size();
	int nCoarseDofs = P->Width();
	SparseMatrix * out = new SparseMatrix(nCoarseDofs, nFineDofs);
	Array<int> internalAggDofs, internalCDofs;
	ElementalMatricesContainer & data = *(dofLinearFunctional[codim]);

	for(int ientity(0); ientity < cDof->GetNumberEntities(static_cast<AgglomeratedTopology::EntityByCodim>(codim)); ++ientity)
	{
		dofAgg->GetViewAgglomerateInternalDofGlobalNumering(static_cast<AgglomeratedTopology::EntityByCodim>(codim), ientity, internalAggDofs);
		cDof->GetInteriorDofs(static_cast<AgglomeratedTopology::EntityByCodim>(codim), ientity, internalCDofs);
		out->AddSubMatrix( internalCDofs, internalAggDofs, data.GetElementalMatrix(ientity) );
	}

	out->Finalize();

	return out;

}
