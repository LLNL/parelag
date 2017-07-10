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

#ifdef ELAG_USE_OPENMP
#include <omp.h>
#endif
#include "../amge/elag_amge.hpp"
#include "elag_linalg.hpp"
#include "MLDivFree.hpp"

#include <fstream>

MLDivFree::MLDivFree(BlockMatrix * A_, Array<DeRhamSequence *> & seqs, Array<int> & label_ess):
			sequence(seqs.GetData(), seqs.Size()),
			l2form(seqs[0]->GetNumberOfForms()-1),
			hdivform(seqs[0]->GetNumberOfForms()-2),
			hcurlform(seqs[0]->GetNumberOfForms()-3),
			nLevels(seqs.Size()),
			arithmeticTrueComplexity(0),
			arithmeticComplexity(0),
			A(static_cast<BlockMatrix *>(NULL)),
			Al(0),
			trueAl(0),
			Cl(0),
			P(nLevels),
			trueP(nLevels),
			AE_dof(0),
			levelTrueStart(0),
			levelTrueStartMultiplier(0),
			levelStart(0),
			levelStartMultiplier(0),
			dof_is_shared_among_AE_data(0),
			trueRhs_data(0),
			trueSol_data(0),
			rhs_data(0),
			sol_data(0),
			essnullspace_data( 0 ),
			t_data( 0 ),
			numerical_zero(1e-6)
{
	SetBlockMatrix(A_);

	AE_dof.SetSize(nLevels-1);
	P.SetSize(nLevels-1);
	trueP.SetSize(nLevels-1);
	Pi.SetSize(nLevels-1);
	if( label_ess.Max() )
	    P00_to_be_deleted.SetSize(nLevels-1);

	for(int i(0); i < nLevels-1; ++i)
	{
		Array<int> dof_offsets(3), elem_offsets(2);

		dof_offsets[0] = 0;
		dof_offsets[1] = seqs[i]->GetAEntityDof(AgglomeratedTopology::ELEMENT, hdivform)->Width();
		dof_offsets[2] = seqs[i]->GetAEntityDof(AgglomeratedTopology::ELEMENT, hdivform)->Width() + seqs[i]->GetAEntityDof(AgglomeratedTopology::ELEMENT, l2form)->Width();

		elem_offsets[0] = 0;
		elem_offsets[1] = seqs[i]->GetAEntityDof(AgglomeratedTopology::ELEMENT, hdivform)->Size();

		AE_dof[i] = new BlockMatrix(elem_offsets,dof_offsets);
		AE_dof[i]->SetBlock(0,0, seqs[i]->GetAEntityDof(AgglomeratedTopology::ELEMENT, hdivform) );
		AE_dof[i]->SetBlock(0,1, seqs[i]->GetAEntityDof(AgglomeratedTopology::ELEMENT, l2form) );

		AE_dof[i]->RowOffsets().MakeDataOwner();
		AE_dof[i]->ColOffsets().MakeDataOwner();

		dof_offsets.LoseData();
		elem_offsets.LoseData();

		Array<int> fine_offsets(3), coarse_offsets(3);

		fine_offsets[0] = 0;
		fine_offsets[1] = seqs[i]->GetP(hdivform)->Size();
		fine_offsets[2] = seqs[i]->GetP(hdivform)->Size() + seqs[i]->GetP(l2form)->Size();

		coarse_offsets[0] = 0;
		coarse_offsets[1] = seqs[i]->GetP(hdivform)->Width();
		coarse_offsets[2] = seqs[i]->GetP(hdivform)->Width() + seqs[i]->GetP(l2form)->Width();

		P[i] = new BlockMatrix(fine_offsets,coarse_offsets);
		if( label_ess.Max() )
		{
		    P00_to_be_deleted[i] = seqs[i]->GetP(hdivform, label_ess);
			P[i]->SetBlock(0,0, P00_to_be_deleted[i] );
		}
		else
			P[i]->SetBlock(0,0, seqs[i]->GetP(hdivform) );
		P[i]->SetBlock(1,1, seqs[i]->GetP(l2form) );

		Pi[i] = new BlockMatrix(coarse_offsets,fine_offsets);
		seqs[i]->GetPi(hdivform)->ComputeProjector();
		seqs[i]->GetPi(l2form)->ComputeProjector();
		Pi[i]->SetBlock(0,0, const_cast<SparseMatrix*>( &(seqs[i]->GetPi(hdivform)->GetProjectorMatrix() )) );
		Pi[i]->SetBlock(1,1, const_cast<SparseMatrix*>( &(seqs[i]->GetPi(l2form)->GetProjectorMatrix()   )) );

		P[i]->RowOffsets().MakeDataOwner();
		P[i]->ColOffsets().MakeDataOwner();
		fine_offsets.LoseData();
		coarse_offsets.LoseData();

		Array<int> true_fine_offsets(3), true_coarse_offsets(3);

		true_fine_offsets[0] = 0;
		true_fine_offsets[1] = seqs[i]->GetNumberOfTrueDofs(hdivform);
		true_fine_offsets[2] = seqs[i]->GetNumberOfTrueDofs(hdivform) + seqs[i]->GetNumberOfTrueDofs(l2form);

		true_coarse_offsets[0] = 0;
		true_coarse_offsets[1] = seqs[i+1]->GetNumberOfTrueDofs(hdivform);
		true_coarse_offsets[2] = seqs[i+1]->GetNumberOfTrueDofs(hdivform) + seqs[i+1]->GetNumberOfTrueDofs(l2form);

		trueP[i] = new BlockOperator(true_fine_offsets, true_coarse_offsets);
		trueP[i]->owns_blocks = 1;
		trueP[i]->SetBlock(0,0, seqs[i]->ComputeTrueP(hdivform, label_ess) );
		trueP[i]->SetBlock(1,1, seqs[i]->ComputeTrueP(l2form) );
		Swap( trueP[i]->RowOffsets(), true_fine_offsets);
		Swap( trueP[i]->ColOffsets(), true_coarse_offsets);
	}

	SparseMatrix * W = seqs[0]->ComputeMassOperator(l2form);
	ConstantCoefficient one(1.);
	Vector ones(W->Size());
	seqs[0]->ProjectCoefficient(l2form, one, ones);

	Cl.SetSize(nLevels);
	for(int i(0); i < nLevels; ++i)
		Cl[i] = seqs[i]->ComputeTrueDerivativeOperator(hcurlform, label_ess);

	Build(ones, *W);

	delete W;

	SetCycle(FULL_SPACE);

}

MLDivFree::~MLDivFree()
{

   for(int i(1); i < nLevels; ++i)
        delete Al[i];

   for(int i(0); i < nLevels; ++i)
        delete trueAl[i];

	for(int i(0); i < P.Size(); ++i)
	{
		delete P[i];
		delete trueP[i];
		delete AE_dof[i];
		delete Pi[i];
	}

	for(int i(0); i < P00_to_be_deleted.Size(); ++i)
	    delete P00_to_be_deleted[i];


	for(int i(0); i < nLevels; ++i)
    {
		delete Maux[i];
	    delete Cl[i];
    }
}

void MLDivFree::SetOperator(const Operator &op)
{
	std::cout << "MLDivFree::SetOperator is ignored \n";
}

void MLDivFree::SetBlockMatrix(BlockMatrix * A_)
{
	A = A_;

	elag_assert( A->Height() == sequence[0]->GetNumberOfDofs(hdivform) + sequence[0]->GetNumberOfDofs(l2form));
	elag_assert( A->GetBlock(0,0).Width() == A->GetBlock(0,0).Height() );
	elag_assert( A->Height() == A->Width() );
}

void MLDivFree::Build(const Vector & ess_nullspace_p, const SparseMatrix & mass_p)
{
	levelTrueStart.SetSize(nLevels+1);
	levelTrueStart[0] = arithmeticTrueComplexity = 0;
	for(int i(0); i < nLevels; ++i)
		levelTrueStart[i+1] = (arithmeticTrueComplexity += sequence[i]->GetNumberOfTrueDofs(hdivform) + sequence[i]->GetNumberOfTrueDofs(l2form));

	trueRhs_data.SetSize(arithmeticTrueComplexity);
	trueSol_data.SetSize(arithmeticTrueComplexity);

	levelStart.SetSize(nLevels+1);
	levelStart[0] = arithmeticComplexity = 0;
	for(int i(0); i < nLevels; ++i)
		levelStart[i+1] = (arithmeticComplexity += sequence[i]->GetNumberOfDofs(hdivform) + sequence[i]->GetNumberOfDofs(l2form));

	dof_is_shared_among_AE_data.SetSize(arithmeticComplexity);
	essnullspace_data.SetSize(arithmeticComplexity);
	t_data.SetSize(arithmeticComplexity);
	rhs_data.SetSize(levelStart[1]);
	sol_data.SetSize(levelStart[1]);

	levelTrueStartMultiplier.SetSize(nLevels);
	for(int i(0); i < nLevels; ++i)
		levelTrueStartMultiplier[i] =  levelTrueStart[i] + sequence[i]->GetNumberOfTrueDofs(hdivform);

	levelStartMultiplier.SetSize(nLevels);
	for(int i(0); i < nLevels; ++i)
		levelStartMultiplier[i] =  levelStart[i] + sequence[i]->GetNumberOfDofs(hdivform);


	Array<int> is_shared;
	for(int i(0); i < nLevels-1; ++i)
	{
		is_shared.MakeRef(dof_is_shared_among_AE_data+levelStart[i], levelStart[i+1] - levelStart[i]);
		computeSharedDof(i, is_shared);
	}



	Al.SetSize(nLevels);

	Al[0] = A;

	for(int i(0); i < nLevels-1; ++i)
		Al[i+1] = PtAP( Al[i], P[i] );

	trueAl.SetSize(nLevels);
	for(int i(0); i < nLevels; ++i)
	{
		SharingMap & maphdiv = sequence[i]->GetDofHandler(hdivform)->GetDofTrueDof();
		SharingMap & mapl2 = sequence[i]->GetDofHandler(l2form)->GetDofTrueDof();
		Array<int> trueblockoffsets(3);
		trueblockoffsets[0] = 0;
		trueblockoffsets[1] = maphdiv.GetTrueLocalSize();
		trueblockoffsets[2] = maphdiv.GetTrueLocalSize() + mapl2.GetTrueLocalSize();

		trueAl[i] = new BlockOperator(trueblockoffsets);
		trueAl[i]->owns_blocks = 1;
		Swap(trueblockoffsets, trueAl[i]->RowOffsets() );
		trueAl[i]->SetBlock(0,0, Assemble(maphdiv, Al[i]->GetBlock(0,0), maphdiv) );
		trueAl[i]->SetBlock(0,1, Assemble(maphdiv, Al[i]->GetBlock(0,1), mapl2) );
		trueAl[i]->SetBlock(1,0, Assemble(mapl2, Al[i]->GetBlock(1,0), maphdiv) );
	}

	Maux.SetSize(nLevels);
	for(int i(0); i < nLevels; ++i)
	{
		Maux[i] = new AuxHypreSmoother( dynamic_cast<HypreParMatrix&>(trueAl[i]->GetBlock(0,0)), *(Cl[i]), HypreSmoother::l1GS, 3);
		Maux[i]->iterative_mode = true;

	}


	Vector x_f, y_f, x_c, y_c;

	x_f.SetDataAndSize(essnullspace_data+levelStartMultiplier[0], levelStart[1] - levelStartMultiplier[0]);
	y_f.SetDataAndSize(           t_data+levelStartMultiplier[0], levelStart[1] - levelStartMultiplier[0]);
	x_f= ess_nullspace_p;
	mass_p.Mult(x_f,y_f);

	for(int ilevel(1); ilevel < nLevels; ++ilevel)
	{
		x_f.SetDataAndSize(essnullspace_data+levelStartMultiplier[ilevel-1], levelStart[ilevel]   - levelStartMultiplier[ilevel-1]);
		y_f.SetDataAndSize(           t_data+levelStartMultiplier[ilevel-1], levelStart[ilevel]   - levelStartMultiplier[ilevel-1]);
		x_c.SetDataAndSize(essnullspace_data+levelStartMultiplier[ilevel]  , levelStart[ilevel+1] - levelStartMultiplier[ilevel]  );
		y_c.SetDataAndSize(           t_data+levelStartMultiplier[ilevel]  , levelStart[ilevel+1] - levelStartMultiplier[ilevel]  );

		Pi[ilevel-1]->GetBlock(PRESSURE,PRESSURE).Mult(x_f, x_c);
		P[ilevel-1]->GetBlock(PRESSURE,PRESSURE).MultTranspose(y_f, y_c);
	}
}

void MLDivFree::SetCycle(MLDivFree::CycleType c)
{
	my_cycle = c;

	switch(c)
	{
	case FULL_SPACE:
		height = width = sequence[0]->GetNumberOfTrueDofs(hdivform) + sequence[0]->GetNumberOfTrueDofs(l2form);
		break;
	case DIVFREE_SPACE:
		height = width = sequence[0]->GetNumberOfTrueDofs(hdivform);
		break;
	}
}

void MLDivFree::Mult(const Vector & xconst, Vector & y) const
{
	elag_assert(xconst.Size() == height);
	elag_assert(y.Size() == width);
	Vector x( xconst.GetData(), xconst.Size() );

	if(!iterative_mode)
	{
		if(xconst.GetData() == y.GetData() )
		{
			x.SetSize(xconst.Size() );
			x = xconst;
		}
		y = 0.0;
	}

	elag_assert(y.CheckFinite() == 0);
	switch(my_cycle)
	{
	case FULL_SPACE:
		MGVcycle(x, y);
		break;
	case DIVFREE_SPACE:
		MGVcycleOnlyU(x, y);
		break;
	}
}

void MLDivFree::Mult(const MultiVector & xconst, MultiVector & y) const
{
	MultiVector x( xconst.GetData(), xconst.NumberOfVectors(), xconst.Size() );

	if(!iterative_mode)
	{
		if(xconst.GetData() == y.GetData() )
		{
			x.SetSizeAndNumberOfVectors(xconst.Size(), x.NumberOfVectors() );
			x = xconst;
		}
		y = 0.0;
	}

	Vector xi, yi;

	for(int i(0); i < x.NumberOfVectors(); ++i)
	{
		x.GetVectorView(i, xi);
		y.GetVectorView(i, yi);
		switch(my_cycle)
		{
		case FULL_SPACE:
			MGVcycle(xi, yi);
			break;
		case DIVFREE_SPACE:
			MGVcycleOnlyU(xi, yi);
			break;
		}

	}

}


void MLDivFree::MGVcycle(const Vector & x, Vector & y) const
{
//	std::fill(trueSol_data, trueSol_data+aritmeticComplexity, 0.);
	elag_assert(x.CheckFinite() == 0);
	int i(0);
	Vector rhs, rhs_coarse, sol, sol_coarse;
	Vector correction(trueAl[0]->Height());

	//Step 1, go fine to coarse
	{
		rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i], levelTrueStart[i+1]-levelTrueStart[i]);
		trueAl[0]->Mult(y, rhs);
		add(1., x, -1., rhs,rhs);
		for( ; i < nLevels-1; ++i)
		{
			sol.SetDataAndSize(trueSol_data+levelTrueStart[i], levelTrueStart[i+1]-levelTrueStart[i]);
			sol = 0.0;
			rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i], levelTrueStart[i+1]-levelTrueStart[i]);
			//subdomainSmoother will always use the value of sol
			subdomainSmoother(i, rhs, sol);
			//nullSpaceSmoother will always use the value of sol
			nullSpaceSmoother(i, rhs, sol);
			correction.SetSize(levelTrueStart[i+1]-levelTrueStart[i]);
			trueAl[i]->Mult(sol, correction);
			add(1., rhs, -1., correction, rhs);
			rhs_coarse.SetDataAndSize(trueRhs_data+levelTrueStart[i+1], levelTrueStart[i+2]-levelTrueStart[i+1]);
			trueP[i]->MultTranspose(rhs, rhs_coarse);
		}
	}

	//Step 2, solve coarse grid problem
	{
		sol.SetDataAndSize(trueSol_data+levelTrueStart[i], levelTrueStart[i+1]-levelTrueStart[i]);
		sol = 0.0;
		rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i], levelTrueStart[i+1]-levelTrueStart[i]);
		coarseSolver(rhs, sol);
	}

	//Step 3, go coarse to fine
	{
		for(i=nLevels-2; i >= 0; --i)
		{
			correction.SetSize(levelTrueStart[i+1]-levelTrueStart[i]);
			correction = 0.0;
			sol.SetDataAndSize(trueSol_data+levelTrueStart[i], levelTrueStart[i+1]-levelTrueStart[i]);
			sol_coarse.SetDataAndSize(trueSol_data+levelTrueStart[i+1], levelTrueStart[i+2]-levelTrueStart[i+1]);
			rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i], levelTrueStart[i+1]-levelTrueStart[i]);
			trueP[i]->Mult(sol_coarse, correction);
			//nullSpaceSmoother will always use the value of corr
			nullSpaceSmoother(i, rhs, correction);
			//subdomainSmoother will always use the value of corr
			subdomainSmoother(i, rhs, correction);
			sol.Add(1., correction);
		}
	}

	y.Add(1., sol);
}

void MLDivFree::MGVcycleOnlyU(const Vector & x, Vector & y) const
{
	int i(0);
	Vector rhs, rhs_coarse, sol, sol_coarse;
	Vector correction(trueAl[0]->GetBlock(0,0).Height());

	//Step 1, go fine to coarse
	{
		rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i], levelTrueStartMultiplier[i]-levelTrueStart[i]);
		trueAl[0]->GetBlock(0,0).Mult(y, rhs);
		add(1., x, -1., rhs, rhs);
		for( ; i < nLevels-1; ++i)
		{
			sol.SetDataAndSize(trueSol_data+levelTrueStart[i], levelTrueStartMultiplier[i]-levelTrueStart[i]);
			sol = 0.0;
			rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i], levelTrueStartMultiplier[i]-levelTrueStart[i]);
			//nullSpaceSmoother will always use the value of sol
			nullSpaceSmoother(i, rhs, sol);
			correction.SetSize(levelTrueStartMultiplier[i]-levelTrueStart[i]);
			trueAl[i]->GetBlock(0,0).Mult(sol, correction);
			add(1., rhs, -1., correction, rhs);
			rhs_coarse.SetDataAndSize(trueRhs_data+levelTrueStart[i+1], levelTrueStartMultiplier[i+1]-levelTrueStart[i+1]);
			trueP[i]->GetBlock(0,0).MultTranspose(rhs, rhs_coarse);
		}
	}

	//Step 2, solve fine grid problem
	{
		sol.SetDataAndSize(trueSol_data+levelTrueStart[i], levelTrueStart[i+1]-levelTrueStart[i]);
		sol = 0.0;
		rhs.SetDataAndSize(trueRhs_data+levelTrueStartMultiplier[i], levelTrueStart[i+1]-levelTrueStartMultiplier[i]);
		rhs = 0.0;
		rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i], levelTrueStart[i+1]-levelTrueStart[i]);
		coarseSolver(rhs, sol);
	}

	//Step 3, go coarse to fine
	{
		for(i=nLevels-2; i >= 0; --i)
		{
			correction.SetSize(levelTrueStartMultiplier[i]-levelTrueStart[i]);
			correction = 0.0;
			sol.SetDataAndSize(trueSol_data+levelTrueStart[i], levelTrueStartMultiplier[i]-levelTrueStart[i]);
			sol_coarse.SetDataAndSize(trueSol_data+levelTrueStart[i+1], levelTrueStartMultiplier[i+1]-levelTrueStart[i+1]);
			rhs.SetDataAndSize(trueRhs_data+levelTrueStart[i], levelTrueStartMultiplier[i]-levelTrueStart[i]);
			trueP[i]->GetBlock(0,0).Mult(sol_coarse, correction);
			//nullSpaceSmoother will always use the value of corr
			nullSpaceSmoother(i, rhs, correction);
			sol.Add(1., correction);
		}
	}

	y.Add(1., sol);

}

//subdomainSmoother will always use the value of sol
void MLDivFree::subdomainSmoother(int i, const Vector & trueRhs, Vector & trueSol) const
{
	elag_assert( trueRhs.CheckFinite() == 0 );
	elag_assert( trueSol.CheckFinite() == 0 );

	Array<int> true_offsets(3);
	true_offsets[0] = 0;
	true_offsets[1] = levelTrueStartMultiplier[i] - levelTrueStart[i];
	true_offsets[2] = levelTrueStart[i+1] - levelTrueStart[i];

	Array<int> offsets(3);
	offsets[0] = 0;
	offsets[1] = levelStartMultiplier[i] - levelStart[i];
	offsets[2] = levelStart[i+1] - levelStart[i];

	BlockVector res(rhs_data, offsets), sol(sol_data, offsets);
	BlockVector tRes(trueRhs.GetData(), true_offsets);
	BlockVector tSol(trueSol.GetData(), true_offsets);

	SharingMap & mapl2 = sequence[i]->GetDofHandler(l2form)->GetDofTrueDof();
	SharingMap & maphdiv = sequence[i]->GetDofHandler(hdivform)->GetDofTrueDof();

	maphdiv.Distribute(tRes.GetBlock(0),res.GetBlock(0));
	mapl2.Distribute(tRes.GetBlock(1),res.GetBlock(1));
	maphdiv.Distribute(tSol.GetBlock(0),sol.GetBlock(0));
	mapl2.Distribute(tSol.GetBlock(1),sol.GetBlock(1));

	elag_assert( res.CheckFinite() == 0);
	elag_assert( sol.CheckFinite() == 0);

	Al[i]->AddMult(sol, res, -1.);
	Vector essnullspace, t;

	essnullspace.SetDataAndSize(const_cast<double*>(essnullspace_data+levelStartMultiplier[i]), levelStart[i+1]-levelStartMultiplier[i] );
	t.SetDataAndSize(const_cast<double*>(t_data+levelStartMultiplier[i]), levelStart[i+1]-levelStartMultiplier[i] );

	elag_assert( essnullspace.CheckFinite() == 0 );
	elag_assert( t.CheckFinite() == 0 );

	int nAE = AE_dof[i]->NumRows();

#pragma omp parallel default(shared)
	{
		Vector sol_loc, rhs_loc, essnullspace_loc, t_loc;
		Array<int> colMapper, loc_dof, loc_dof_p;

		colMapper.SetSize( Al[i]->NumCols() );
		colMapper = -1;
#pragma omp for schedule(guided) nowait
		for(int iAE = 0; iAE < nAE; ++iAE)
		{
			int nlocdof   = getLocalInternalDofs(i, iAE, loc_dof);
			int nlocdof_p = getLocalDofs(PRESSURE, i, iAE, loc_dof_p);

			if(nlocdof == 0)
			{
				std::cout << "AE " << iAE << " has 0 internal dofs! \n";
				mfem_error("");
			}

			BlockMatrix * Aloc = ExtractRowAndColumns(Al[i], loc_dof, loc_dof, colMapper);
			essnullspace.GetSubVector(loc_dof_p, essnullspace_loc);

			if( !isRankDeficient( Aloc->GetBlock(0,1), essnullspace_loc )  )
			{
				rhs_loc.SetSize(nlocdof);
				sol_loc.SetSize(nlocdof);
				res.GetSubVector(loc_dof, rhs_loc.GetData());
				elag_assert(rhs_loc.CheckFinite() == 0 );
                SymmetrizedUmfpack solver(*Aloc);
				solver.Mult(rhs_loc, sol_loc);

				if(sol_loc.CheckFinite() != 0)
				{
					essnullspace_loc.Print(std::cout, essnullspace_loc.Size());
					Aloc->PrintMatlab(std::cout);
					elag_error(1);
				}
				sol.AddElementVector(loc_dof, sol_loc.GetData() );
			}
			else
			{
				rhs_loc.SetSize(nlocdof+1);
				sol_loc.SetSize(nlocdof+1);
				res.GetSubVector(loc_dof, rhs_loc.GetData());
				rhs_loc(nlocdof) = 0.;

				t_loc.SetSize(nlocdof_p);
				t.GetSubVector(loc_dof_p, t_loc.GetData() );

				SparseMatrix * T = createSparseMatrixRepresentationOfScalarProduct(t_loc.GetData(), t_loc.Size());
                SparseMatrix * Tt= Transpose(*T);

                Array<int> a_offset(4);
                a_offset[0] = Aloc->RowOffsets()[0];
                a_offset[1] = Aloc->RowOffsets()[1];
                a_offset[2] = Aloc->RowOffsets()[2];
                a_offset[3] = Aloc->RowOffsets()[2]+1;
                BlockMatrix a_Aloc(a_offset);
                a_Aloc.SetBlock(0,0, const_cast<SparseMatrix *>(&(Aloc->GetBlock(0,0))) );
                a_Aloc.SetBlock(1,0, const_cast<SparseMatrix *>(&(Aloc->GetBlock(1,0))) );
                a_Aloc.SetBlock(0,1, const_cast<SparseMatrix *>(&(Aloc->GetBlock(0,1))) );
                a_Aloc.SetBlock(2,1, T);
                a_Aloc.SetBlock(1,2,Tt);
                SymmetrizedUmfpack solver(a_Aloc);
				solver.Mult(rhs_loc, sol_loc);
				elag_assert(sol_loc.CheckFinite() == 0 );
				sol.AddElementVector(loc_dof, sol_loc.GetData() );
                                
                delete Tt;
				destroySparseMatrixRepresentationOfScalarProduct(T);
			}

			delete Aloc;
		}
	}

	maphdiv.IgnoreNonLocal(sol.GetBlock(0),tSol.GetBlock(0));
	mapl2.IgnoreNonLocal(sol.GetBlock(1),tSol.GetBlock(1));

	elag_assert( tSol.CheckFinite() == 0);

}


//nullSpaceSmoother will always use the value of sol
void MLDivFree::nullSpaceSmoother(int i, const Vector & rhs, Vector & sol) const
{
	elag_assert( rhs.CheckFinite() == 0);
	Vector rrhs, rsol;
	rrhs.SetDataAndSize(rhs.GetData(), levelTrueStartMultiplier[i]-levelTrueStart[i]);
	rsol.SetDataAndSize(sol.GetData(), levelTrueStartMultiplier[i]-levelTrueStart[i]);
	Maux[i]->Mult(rrhs, rsol);

	elag_assert( sol.CheckFinite() == 0);
}

//coarse solver
void MLDivFree::coarseSolver(const Vector & rhs, Vector & sol) const
{
	int i = nLevels-1;
	BlockMatrix * Ac = Al[i];

	SharingMap & mapl2 = sequence[i]->GetDofHandler(l2form)->GetDofTrueDof();
	SharingMap & maphdiv = sequence[i]->GetDofHandler(hdivform)->GetDofTrueDof();

	Array<int> true_offsets(3);
	true_offsets[0] = 0;
	true_offsets[1] = levelTrueStartMultiplier[i] - levelTrueStart[i];
	true_offsets[2] = levelTrueStart[i+1] - levelTrueStart[i];

	HypreParMatrix * PAc00 = Assemble(maphdiv, Ac->GetBlock(0,0), maphdiv);
	HypreParMatrix * PAc10 = Assemble(mapl2  , Ac->GetBlock(1,0), maphdiv);
	HypreParMatrix * PAc01 = Assemble(maphdiv, Ac->GetBlock(0,1), mapl2  );

	BlockOperator cOp(true_offsets);
	cOp.owns_blocks = 1;
	cOp.SetBlock(0,0, PAc00);
	cOp.SetBlock(1,0, PAc10);
	cOp.SetBlock(0,1, PAc01);

	Vector d(PAc00->Height());
	PAc00->GetDiag(d);

	HypreParMatrix * invDBt = Assemble(maphdiv, Ac->GetBlock(0,1), mapl2  );
	invDBt->InvScaleRows(d);
	HypreParMatrix * S = ParMult(PAc10,invDBt);
	delete invDBt;

	BlockDiagonalPreconditioner cPr(true_offsets);
	cPr.owns_blocks = 1;
	cPr.SetDiagonalBlock(0, new HypreSmoother(*PAc00) );
	cPr.SetDiagonalBlock(1, new HypreExtension::HypreBoomerAMG(*S)    );

	MINRESSolver sc(maphdiv.GetComm());
	sc.SetMaxIter(1000);
	sc.SetPrintLevel(0);
	sc.SetAbsTol(1e-20);
	sc.SetRelTol(1e-8);
	sc.SetOperator(cOp);
	sc.SetPreconditioner(cPr);
	sol = 0.0;
	elag_assert( rhs.CheckFinite() == 0 );
	sc.Mult(rhs,sol);

	delete S;

/*
	Vector essnullspace, t;
	essnullspace.SetDataAndSize(essnullspace_data+levelTrueStartMultiplier[i], levelTrueStart[i+1]-levelTrueStartMultiplier[i] );
	t.SetDataAndSize(t_data+levelTrueStartMultiplier[i], levelTrueStart[i+1]-levelTrueStartMultiplier[i] );
*/

}

void MLDivFree::computeSharedDof(int ilevel, Array<int> & is_shared)
{
	elag_assert(AE_dof[ilevel]->NumCols() == is_shared.Size() );

	is_shared = 0;

	int nAE = AE_dof[ilevel]->NumRows();

	Array<int> cols;
	Vector svec;

	int hdivform = sequence[ilevel]->GetNumberOfForms() - 2;
	SharingMap &map = sequence[ilevel]->GetDofHandler(hdivform)->GetDofTrueDof();

	const Array<int> & proc_shared_dof = map.SharedEntitiesId();

	for(const int * it = proc_shared_dof.GetData(); it != proc_shared_dof.GetData()+proc_shared_dof.Size(); ++it)
		++is_shared[*it];

	for(int iAE(0); iAE < nAE; ++iAE)
	{
		AE_dof[ilevel]->GetRow(iAE, cols, svec);
		for(int * it = cols.GetData(), * end = cols.GetData()+cols.Size(); it != end; ++it)
			++is_shared[*it];
	}

	for(int *it = is_shared.GetData(), *end = is_shared.GetData()+is_shared.Size(); it != end; ++it)
		(*it) = (*it > 1);

}

int MLDivFree::getLocalInternalDofs(int ilevel, int iAE, Array<int> & loc_dof) const
{
	Vector val;
	Array<int> dofs;
	AE_dof[ilevel]->GetRow(iAE,dofs,val);
	loc_dof.SetSize(dofs.Size());
	int * idof = loc_dof.GetData();

	int nlocdof = 0;

	const Array<int> is_shared(const_cast<int *>(dof_is_shared_among_AE_data+levelStart[ilevel]), levelStart[ilevel+1] - levelStart[ilevel]);

	for(int * it = dofs.GetData(), * end = dofs.GetData()+dofs.Size(); it != end; ++it)
	{
		if(!is_shared[*it])
		{
			*(idof++) = *it;
			++nlocdof;
		}
	}
	loc_dof.SetSize(nlocdof);
	return nlocdof;
}

int MLDivFree::getLocalInternalDofs(int comp, int ilevel, int iAE, Array<int> & loc_dof) const
{
	int * i_AE_dof = AE_dof[ilevel]->GetBlock(0, comp).GetI();
	int * j_AE_dof = AE_dof[ilevel]->GetBlock(0, comp).GetJ();
	int * comp_offset = AE_dof[ilevel]->ColOffsets().GetData();

	int * start = j_AE_dof + i_AE_dof[iAE];
	int * end   = j_AE_dof + i_AE_dof[iAE+1];

	loc_dof.SetSize(i_AE_dof[iAE+1] - i_AE_dof[iAE] );
	int * idof = loc_dof.GetData();
	int nlocdof = 0;

	const Array<int> is_shared(const_cast<int *>(dof_is_shared_among_AE_data+levelStart[ilevel]+comp_offset[comp]), comp_offset[comp+1]-comp_offset[comp]);

	for(int * it = start; it != end; ++it)
	{
		if(!is_shared[*it])
		{
			*(idof++) = *it;
			++nlocdof;
		}
	}
	loc_dof.SetSize(nlocdof);
	return nlocdof;

}

int MLDivFree::getLocalDofs(int comp, int ilevel, int iAE, Array<int> & loc_dof) const
{
	int * i_AE_dof = AE_dof[ilevel]->GetBlock(0, comp).GetI();
	int * j_AE_dof = AE_dof[ilevel]->GetBlock(0, comp).GetJ();

	int * start = j_AE_dof + i_AE_dof[iAE];
	int len     = i_AE_dof[iAE+1] - i_AE_dof[iAE];

	loc_dof.MakeRef(start, len);

	return len;
}

int MLDivFree::isRankDeficient(const SparseMatrix & Bt, const Vector & x) const
{
	// Here we check the l_inf norm of y = Bt*x.
	// In practice we compute the mat-vect row by row and we stop if abs( y(i) ) > numerical_zero
	const int * i_Bt = Bt.GetI();
	const int * j_Bt = Bt.GetJ();
	const double * a_Bt = Bt.GetData();
	int nrows = Bt.Size();
	int ncols = Bt.Width();

	if(ncols != x.Size() )
		mfem_error("MLDivFree::isRankDeficient: ncols != x.Size()");

	double val(0);

	for(int irow = 0; irow < nrows; ++irow)
	{
		val = 0;
		for(int j = i_Bt[irow]; j < i_Bt[irow+1]; ++j)
			val += a_Bt[j] * x(j_Bt[j]);
		if(fabs(val) > numerical_zero)
			return 0;
	}

	return 1;
}

BlockMatrix * MLDivFree::PtAP(BlockMatrix * A, BlockMatrix * P) const
{
	int nBlocks = A->NumRowBlocks();

	BlockMatrix * Pt = Transpose(*P);
	BlockMatrix * Ac = new BlockMatrix( P->ColOffsets() );
	Ac->owns_blocks = 1;

	for(int iblock(0); iblock < nBlocks; ++iblock)
		for(int jblock(0); jblock < nBlocks; ++jblock)
		{
			if(!A->IsZeroBlock(iblock, jblock))
			{
				const SparseMatrix & Ptii = Pt->GetBlock(iblock,iblock);
				const SparseMatrix & Pjj = P->GetBlock(jblock,jblock);
				SparseMatrix * PtiiAij = mfem::Mult(Ptii, A->GetBlock(iblock, jblock));
				SparseMatrix * PtiiAijPjj = mfem::Mult(*PtiiAij, Pjj);
				Ac->SetBlock(iblock, jblock, PtiiAijPjj);
				delete PtiiAij;
			}
		}

	Ac->EliminateZeroRows();

	delete Pt;

	return Ac;

}
