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

#include <numeric>
#include <algorithm>
#include <fstream>
#include <limits>

#include "elag_amge.hpp"

#include "../linalg/MatrixUtils.hpp"
#include "../linalg/SubMatrixExtraction.hpp"
#include "../linalg/SVDCalculator.hpp"


DeRhamSequence::DeRhamSequence(AgglomeratedTopology * topo_, int nspaces_):
	comm(topo_->GetComm()),
	jformStart(topo_->Dimensions() - topo_->Codimensions()),
	topo(topo_),
	nforms(nspaces_),
	dof(nforms),
	dofAgg(nforms),
	D(nforms-1),
	M(nforms, nforms),
	targets(0),
	pvTraces(nforms),
	P(nforms),
	Pi(nforms),
	coarserSequence( static_cast<DeRhamSequence *>(NULL) ),
	finerSequence( static_cast<DeRhamSequence *>(NULL) ),
	tolSVD(1e-9),
	smallEntry( std::numeric_limits<double>::epsilon() )
{
	dof = static_cast<DofHandler*>(NULL);
	dofAgg = static_cast<DofAgglomeration*>(NULL);
	D = static_cast<SparseMatrix *>(NULL);
	M = static_cast<SparseMatrix *>(NULL);

	pvTraces = static_cast<Vector *>(NULL);
	P = static_cast<SparseMatrix *>(NULL);
	Pi  = static_cast<CochainProjector *>(NULL);

	DeRhamSequence_os << " SVD Tolerance: " << tolSVD << "\n";
	DeRhamSequence_os << " Small Entry: "   << smallEntry << "\n";
}

void DeRhamSequence::SetTargets(const Array<MultiVector *> & targets_)
{
	if(targets_.Size() != nforms)
		mfem_error("Size of target does not match ");

	//(1) We copy the targets
	targets.SetSize(nforms);
	targets = static_cast<MultiVector *>(NULL);
	for(int jform(jformStart); jform < nforms; ++jform)
	{
		if(targets_[jform] == NULL)
		{
			targets[jform] = new MultiVector(0, dof[jform]->GetNDofs());
		}
		else
		{
			if(targets_[jform]->Size() != dof[jform]->GetNDofs() )
				mfem_error("Target has wrong size! :( ");

			targets[jform] = new MultiVector;
			targets_[jform]->Copy(*targets[jform]);
		}
	}
	// TODO check the exactness of the targets

}

DeRhamSequence * DeRhamSequence::Coarsen()
{
	coarserSequence = new DeRhamSequenceAlg(topo->coarserTopology, nforms);
	coarserSequence->jformStart = jformStart;
	coarserSequence->finerSequence = this;

	for(int codim = 0; codim < nforms; ++codim)
	{
		int jform = nforms-codim-1;

		if(jform < jformStart)
			break;

		dofAgg[jform] = new DofAgglomeration(topo, dof[jform]);
		coarserSequence->dof[jform] = new DofHandlerALG(codim, topo->coarserTopology);

		P[jform] = new SparseMatrix( dof[jform]->GetNDofs(), extimateUpperBoundNCoarseDof(jform) );
		Pi[jform] = new CochainProjector(coarserSequence->topo, coarserSequence->dof[jform],dofAgg[jform], P[jform]);


#ifdef ELAG_DEBUG
		dofAgg[jform]->CheckAdof();
#endif

		computeCoarseTraces(jform);

		if(codim > 0)
		{
			hFacetExtension(jform);
			if(codim > 1)
			{
				hRidgeExtension(jform);
				if(codim > 2)
					hPeakExtension(jform);
			}

			// The projector for jform can be finalized now.
			// Also dof[jform] is finalized;
			P[jform]->SetWidth();
			P[jform]->Finalize();
			coarserSequence->D[jform]->SetWidth(P[jform]->Width());
			coarserSequence->D[jform]->Finalize();
		}
		else
		{
			P[jform]->SetWidth();
			P[jform]->Finalize();
		}

		Pi[jform]->ComputeProjector();
		if(jform == nforms-1)
		{
			coarserSequence->dof[jform]->GetDofTrueDof().SetUp(
					Pi[jform]->GetProjectorMatrix(), dof[jform]->GetDofTrueDof(), *(P[jform]) );
		}
		else
		{
			SparseMatrix * unextendedP  = getUnextendedInterpolator(jform);
			SparseMatrix * incompletePi = Pi[jform]->GetIncompleteProjector();
//			HypreExtension::PrintAsHypreParCSRMatrix(topo->GetComm(), *unextendedP, "uP");
//			HypreExtension::PrintAsHypreParCSRMatrix(topo->GetComm(), *incompletePi, "iPi");
			coarserSequence->dof[jform]->GetDofTrueDof().SetUp(*incompletePi, dof[jform]->GetDofTrueDof(), *unextendedP);
			delete unextendedP;
			delete incompletePi;
		}

#ifdef ELAG_DEBUG
		if(!coarserSequence->dof[jform]->Finalized())
		{
			std::cout<<" Coarser Sequence Dof Handler " << jform << " is not finalized! \n";
			mfem_error();
		}

		if( coarserSequence->dof[jform]->GetNDofs() != P[jform]->Width() )
			mfem_error("coarserSequence->dof[jform]->GetNDofs() != P[jform]->Width()");
#endif

	}

	//(1) We coarsen the targets
	coarserSequence->targets.SetSize(nforms);
	coarserSequence->targets = static_cast<MultiVector *>(NULL);
	for(int jform(jformStart); jform < nforms; ++jform)
	{
		coarserSequence->targets[jform] = new MultiVector(targets[jform]->NumberOfVectors(), P[jform]->Width());
		Pi[jform]->Project( *(targets[jform]) , *(coarserSequence->targets[jform]));
	}

	return coarserSequence;
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
	for(int jform(jformStart); jform < nforms-1; ++jform)
	{
		if(D[jform]->NumNonZeroElems() == 0 || D[jform]->MaxNorm() < 1e-6 )
		{
			std::cout << "nnz(D_" << jform << ") =  " << D[jform]->NumNonZeroElems() << "\n";
			std::cout << "maxNorm(D_" << jform << ") =  " << D[jform]->MaxNorm() << "\n";
			mfem_error();
		}
	}
	for(int jform(jformStart); jform < nforms-2; ++jform)
	{
		SparseMatrix * DD = Mult(*D[jform+1], *D[jform]);

		double err = DD->MaxNorm();
		if( err > 1e-9 )
			DeRhamSequence_os << "|| D_"<<jform+1<< " * D_"<< jform << " || = "<<err << std::endl;

		elag_assert(err <= 1e-9);

		delete DD;
	}
}

void DeRhamSequence::CheckTrueD()
{
	Array<HypreParMatrix * > trueD(nforms-1);

	for(int jform(jformStart); jform < nforms-1; ++jform)
		trueD[jform] = ComputeTrueDerivativeOperator(jform);

	for(int jform(jformStart); jform < nforms-2; ++jform)
	{
		HypreParMatrix * DD = ParMult(trueD[jform+1], trueD[jform]);

		double err = hypre_ParCSRMatrixNormlinf(*DD);
		if( err > 1e-9 )
			DeRhamSequence_os << "|| D_"<<jform+1<< " * D_"<< jform << " || = "<< err << std::endl;

		elag_assert(err <= 1e-9);

		delete DD;
	}

	for(int jform(jformStart); jform < nforms-1; ++jform)
		delete trueD[jform];
}

void DeRhamSequence::CheckDP()
{
	if(coarserSequence)
	{
		for(int jform(jformStart); jform < nforms-1; ++jform)
		{
			SparseMatrix * Pu = P[jform];
			SparseMatrix * Pp = P[jform+1];
			SparseMatrix * Dfine = D[jform];
			SparseMatrix * Dcoarse = coarserSequence->D[jform];

			SparseMatrix * DfPu = Mult(*Dfine, *Pu);
			SparseMatrix * PpDc = Mult(*Pp, *Dcoarse);

			std::stringstream name1, name2;
			name1 << "D_{"<<jform<<",fine}*P_"<<jform;
			name2 << "P_"<<jform+1<<"* D_{"<<jform<<",coarse}";

			bool out = AreAlmostEqual(*DfPu, *PpDc, name1.str(), name2.str(), 1.e-6);
			elag_assert(out);

			delete PpDc;
			delete DfPu;

		}
	}
}

void DeRhamSequence::CheckTrueDP()
{
	if(coarserSequence)
	{
		for(int jform(jformStart); jform < nforms-1; ++jform)
		{
			HypreParMatrix * Pu = ComputeTrueP(jform);
			HypreParMatrix * Pp = ComputeTrueP(jform+1);
			HypreParMatrix * Dfine = ComputeTrueDerivativeOperator(jform);
			HypreParMatrix * Dcoarse = coarserSequence->ComputeTrueDerivativeOperator(jform);

			HypreParMatrix * DfPu = ParMult(Dfine, Pu);
			HypreParMatrix * PpDc = ParMult(Pp, Dcoarse);

			HYPRE_Int ierr = hypre_ParCSRMatrixCompare(*DfPu, *PpDc, 1e-9, 1);

			elag_assert(ierr == 0);

			delete PpDc;
			delete DfPu;
			delete Pu;
			delete Pp;
			delete Dfine;
			delete Dcoarse;

		}
	}
}

void DeRhamSequence::CheckCoarseMassMatrix()
{
	if(coarserSequence)
	{
		for(int jform(jformStart); jform < nforms; ++jform)
		{
			SparseMatrix * Mfine   = ComputeMassOperator(jform);
			SparseMatrix * Mcoarse = coarserSequence->ComputeMassOperator(jform);

			SparseMatrix * Pj = P[jform];
			SparseMatrix * Pt = Transpose(*Pj);

			SparseMatrix * tmp = Mult(*Pt, *Mfine);
			SparseMatrix * Mrap = Mult(*tmp, *Pj);

			std::stringstream name1, name2;
			name1 << "Mcoarse_"<<jform;
			name2 << "Mrap_"<<jform;

			AreAlmostEqual(*Mcoarse, *Mrap, name1.str(), name2.str(), 1.e-6, false);

			delete Mrap;
			delete tmp;
			delete Pt;
			delete Mcoarse;
			delete Mfine;

		}
	}
}

void DeRhamSequence::CheckTrueCoarseMassMatrix()
{
	if(coarserSequence)
	{
		for(int jform(jformStart); jform < nforms; ++jform)
		{
			HypreParMatrix * Mfine   = ComputeTrueMassOperator(jform);
			HypreParMatrix * Mcoarse = coarserSequence->ComputeTrueMassOperator(jform);

			HypreParMatrix * Pj = this->ComputeTrueP(jform);

			HypreParMatrix * Mrap = RAP(Mfine, Pj);

			int ierr = hypre_ParCSRMatrixCompare(*Mcoarse, *Mrap, 1e-9, 1);
			elag_assert( ierr == 0 );

			delete Mrap;
			delete Mcoarse;
			delete Mfine;
			delete Pj;

		}
	}
}

void DeRhamSequence::CheckCoarseDerivativeMatrix()
{

#ifdef SLOW_METHOD
	if(coarserSequence)
	{
		for(int jform(jformStart); jform < nforms - 1; ++jform)
		{
			SparseMatrix * Df = D[jform];
			SparseMatrix * Dc = coarserSequence->D[jform];

			SparseMatrix * Pu = P[jform];
			CochainProjector * Pip = Pi[jform+1];

			SparseMatrix * DfPu = Mult(*Df, *Pu);

			int dim_fine_u = Df->Width();
			int dim_fine_p = Df->Size();

			int dim_coarse_u = Dc->Width();
			int dim_coarse_p = Dc->Size();

			MultiVector id(dim_coarse_u, dim_coarse_u);
			id = 0.0;
			for(int i(0); i < dim_coarse_u; ++i)
				id.GetDataFromVector(i)[i] = 1.;

			MultiVector DfPuDense(dim_coarse_u, dim_fine_p);
			MatrixTimesMultiVector(*DfPu, id, DfPuDense);

			MultiVector DcDense(dim_coarse_u, dim_coarse_p);
			Pip->Project(DfPuDense, DcDense);

			DenseMatrix DcDenseAsMat(DcDense.GetData(), dim_coarse_p, dim_coarse_u);

			AddMatrix(1., DcDenseAsMat, -1., *Dc, DcDenseAsMat);

			double err = DcDenseAsMat.MaxMaxNorm();

			if(err > tolSVD)
				DeRhamSequence_os << "i = " << jform << " || Dc - Pi Df P ||_max_max = " << err << "\n";

			DcDenseAsMat.ClearExternalData();

			delete DfPu;
		}
	}
#endif
	if(coarserSequence)
	{
		for(int jform(jformStart); jform < nforms - 1; ++jform)
		{
			SparseMatrix * Df = D[jform];
			SparseMatrix * Dc = coarserSequence->D[jform];
			SparseMatrix * Pu = P[jform];
			SparseMatrix & Pip = const_cast<SparseMatrix&>( Pi[jform+1]->GetProjectorMatrix() );
			SparseMatrix * DfPu = Mult(*Df, *Pu);
			SparseMatrix * Dc2  = Mult(Pip, *DfPu);

			std::stringstream name1;			name1 <<"coarseD"<<jform;
			std::stringstream name2;			name2 <<"galCoarseD"<<jform;
			AreAlmostEqual(*Dc, *Dc2, name1.str(), name2.str(), 1e-9);

			delete DfPu;
			delete Dc2;
		}
	}
}

void DeRhamSequence::CheckTrueCoarseDerivativeMatrix()
{
	if(coarserSequence)
	{
		for(int jform(jformStart); jform < nforms - 1; ++jform)
		{
			HypreParMatrix * Df = ComputeTrueDerivativeOperator(jform);
			HypreParMatrix * Dc = coarserSequence->ComputeTrueDerivativeOperator(jform);
			HypreParMatrix * Pu = ComputeTrueP(jform);
			HypreParMatrix * Pip = ComputeTruePi(jform+1);
			HypreParMatrix * DfPu = ParMult(Df, Pu);
			HypreParMatrix * Dc2  = ParMult(Pip, DfPu);

			int ierr = hypre_ParCSRMatrixCompare(*Dc, *Dc2, 1e-9, 1);
			elag_assert( ierr == 0 );


			delete Dc2;
			delete DfPu;
			delete Pip;
			delete Pu;
			delete Dc;
			delete Df;
		}
	}
}


void DeRhamSequence::CheckPi()
{
	if(coarserSequence)
	{
		for(int jform(jformStart); jform < nforms; ++jform)
		{
			DeRhamSequence_os << "Check Pi of " << jform << "\n";

			//(1) check that || (I - Pi P) x || = 0 for random x
			Pi[jform]->CheckInvariants();


			//(2) check how close we are to the targets
			MultiVector ct(targets[jform]->NumberOfVectors(), P[jform]->Width());
			MultiVector res(targets[jform]->NumberOfVectors(), P[jform]->Size());
			Pi[jform]->Project( *(targets[jform]), ct, res );

			SparseMatrix * Mg = ComputeMassOperator(jform);
			WeightedInnerProduct dot(*Mg);
			Vector view;

			bool showres(false);
			for(int i(0); i < res.NumberOfVectors(); ++i)
			{
				res.GetVectorView(i, view);
				double view_dot_view = dot(view, view);

				if( view_dot_view > tolSVD)
					DeRhamSequence_os << "|| t - Pi t || = " << view_dot_view << "\n";

				if( view_dot_view > tolSVD)
					showres = true;
			}

			if( showres )
			{
				show(jform, res);
				ct = 0.0;
				Pi[jform]->Project( res, ct);

				if( ct.Normlinf() > tolSVD)
					DeRhamSequence_os << "|| Pi( I - P*Pi ) t || = " << ct.Normlinf() << "\n";

				if( jform != nforms-1)
				{
					MultiVector dres(res.NumberOfVectors(), GetNumberOfDofs(jform+1));
					MatrixTimesMultiVector(*(D[jform]), res, dres);

					if(dres.Normlinf() > tolSVD)
						DeRhamSequence_os << "|| D( I - P*Pi ) t || = " << dres.Normlinf() << "\n";
					show(jform+1, dres);
				}

				Array<int> all_bdr(topo->FacetBdrAttribute().Width()), dofMarker( dof[jform]->GetNDofs() );
				all_bdr = 1; dofMarker = 0;
				dof[jform]->MarkDofsOnSelectedBndr(all_bdr, dofMarker);

				Vector s( dof[jform]->GetNDofs() );
				std::copy(dofMarker.GetData(), dofMarker.GetData()+dof[jform]->GetNDofs(), s.GetData() );
				res.Scale(s);

				if(res.Normlinf() > tolSVD)
					DeRhamSequence_os << "|| [t - Pi t](boundary) ||_inf = " << res.Normlinf() << "\n";
			}

			delete Mg;
		}
	}
}

void DeRhamSequence::ComputeSpaceInterpolationError(int jform, const MultiVector & fineVector)
{
	if(finerSequence)
	{

		DeRhamSequence * seq = this;

		while(seq->finerSequence)
			seq = seq->finerSequence;

		int nVectors = fineVector.NumberOfVectors();
		int fSize = seq->dof[jform]->GetNDofs();
		int cSize = dof[jform]->GetNDofs();

		if(fineVector.Size() != fSize)
			mfem_error("DeRhamSequenceAO::ComputeSpaceInterpolationError #1");

		MultiVector * v = new MultiVector(nVectors, fSize );
		*v = fineVector;
		MultiVector *vhelp = new MultiVector(nVectors, seq->P[jform]->Width() );

		while(seq != this)
		{
			vhelp->SetSizeAndNumberOfVectors(seq->P[jform]->Width(), nVectors);
			seq->Pi[jform]->Project(*v, *vhelp);
			swap(vhelp, v);
			seq = seq->coarserSequence;
		}

		if(v->Size() != cSize)
			mfem_error("DeRhamSequenceAO::ComputeSpaceInterpolationError #2");

		while(seq->finerSequence)
		{
			seq = seq->finerSequence;
			vhelp->SetSizeAndNumberOfVectors(seq->P[jform]->Size(), nVectors);
			MatrixTimesMultiVector(*seq->P[jform], *v, *vhelp);
			swap(vhelp, v);
		}

		if(v->Size() != fSize)
			mfem_error("DeRhamSequenceAO::ComputeSpaceInterpolationError #3");

		MultiVector difference(nVectors, fSize);
		subtract(fineVector, *v, difference);

		SparseMatrix * Mg = seq->ComputeMassOperator(jform);

		Vector view_diff, view_fineV;
		Array<double> L2diff(nVectors), L2fineVector(nVectors);

		DeRhamSequence_os << std::setw(14) << cSize;
		for(int iVect(0); iVect < nVectors; ++iVect)
		{
			difference.GetVectorView(iVect, view_diff);
			L2diff[iVect] = Mg->InnerProduct( view_diff, view_diff);

			const_cast<MultiVector &>(fineVector).GetVectorView(iVect, view_fineV);
			L2fineVector[iVect] = Mg->InnerProduct(view_fineV, view_fineV);
			DeRhamSequence_os << std::setw(14) << sqrt( L2diff[iVect] / L2fineVector[iVect]);
		}
		delete Mg;

		if(jform < nforms - 1)
		{
			SparseMatrix * Wg = seq->ComputeMassOperator(jform+1);
			int wSize = seq->D[jform]->Size();
			MultiVector ddiff(nVectors, wSize), dv(nVectors, wSize);
			MatrixTimesMultiVector(*(seq->D[jform]), difference, ddiff);
			MatrixTimesMultiVector(*(seq->D[jform]), fineVector, dv);

			Vector view_diff, view_fineV;
			double L2ddiff, L2dv;

			for(int iVect(0); iVect < nVectors; ++iVect)
			{
				ddiff.GetVectorView(iVect, view_diff);
				L2ddiff = Wg->InnerProduct( view_diff, view_diff);

				dv.GetVectorView(iVect, view_fineV);
				L2dv = Wg->InnerProduct(view_fineV, view_fineV);
				if( fabs(L2fineVector[iVect]+L2dv) < 1e-9)
					L2dv = 1.;
				DeRhamSequence_os << std::setw(14) << sqrt( (L2diff[iVect] + L2ddiff)/( L2fineVector[iVect] + L2dv ) );
			}

			delete Wg;

		}

		delete v;
		delete vhelp;

		DeRhamSequence_os << std::endl;
	}

	if(coarserSequence)
		coarserSequence->ComputeSpaceInterpolationError(jform, fineVector);
}

HypreParMatrix * DeRhamSequence::ComputeTrueDerivativeOperator(int jform)
{
	return IgnoreNonLocalRange(dof[jform+1]->GetDofTrueDof(), *(D[jform]), dof[jform]->GetDofTrueDof());
}

HypreParMatrix * DeRhamSequence::ComputeTrueDerivativeOperator(int jform, Array<int> & ess_label)
{
	SerialCSRMatrix * Dbc = ComputeDerivativeOperator(jform, ess_label);
	HypreParMatrix * out = IgnoreNonLocalRange(dof[jform+1]->GetDofTrueDof(), *Dbc, dof[jform]->GetDofTrueDof());
	delete Dbc;
	return out;
}
HypreParMatrix * DeRhamSequence::ComputeTrueMassOperator(int jform)
{
	SerialCSRMatrix * Mj = ComputeMassOperator(jform);
	HypreParMatrix * out = Assemble(dof[jform]->GetDofTrueDof(), *Mj, dof[jform]->GetDofTrueDof());
	delete Mj;
	return out;
}
HypreParMatrix * DeRhamSequence::ComputeTrueMassOperator(int jform, Vector & elemMatrixScaling)
{
	SerialCSRMatrix * Mj = ComputeMassOperator(jform, elemMatrixScaling);
	HypreParMatrix * out = Assemble(dof[jform]->GetDofTrueDof(), *Mj, dof[jform]->GetDofTrueDof());
	delete Mj;
	return out;
}
HypreParMatrix * DeRhamSequence::ComputeTrueP(int jform)
{
	return IgnoreNonLocalRange(dof[jform]->GetDofTrueDof(), *(P[jform]), coarserSequence->dof[jform]->GetDofTrueDof());
}
HypreParMatrix * DeRhamSequence::ComputeTrueP(int jform, Array<int> & ess_label)
{
	SerialCSRMatrix * Pj = GetP(jform, ess_label);
	HypreParMatrix * out = IgnoreNonLocalRange(dof[jform]->GetDofTrueDof(), *Pj, coarserSequence->dof[jform]->GetDofTrueDof());
	delete Pj;
	return out;
}

HypreParMatrix * DeRhamSequence::ComputeTruePi(int jform)
{
	const SerialCSRMatrix & myPi = GetPi(jform)->GetProjectorMatrix();
	HypreParMatrix * out = IgnoreNonLocalRange(coarserSequence->dof[jform]->GetDofTrueDof(), const_cast<SerialCSRMatrix &>(myPi), dof[jform]->GetDofTrueDof());
	return out;
}


HypreParMatrix * DeRhamSequence::ComputeTrueProjectorFromH1ConformingSpace(int jform)
{
	elag_error_msg(NOT_IMPLEMENTED_YET, "DeRhamSequence::ComputeTrueProjectorFromH1ConformingSpace");
	SerialCSRMatrix * myPi = ComputeProjectorFromH1ConformingSpace(jform);
	SharingMap vectorH1(dof[jform]->GetDofTrueDof().GetComm());
//	vectorH1.SetUp(dof[jform]->GetDofTrueDof(), nforms-1);
	HypreParMatrix * out = IgnoreNonLocalRange(dof[jform]->GetDofTrueDof(), *myPi, vectorH1);
	delete myPi;
	return out;
}

void DeRhamSequence::ComputeTrueProjectorFromH1ConformingSpace(int jform, HypreParMatrix *& Pix, HypreParMatrix *& Piy, HypreParMatrix *& Piz)
{
	Array2D<SparseMatrix *> myPixyz(1,nforms-1);
	{
		SerialCSRMatrix * myPi = ComputeProjectorFromH1ConformingSpace(jform);
		ExtractComponents( *myPi, myPixyz, Ordering::byVDIM);
		delete myPi;
	}

	Pix = IgnoreNonLocalRange(dof[jform]->GetDofTrueDof(), *(myPixyz(0,0)), dof[0]->GetDofTrueDof());
	delete myPixyz(0,0);
	Piy = IgnoreNonLocalRange(dof[jform]->GetDofTrueDof(), *(myPixyz(0,1)), dof[0]->GetDofTrueDof());
	delete myPixyz(0,1);
	if(nforms == 3)
		Piz = NULL;
	else
	{
		Piz = IgnoreNonLocalRange(dof[jform]->GetDofTrueDof(), *(myPixyz(0,2)), dof[0]->GetDofTrueDof());
		delete myPixyz(0,2);
	}

}


SparseMatrix * DeRhamSequence::ComputeDerivativeOperator(int jform, Array<int> & ess_label)
{
	SparseMatrix * out = DeepCopy( *(D[jform]) );
	Array<int> marker(out->Width() );
	marker = 0;
	dof[jform]->MarkDofsOnSelectedBndr(ess_label, marker);
	out->EliminateCols(marker);
	return out;
}

SparseMatrix * DeRhamSequence::GetP(int jform, Array<int> & ess_label)
{
	SparseMatrix * out = DeepCopy( *(P[jform]) );
	Array<int> marker(out->Width() );
	marker = 0;
	coarserSequence->dof[jform]->MarkDofsOnSelectedBndr(ess_label, marker);
	out->EliminateCols(marker);
	return out;
}

#include <fstream>

void DeRhamSequence::DumpD()
{
	for(int jform(jformStart); jform < nforms-1; ++jform)
	{
		std::stringstream fname;
		fname << "D" << jform << ".mtx";
		std::ofstream fid( fname.str().c_str() );
		D[jform]->PrintMatlab( fid );
	}
}

void DeRhamSequence::DumpP()
{
	if(coarserSequence)
	{
		for(int jform(jformStart); jform < nforms; ++jform)
		{
			std::stringstream fname;
			fname << "P" << jform << ".mtx";
			std::ofstream fid( fname.str().c_str() );
			P[jform]->PrintMatlab( fid );
		}
	}
	else
	{
		std::cout << "This is already the coarser level \n";
	}
}

SparseMatrix * DeRhamSequence::ComputeLumpedMassOperator(int jform)
{
	SparseMatrix * out;
	if(jform == nforms-1)
		out =  ComputeMassOperator(jform);
	else
	{
		int ndof = dof[jform]->GetNDofs();
		out = diagonalMatrix(ndof);
		double * a = out->GetData();

		Array<int> rdof, gdof;
		DenseMatrix locmatrix;
		Vector locdiag;
		Vector evals;
		double eval_min;

		for(int i(0); i < topo->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT); ++i)
		{
			dof[jform]->GetrDof(AgglomeratedTopology::ELEMENT,i,rdof);
			dof[jform]->GetDofs(AgglomeratedTopology::ELEMENT,i,gdof);
			int nlocdof = rdof.Size();
			locmatrix.SetSize(nlocdof,nlocdof);
			M(AgglomeratedTopology::ELEMENT, jform)->GetSubMatrix(rdof,rdof,locmatrix);
			locmatrix.GetDiag(locdiag);
			locmatrix.InvSymmetricScaling(locdiag);
			locmatrix.Eigenvalues(evals);
			eval_min = evals.Min();
			for(int idof(0); idof < nlocdof; ++idof)
				a[gdof[idof]] += eval_min * locdiag(idof);

		}
	}
	return out;
}

SparseMatrix * DeRhamSequence::ComputeMassOperator(int jform, Vector & elemMatrixScaling)
{
	int nElements = topo->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT);

	if(elemMatrixScaling.Size() != nElements)
		mfem_error("SparseMatrix * DeRhamSequenceAO::ComputeMassOperator2 Pre#1\n");

	int nrows = M(AgglomeratedTopology::ELEMENT, jform)->Size();
	int ncols = M(AgglomeratedTopology::ELEMENT, jform)->Width();
	int nnz   = M(AgglomeratedTopology::ELEMENT, jform)->NumNonZeroElems();

	int * i = M(AgglomeratedTopology::ELEMENT, jform)->GetI();
	int * j = M(AgglomeratedTopology::ELEMENT, jform)->GetJ();
	double * data = new double[nnz];
	std::copy(M(AgglomeratedTopology::ELEMENT, jform)->GetData(), M(AgglomeratedTopology::ELEMENT, jform)->GetData()+nnz, data);

	SparseMatrix sM(i,j,data, nrows, ncols);
	{
		int nLocalDofs, irow(0);
		double s;

		for(int ie(0); ie < nElements; ++ie)
		{
			s = elemMatrixScaling(ie);
			nLocalDofs = dof[jform]->GetEntityNDof(AgglomeratedTopology::ELEMENT, ie);
			for(int iloc(0); iloc < nLocalDofs; ++iloc)
				sM.ScaleRow(irow++, s);
		}
	}

	SparseMatrix * out = Assemble(AgglomeratedTopology::ELEMENT, sM, *dof[jform], *dof[jform]);

	sM.LoseData();
	delete[] data;

	return out;
}

SparseMatrix * DeRhamSequence::ComputeLumpedMassOperator(int jform, Vector & elemMatrixScaling)
{
	SparseMatrix * out;
	if(jform == nforms-1)
		out =  ComputeMassOperator(jform, elemMatrixScaling);
	else
	{
		int ndof = dof[jform]->GetNDofs();
		out = diagonalMatrix(ndof);
		double * a = out->GetData();

		Array<int> rdof, gdof;
		DenseMatrix locmatrix;
		Vector locdiag;
		Vector evals;
		double eval_min;

		for(int i(0); i < topo->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT); ++i)
		{
			dof[jform]->GetrDof(AgglomeratedTopology::ELEMENT,i,rdof);
			dof[jform]->GetDofs(AgglomeratedTopology::ELEMENT,i,gdof);
			int nlocdof = rdof.Size();

			locmatrix.SetSize(nlocdof,nlocdof);
			M(AgglomeratedTopology::ELEMENT, jform)->GetSubMatrix(rdof,rdof,locmatrix);
			locmatrix.GetDiag(locdiag);
			locmatrix.InvSymmetricScaling(locdiag);
			locmatrix.Eigenvalues(evals);
			eval_min = evals.Min() * elemMatrixScaling(i);

			for(int idof(0); idof < nlocdof; ++idof)
				a[gdof[idof]] += eval_min * locdiag(idof);
		}
	}

	return out;
}

DeRhamSequence::~DeRhamSequence()
{
	for(int jform(0); jform < nforms; ++jform)
		delete dofAgg[jform];

	for(int i(0); i < dof.Size(); ++i)
		delete dof[i];

	for(int i(0); i < D.Size(); ++i)
		delete D[i];

	for(int i(0); i < P.Size(); ++i)
		delete P[i];

	for(int i(0); i < Pi.Size(); ++i)
		delete Pi[i];

	for(int i(0); i < targets.Size(); ++i)
		delete targets[i];

	for(int i(0); i < pvTraces.Size(); ++i)
		delete pvTraces[i];

	for(int icodim(0); icodim < M.NumRows(); ++icodim)
		for(int jform(0); jform < M.NumCols()-icodim; ++jform)
			delete M(icodim,jform);

//	delete coarserSequence;
}

int DeRhamSequence::extimateUpperBoundNCoarseDof(int jform)
{
	int nDimensions = topo->Dimensions();

	int ntargets, ndtargets;

	ntargets = targets[jform]->NumberOfVectors();
	ndtargets = (jform < nDimensions) ? targets[jform+1]->NumberOfVectors() : 0;

	if(nDimensions == 3)
	{

		int nv  = coarserSequence->topo->GetNumberLocalEntities(AgglomeratedTopology::PEAK);
		int ned = coarserSequence->topo->GetNumberLocalEntities(AgglomeratedTopology::RIDGE);
		int nf  = coarserSequence->topo->GetNumberLocalEntities(AgglomeratedTopology::FACET);
		int nel = coarserSequence->topo->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT);


		switch(jform)
		{
		case 0:
			return nv + (ned+nf+nel) * ndtargets;	//If the gradient of the H1targets is a subset of the Hcurl Targets,
			                                                        //then I have 1 dof for each vertex, and one dof for
			                                                        // each Hcurl target with 0 curl.
			break;
		case 1:
			return ned * (ntargets+1) + (nf+nel) * (ntargets+ndtargets);
			break;
		case 2:
			return nf*(ntargets+1) + nel*(ntargets+ndtargets);
			break;
		case 3:
			return nel*(ntargets+1);
			break;
		default:
			mfem_error("Not a valid form");
			return -1;
		}

	}
	else //nDimensions == 2
	{
		int nv = coarserSequence->topo->GetNumberLocalEntities(AgglomeratedTopology::RIDGE);
		int ned  = coarserSequence->topo->GetNumberLocalEntities(AgglomeratedTopology::FACET);
		int nel = coarserSequence->topo->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT);

		switch(jform)
		{
		case 0:
			return nv + (ned+nel) * (ntargets+ndtargets);
			break;
		case 1:
                        if( int(ned * (ntargets+1) + nel * (ntargets+ndtargets)) < 0 )
                            mfem_error("Integer out of bound!");
			return ned * (ntargets+1) + nel * (ntargets+ndtargets);
			break;
		case 2:
			return nel*(ntargets+1);
			break;
		default:
			mfem_error("Not a valid form");
			return -1;
		}
	}
}

void DeRhamSequence::compute0formCoarseTraces()
{
	int jform(0);
	AgglomeratedTopology::EntityByCodim codim = static_cast<AgglomeratedTopology::EntityByCodim>(topo->Dimensions()-jform);
	const int nDofs = dof[jform]->GetNDofs();

	const SparseMatrix & AEntity_dof = dofAgg[jform]->GetAEntityDofTable(codim);
	const int * const i_AEntity_dof = AEntity_dof.GetI();
	const int * const j_AEntity_dof = AEntity_dof.GetJ();
	const int nAE   = AEntity_dof.Size();

	pvTraces[jform] = new Vector(nDofs);
	Vector & pv(*pvTraces[jform]);
	computePVTraces(codim, pv);

	DofHandlerALG * jdof = dynamic_cast<DofHandlerALG*>(coarserSequence->dof[jform]);

	if(jdof == NULL)
		mfem_error("DeRhamSequence::computeCoarseTraces");

	jdof->AllocateDofTypeArray( extimateUpperBoundNCoarseDof(jform) );

	int fdof;
	for(int iAE(0); iAE < nAE; ++iAE)
	{
		if(i_AEntity_dof[iAE+1] - i_AEntity_dof[iAE] != 1)
			mfem_error("DeRhamSequence::compute0formCoarseTraces");

		fdof = j_AEntity_dof[iAE];
		P[jform]->Set(fdof, iAE, 1.);
		jdof->SetDofType(iAE, DofHandlerALG::RangeTSpace);
		jdof->SetNumberOfInteriorDofsRangeTSpace(codim,iAE,1);
		DenseMatrix * subm = new DenseMatrix(1,1);
		(*subm) = 1.;
		Pi[jform]->SetDofFunctional(codim, iAE, subm);
	}

	jdof->BuildEntityDofTable(codim);

	int * i_m = new int[nAE+1];
	int * j_m = new int[nAE];
	double * a_m = new double[nAE];

	fillSparseIdentity(i_m, j_m, a_m, nAE);

	coarserSequence->M(codim, jform) = new SparseMatrix(i_m, j_m, a_m, nAE, nAE);

}

void DeRhamSequence::computeCoarseTraces(int jform)
{

	DeRhamSequence_os <<"Compute Coarse Traces " << jform << "\n";

	if(jform == 0)
	{
		compute0formCoarseTraces();
		return;
	}

	if( targets[jform] && targets[jform]->NumberOfVectors() )
		computeCoarseTracesWithTargets(jform);
	else
		computeCoarseTracesNoTargets(jform);


}
void DeRhamSequence::computeCoarseTracesNoTargets(int jform)
{
	if(jform == 0)
	{
		compute0formCoarseTraces();
		return;
	}

	AgglomeratedTopology::EntityByCodim codim = static_cast<AgglomeratedTopology::EntityByCodim>(topo->Dimensions()-jform);
	const int nDofs = dof[jform]->GetNDofs();

	const SparseMatrix & AEntity_dof = dofAgg[jform]->GetAEntityDofTable(codim);
	const int * const i_AEntity_dof = AEntity_dof.GetI();
	const int * const j_AEntity_dof = AEntity_dof.GetJ();
	const int nAE   = AEntity_dof.Size();

	pvTraces[jform] = new Vector(nDofs);
	Vector & pv(*pvTraces[jform]);
	computePVTraces(codim, pv);

	DofHandlerALG * jdof = dynamic_cast<DofHandlerALG*>(coarserSequence->dof[jform]);

	if(jdof == NULL)
		mfem_error("DeRhamSequence::computeCoarseTraces");

	jdof->AllocateDofTypeArray( extimateUpperBoundNCoarseDof(jform) );

	SparseMatrix * M_d = AssembleAgglomerateMatrix(codim, *(M(codim, jform)), dofAgg[jform], dofAgg[jform]);

	Array<int> rows;
	Vector local_pv_trace;
	MultiVector basis;

	int * Imass = new int[nAE+1];
	int * Jmass = new int[nAE];
	double * Amass = new double[nAE];

	for(int iAE(0); iAE < nAE; ++iAE)
	{
		int start    = i_AEntity_dof[iAE];
		int end      = i_AEntity_dof[iAE+1];
		int loc_size = end-start;

		for(const int * fdof = j_AEntity_dof+start; fdof != j_AEntity_dof+end; ++fdof)
			P[jform]->Set(*fdof, iAE, pv(*fdof));
		jdof->SetDofType(iAE, DofHandlerALG::RangeTSpace);
		jdof->SetNumberOfInteriorDofsRangeTSpace(codim, iAE, 1);
		jdof->SetNumberOfInteriorDofsNullSpace(codim, iAE, 0);

		rows.MakeRef(const_cast<int *>(j_AEntity_dof)+start, loc_size);
		pv.GetSubVector(rows, local_pv_trace);
		SparseMatrix * Mloc = ExtractSubMatrix(*M_d, start, end, start, end);

		double mass = Mloc->InnerProduct(local_pv_trace, local_pv_trace);

		Jmass[iAE] = Imass[iAE] = iAE;
		Amass[iAE] = mass;

		MultiVector basis(local_pv_trace.GetData(), 1, loc_size);
		Pi[jform]->CreateDofFunctional(codim, iAE, basis, *Mloc);

		delete Mloc;
	}
	Imass[nAE] = nAE;

	jdof->BuildEntityDofTable(codim);
	coarserSequence->M(codim, jform) = new SparseMatrix( Imass, Jmass, Amass, nAE, nAE);

	delete M_d;

}
void DeRhamSequence::computeCoarseTracesWithTargets_old(int jform)
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

	if(jform == 0)
	{
		compute0formCoarseTraces();
		return;
	}

	std::cout << "OLD \n";

	AgglomeratedTopology::EntityByCodim codim = static_cast<AgglomeratedTopology::EntityByCodim>(topo->Dimensions() - jform);

	const int nDofs = dof[jform]->GetNDofs();

	const SparseMatrix & AEntity_dof = dofAgg[jform]->GetAEntityDofTable(codim);
	const int * const i_AEntity_dof = AEntity_dof.GetI();
	const int * const j_AEntity_dof = AEntity_dof.GetJ();
	const int nAE   = AEntity_dof.Size();
	const int nTargets = targets[jform]->NumberOfVectors();
	const int max_nfDof_per_AE = AEntity_dof.MaxRowSize();

	pvTraces[jform] = new Vector(nDofs);
	Vector & pv(*(pvTraces[jform]));
	pv = 0.0;
	computePVTraces(codim, pv);

	SparseMatrix * M_d = AssembleAgglomerateMatrix(codim, *(M(codim, jform)), dofAgg[jform], dofAgg[jform]);
	MultiVector * t_d = dofAgg[jform]->DistributeGlobalMultiVector(codim, *(targets[jform]) );
	Vector * pv_d     = dofAgg[jform]->DistributeGlobalVector(     codim, pv                );
	Vector diagM(M_d->Size()), scaling(M_d->Size());
	M_d->GetDiag(diagM);
	std::transform(diagM.GetData(), diagM.GetData()+diagM.Size(), scaling.GetData(), ::sqrt );

#ifdef ELAG_DEBUG
	if( (nDofs != pv.Size()) || (nDofs != targets[jform]->Size()) )
		mfem_error("computeCoarseTraces #1");

	if( M_d->Size() != t_d->Size() || pv_d->Size() != M_d->Size() )
		mfem_error("computeCoarseTraces #2");

	if( M_d->Size() != M_d->Width() )/*|| M_d->Size() != M_d->NumNonZeroElems() )*/
		mfem_error("computeCoarseTraces #3");

#endif

	MultiVector t_view;
	Vector pv_view, scaling_view, diagM_view;
	Vector s(nTargets);

	SVD_Calculator svd;
	svd.setFlagON();
	svd.AllocateOptimalSize(max_nfDof_per_AE, nTargets);

	DofHandlerALG * jdof = dynamic_cast<DofHandlerALG*>(coarserSequence->dof[jform]);

	if(jdof == NULL)
		mfem_error("DeRhamSequence::computeCoarseTraces");

	double s_max_tol;

	jdof->AllocateDofTypeArray( extimateUpperBoundNCoarseDof(jform) );

	int cdof_counter(0);
	for(int iAE(0); iAE < nAE; ++iAE)
	{
		int start    = i_AEntity_dof[iAE];
		int end      = i_AEntity_dof[iAE+1];
		int loc_size = end-start;
		t_d->GetRangeView(start, end, t_view);
		pv_view.SetDataAndSize(pv_d->GetData()+start, loc_size);
		diagM_view.SetDataAndSize(diagM.GetData()+start, loc_size);
		scaling_view.SetDataAndSize(scaling.GetData()+start, loc_size);
		SparseMatrix * Mloc = ExtractSubMatrix(*M_d, start, end, start, end);
		WeightedInnerProduct inner_product(*Mloc);
		Deflate(t_view, pv_view, inner_product);
		svd.ComputeON(scaling_view, t_view, s);

		s_max_tol = inner_product(pv_view, pv_view)*tolSVD;

		jdof->SetDofType(cdof_counter++, DofHandlerALG::RangeTSpace);
		jdof->SetNumberOfInteriorDofsRangeTSpace(codim, iAE, 1);

		int i(0);
		for(; i < s.Size(); ++i)
		{
			if(s(i) < s_max_tol)
				break;
			jdof->SetDofType(cdof_counter++, DofHandlerALG::NullSpace);
		}

		jdof->SetNumberOfInteriorDofsNullSpace(codim, iAE, i);

		delete Mloc;

	}

//	t_d->RoundSmallEntriesToZero(smallEntry);

	jdof->BuildEntityDofTable(codim);
	const SparseMatrix & AEntity_CDof = jdof->GetEntityDofTable(codim);
	const int * i_AEntity_CDof = AEntity_CDof.GetI();
	const int * const j_AEntity_CDof = AEntity_CDof.GetJ();

	int nCdof = jdof->GetNrDofs(codim);

	if(nCdof != cdof_counter)
		mfem_error("nCdof does not match!");

	ElementalMatricesContainer mass(nAE);
//	coarserSequence->M(codim, jform) = new SparseMatrix(nCdof, nCdof);

	Array<int> rows;
	Array<int> cols(nTargets+1);
	for(int iAE(0); iAE < nAE; ++iAE)
	{
		int start    = i_AEntity_dof[iAE];
		int end      = i_AEntity_dof[iAE+1];
		int col_start    = i_AEntity_CDof[iAE];
		int col_end      = i_AEntity_CDof[iAE+1];
		int loc_size = end-start;
		int nNullSpaceDofs = jdof->GetEntityNDof(codim, iAE)-1;

		SparseMatrix * Mloc = ExtractSubMatrix(*M_d, start, end, start, end);

		pv_view.SetDataAndSize(pv_d->GetData()+start, loc_size);
		diagM_view.SetDataAndSize(diagM.GetData()+start, loc_size);
		WeightedInnerProduct inner_product(*Mloc);

		double pv_dot_pv = inner_product(pv_view, pv_view);

		DenseMatrix alldata(loc_size, nNullSpaceDofs+1);

		DenseMatrix tdata(alldata.Data()+loc_size, loc_size, nNullSpaceDofs);
		t_d->CopyToDenseMatrix(0,nNullSpaceDofs,start,end, tdata);
		tdata *= sqrt(pv_dot_pv);

		Vector alldata_0;
		alldata.GetColumnReference(0, alldata_0);
		alldata_0.Set(1.,pv_view);

		rows.MakeRef(const_cast<int *>(j_AEntity_dof)+start, loc_size);
		cols.MakeRef(const_cast<int *>(j_AEntity_CDof) + col_start, col_end-col_start );

#if elemAGG_Debug
		if(rows.Max() >= P[jform]->Size() )
			mfem_error("rows.Max() >= P[jform]->Size()");

		if(cols.Max() >= P[jform]->Width() )
			mfem_error("cols.Max() >= P[jform]->Width()");
#endif
		P[jform]->SetSubMatrix(rows, cols, alldata);
		tdata.ClearExternalData();

		MultiVector allDataAsMV(alldata.Data(), alldata.Width(), loc_size);
		Pi[jform]->CreateDofFunctional(codim, iAE, allDataAsMV, *Mloc);
		DenseMatrix * cElemMass = new DenseMatrix(alldata.Width(), alldata.Width());
		inner_product(allDataAsMV, allDataAsMV, *cElemMass);
		cElemMass->Symmetrize();
		mass.SetElementalMatrix(iAE, cElemMass );



#if 1
		if(col_start != j_AEntity_CDof[col_start])
			mfem_error("CheckFailed col_start != j_AEntity_CDof[col_start]");
#endif

		delete Mloc;

	}

	DeRhamSequence_os << "*** Number of PV dofs        = " << nAE << "\n"<<
		     "*** Number of NullSpace Dofs = " << i_AEntity_CDof[nAE] - nAE << "\n";

	coarserSequence->M(codim, jform) = mass.GetAsSparseMatrix();

	delete M_d;
	delete t_d;
	delete pv_d;
}

// C = [a | b];
void concatenate(Vector & a, MultiVector & b, DenseMatrix & C)
{
	int nrow_a = a.Size();
	int nrow_b = b.Size();
	int nrow_C = C.Height();
	int ncol_b = b.NumberOfVectors();
	int ncol_C = C.Width();

	elag_assert(nrow_a == nrow_b);
	elag_assert(nrow_a == nrow_C);
	elag_assert(ncol_C <= ncol_b + 1);
	elag_assert(b.LeadingDimension() == nrow_b);

	double * a_data = a.GetData();
	double * C_data = C.Data();

	C_data = std::copy(a_data, a_data+nrow_a, C_data);
	if(b.LeadingDimension() == b.Size())
	{
		double * b_data = b.GetData();
		int hw = nrow_C * (ncol_C-1);
		if(hw > 0)
			std::copy(b_data, b_data + hw, C_data);
	}
	else
	{
		elag_error(1);
	}

}

void DeRhamSequence::computeCoarseTracesWithTargets(int jform)
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
	if(jform == 0)
	{
		compute0formCoarseTraces();
		return;
	}

	AgglomeratedTopology::EntityByCodim codim = static_cast<AgglomeratedTopology::EntityByCodim>(topo->Dimensions() - jform);

	SharingMap & entity_TrueEntity = topo->EntityTrueEntity(codim);
	SharingMap & AE_TrueAE = topo->coarserTopology->EntityTrueEntity(codim);
	SharingMap & dof_TrueDof = dof[jform]->GetDofTrueDof();

	const int nDofs = dof[jform]->GetNDofs();

	const SparseMatrix & AEntity_dof = dofAgg[jform]->GetAEntityDofTable(codim);
	const int * const i_AEntity_dof = AEntity_dof.GetI();
	const int * const j_AEntity_dof = AEntity_dof.GetJ();
	const int nAE   = AEntity_dof.Size();
	const int nTargets = targets[jform]->NumberOfVectors();
	const int max_nfDof_per_AE = AEntity_dof.MaxRowSize();

	pvTraces[jform] = new Vector(nDofs);
	Vector & pv(*(pvTraces[jform]));
	pv = 0.0;
	computePVTraces(codim, pv);

	SparseMatrix * M_d = AssembleAgglomerateMatrix(codim, *(M(codim, jform)), dofAgg[jform], dofAgg[jform]);
	Vector diagM(M_d->Size()), scaling(M_d->Size());
	M_d->GetDiag(diagM);
	std::transform(diagM.GetData(), diagM.GetData()+diagM.Size(), scaling.GetData(), ::sqrt );

#ifdef ELAG_DEBUG
	if( (nDofs != pv.Size()) || (nDofs != targets[jform]->Size()) )
		mfem_error("computeCoarseTraces #1");

	if( M_d->Size() != M_d->Width() )/*|| M_d->Size() != M_d->NumNonZeroElems() )*/
		mfem_error("computeCoarseTraces #3");

#endif

	ElementalMatricesContainer my_p(nAE);
	ElementalMatricesContainer mass(nAE);

	Vector scaling_view, diagM_view;
	Vector s(nTargets);

	SVD_Calculator svd;
	svd.setFlagON();
	svd.AllocateOptimalSize(max_nfDof_per_AE, nTargets);

	double s_max_tol;
	Array<int> AE_dofs_offsets(nAE+1);
	AE_dofs_offsets[0] = 0;

	//Temporary Variable will be overwritten later.
	int * AE_ndofs = AE_dofs_offsets.GetData() + 1;
	Vector loc_pv;
	MultiVector loc_targets;
	//========== FIRST LOOP: Compute SVDs and set AE_ndofs ========================//
	for(int iAE(0); iAE < nAE; ++iAE)
	{
		if( AE_TrueAE.IsShared(iAE) == -1 )	//If the entity is shared and it does not belong to this processor
			AE_ndofs[iAE] = 0;
		else	//If this processor owns the entity
		{
			int start    = i_AEntity_dof[iAE];
			int end      = i_AEntity_dof[iAE+1];
			int loc_size = end-start;
			Array<int> dof_in_AE(const_cast<int *>(j_AEntity_dof) + start, loc_size);

			loc_pv.SetSize(loc_size);
			loc_targets.SetSizeAndNumberOfVectors(loc_size, nTargets+1);

			targets[jform]->GetSubMultiVector(dof_in_AE, loc_targets);
			pv.GetSubVector(dof_in_AE, loc_pv);

			SparseMatrix * Mloc = ExtractSubMatrix(*M_d, start, end, start, end);
			WeightedInnerProduct inner_product(*Mloc);
			Deflate(loc_targets, loc_pv, inner_product);
			
			if( IsDiagonal(*Mloc) )
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
			
			s_max_tol = inner_product(loc_pv, loc_pv)*tolSVD;
			int i(0);
			for(; i < s.Size(); ++i)
			{
				if(s(i) < s_max_tol)
					break;
			}

			AE_ndofs[iAE] = i+1;

			double pv_dot_pv = inner_product(loc_pv, loc_pv);
			loc_targets *= sqrt(pv_dot_pv);

			DenseMatrix * p_loc = new DenseMatrix(loc_size, AE_ndofs[iAE]);
			concatenate(loc_pv, loc_targets, *p_loc);
			my_p.SetElementalMatrix(iAE, p_loc);

			DenseMatrix * cElemMass = new DenseMatrix(AE_ndofs[iAE], AE_ndofs[iAE]);
			inner_product(*p_loc, *p_loc, *cElemMass);
			cElemMass->Symmetrize();
#if ELAG_DEBUG
			{
				DenseMatrix tmp(*cElemMass);
				int n = tmp.Height();
				for(int i = 0; i < n; ++i)
					tmp(i,i) -= pv_dot_pv;

				int err = tmp.MaxMaxNorm();
				if( err > 1e-9 )
				{
					std::cout << "Form: " << jform << " iAE: " << iAE << "\n";
					std::cout << "IsShared: " << AE_TrueAE.IsShared(iAE) << "\n";

					std::cout<<"AE_ndofs: " << AE_ndofs[iAE] << "\n";
					s.Print(std::cout<<"SingularValues = \n", s.Size() );
					p_loc->PrintMatlab(std::cout<<"P_loc = \n");
					cElemMass->PrintMatlab(std::cout<<"cElemMass = \n");
					Mloc->PrintMatlab(std::cout<<"M_iAE = \n");
				}

				elag_assert(err < 1e-9 );
			}
#endif
			mass.SetElementalMatrix(iAE, cElemMass );

			MultiVector allDataAsMV(p_loc->Data(), p_loc->Width(), loc_size);
			Pi[jform]->CreateDofFunctional(codim, iAE, allDataAsMV, *Mloc);

			delete Mloc;
		}
	}

	{
		Array<int> tmp(AE_ndofs, nAE);
		AE_TrueAE.Synchronize(tmp);
	}

	DofHandlerALG * jdof = dynamic_cast<DofHandlerALG*>(coarserSequence->dof[jform]);
	if(jdof == NULL)
		mfem_error("DeRhamSequence::computeCoarseTraces");
	jdof->AllocateDofTypeArray( extimateUpperBoundNCoarseDof(jform) );
	int cdof_counter = 0;
	//========== SECOND LOOP: Build the entity_dof table ========================//
	for(int iAE(0); iAE < nAE; ++iAE)
	{
		jdof->SetDofType(cdof_counter++, DofHandlerALG::RangeTSpace);
		jdof->SetNumberOfInteriorDofsRangeTSpace(codim, iAE, 1);
		for(int i(0); i < AE_ndofs[iAE]-1; ++i)
			jdof->SetDofType(cdof_counter++, DofHandlerALG::NullSpace);
		jdof->SetNumberOfInteriorDofsNullSpace(codim, iAE, AE_ndofs[iAE]-1);
	}
	jdof->BuildEntityDofTable(codim);

	int nCdof = jdof->GetNrDofs(codim);
	if(nCdof != cdof_counter)
		mfem_error("nCdof does not match!");

	const SparseMatrix & AEntity_CDof = jdof->GetEntityDofTable(codim);
	const int * i_AEntity_CDof = AEntity_CDof.GetI();
	const int * const j_AEntity_CDof = AEntity_CDof.GetJ();

	//========== THIRD LOOP: Assemble P_partial ========================//
	Array<int> rows;
	Array<int> cols;
	SparseMatrix * P_partial = new SparseMatrix(nDofs, AEntity_CDof.Width());
	for(int iAE(0); iAE < nAE; ++iAE)
	{
		if( AE_TrueAE.IsShared(iAE) != -1 )
		{
			int start    = i_AEntity_dof[iAE];
			int end      = i_AEntity_dof[iAE+1];
			int loc_size = end-start;

			int col_start    = i_AEntity_CDof[iAE];
			int col_end      = i_AEntity_CDof[iAE+1];


			rows.MakeRef(const_cast<int *>(j_AEntity_dof)+start, loc_size);
			cols.MakeRef(const_cast<int *>(j_AEntity_CDof) + col_start, col_end-col_start );

	#if ELAG_DEBUG
			if(rows.Max() >= P[jform]->Size() )
				mfem_error("rows.Max() >= P[jform]->Size()");

			if(cols.Max() >= P[jform]->Width() )
				mfem_error("cols.Max() >= P[jform]->Width()");

			elag_assert( my_p.GetElementalMatrix(iAE).Height() == rows.Size() );
			elag_assert( my_p.GetElementalMatrix(iAE).Width() == cols.Size() );
	#endif
				P_partial->SetSubMatrix(rows, cols, my_p.GetElementalMatrix(iAE));
		}
	}
	P_partial->Finalize();

	SharingMap cdof_TrueCDof(comm);
    cdof_TrueCDof.SetUp(AE_TrueAE, const_cast<SparseMatrix&>(AEntity_CDof) );
    SparseMatrix * P_ok = AssembleNonLocal(dof_TrueDof, *P_partial, cdof_TrueCDof);

	AddOpenFormat(*P[jform], *P_ok);

	delete P_partial;
	delete P_ok;

	for(int iAE(0); iAE < nAE; ++iAE)
	{
		if( AE_TrueAE.IsShared(iAE) == -1 )	//If the entity is shared and it does not belong to this processor
		{
			int start    = i_AEntity_dof[iAE];
			int end      = i_AEntity_dof[iAE+1];
			int loc_size = end-start;
			int c_start    = i_AEntity_CDof[iAE];
			int c_end      = i_AEntity_CDof[iAE+1];
			int c_size = c_end-c_start;
			Array<int> dof_in_AE(const_cast<int *>(j_AEntity_dof) + start, loc_size);
			Array<int> cdof_in_AE(const_cast<int *>(j_AEntity_CDof) + c_start, c_size );

			SparseMatrix * Mloc = ExtractSubMatrix(*M_d, start, end, start, end);
			WeightedInnerProduct inner_product(*Mloc);

			elag_assert(AE_ndofs[iAE] == c_size);

			DenseMatrix p_loc(loc_size, AE_ndofs[iAE]);
			P[jform]->GetSubMatrix(dof_in_AE, cdof_in_AE, p_loc);

			DenseMatrix * cElemMass = new DenseMatrix(AE_ndofs[iAE], AE_ndofs[iAE]);
			inner_product(p_loc, p_loc, *cElemMass);
			cElemMass->Symmetrize();
			mass.SetElementalMatrix(iAE, cElemMass );

#if ELAG_DEBUG
			{
				DenseMatrix tmp(*cElemMass);
				int n = tmp.Height();
				double pv_dot_pv = tmp(0,0);
				for(int i = 0; i < n; ++i)
					tmp(i,i) -= pv_dot_pv;

				int err = tmp.MaxMaxNorm();
				if( err > 1e-9 )
				{
					std::cout << "Form: " << jform << " iAE: " << iAE << "\n";

					std::cout<<"AE_ndofs: " << AE_ndofs[iAE] << "\n";
					s.Print(std::cout<<"SingularValues = \n", s.Size() );
					p_loc.PrintMatlab(std::cout<<"P_loc = \n");
					cElemMass->PrintMatlab(std::cout<<"cElemMass = \n");
					Mloc->PrintMatlab(std::cout<<"M_iAE = \n");
				}

				elag_assert(err < 1e-9 );
			}
#endif

			MultiVector allDataAsMV(p_loc.Data(), p_loc.Width(), loc_size);
			Pi[jform]->CreateDofFunctional(codim, iAE, allDataAsMV, *Mloc);

			delete Mloc;
		}
	}


	DeRhamSequence_os << "*** Number of PV dofs        = " << nAE << "\n"<<
		     "*** Number of NullSpace Dofs = " << i_AEntity_CDof[nAE] - nAE << "\n";

	coarserSequence->M(codim, jform) = mass.GetAsSparseMatrix();

	delete M_d;
}

void DeRhamSequence::hFacetExtension(int jform)
{

	DeRhamSequence_os << "Enter hFacetExtension of " << jform << "\n";

	AgglomeratedTopology::EntityByCodim codim_bdr = static_cast<AgglomeratedTopology::EntityByCodim>(nforms - jform - 1);
	AgglomeratedTopology::EntityByCodim codim_dom = static_cast<AgglomeratedTopology::EntityByCodim>(nforms - jform - 2);

	DofHandlerALG * uCDofHandler = dynamic_cast<DofHandlerALG *>(coarserSequence->dof[jform]);
	DofHandlerALG * pCDofHandler = dynamic_cast<DofHandlerALG *>(coarserSequence->dof[jform+1]);

	if(uCDofHandler == NULL || pCDofHandler == 0)
		mfem_error("The coarse dof handlers are not of DofHandlerALG type");

	/*
	 * Here we do block solvers for the following systems
	 *
	 * 	[ A_ii   B_ai^T    0   ] [ u ]     [A_ib*R_d ]    [ A_ii   B_ai^T    0   ] [ u ]     [ 0 ]
	 * 	[ B_ai     0       T^T ] [ p ] = - [B_ab*R_d ],   [ B_ai     0       T^T ] [ p ] = - [ x ]
	 * 	[ 0        T       0   ] [ l ]     [ 0       ]    [ 0        T       0   ] [ l ]     [ 0 ]
	 *
	 *	where:
	 *
	 *	u is the sparse matrix representing the Projector operator between the coarse and fine ispace.
	 *	p is the sparse matrix of lagrangian multipliers (that can be ignored)
	 *	l is the sparse matrix representing the coarse derivative operator
	 *
	 *	A_ii is the sparse matrix discretizing (u,v) + <\der u, \der v> restricted to the interal dofs of each agglomerate
	 *	A_ib is the sparse rectangular matrix discretizing (u,v) + <\der u, \der v> whose rows are restricted to the interal dofs of each agglomerate,
	 *	     and the columns are restricted to the boundary dofs of each agglomerate.
	 *	B_ai is the sparse rectangular matrix discretizing <\der u, q> whose columns are restricted to the internal dofs of each agglomerate.
	 *	B_ab is the sparse rectangular matrix discretizing <\der u, q> whose columns are restricted to the bdr dofs of each agglomerate.
	 *	T is the sparse matrix discretizing <Q,q> where Q belongs to the coarse PV (jform+1) and q belongs to the fine (iform+1).
	 */
	//		my_os << "Harmonic extension for space " << jform << " on variety of codimension " << codim_dom << std::endl;
	// Agglomerate-based Mass Matrix (primal variable)
	SparseMatrix * M_d = AssembleAgglomerateMatrix(codim_dom, *(M(codim_dom, jform)), dofAgg[jform], dofAgg[jform]);
	// Agglomerate-based Derivative operator ( primal variable --> derivative )
	SparseMatrix * D_d = DistributeAgglomerateMatrix(codim_dom, *(D[jform]), dofAgg[jform+1], dofAgg[jform] );
	// Agglomerate-based Mass Matrix ( derivative )
	SparseMatrix * W_d = AssembleAgglomerateMatrix(codim_dom, *(M(codim_dom, jform+1)), dofAgg[jform+1], dofAgg[jform+1]);
	SparseMatrix * B_d = Mult(*W_d, *D_d);

	// Matrix Containing the lifting of the bc on the boundary
	SparseMatrix * R_t = TransposeAbstractSparseMatrix(*P[jform], 1);
	SparseMatrix * Dt = Transpose(*D[jform]);
	SparseMatrix * M_gd = Assemble(codim_dom, *M_d, dofAgg[jform],  NULL  );
	SparseMatrix * W_gd = Assemble(codim_dom, *W_d, dofAgg[jform+1], NULL);
	R_t->SetWidth( M_gd->Size() );


	SparseMatrix * minusRtM_d = Mult(*R_t, *M_gd);
	*minusRtM_d *= -1.0;
	R_t->SetWidth( Dt->Size() );
	SparseMatrix * minusRtDt = Mult(*R_t, *Dt);
	*minusRtDt *= -1.0;
	minusRtDt->SetWidth( W_gd->Size() );
	SparseMatrix * minusRtBt_d = Mult(*minusRtDt, *W_gd);

	delete minusRtDt;
	delete Dt;
	delete W_gd;
	delete M_gd;


	// hyperAgglomeratedElement_AFacetCDofs
	SparseMatrix * hAE_AFCD = Mult(coarserSequence->topo->B(codim_dom), const_cast<SparseMatrix &>(coarserSequence->dof[jform]->GetEntityDofTable(codim_bdr)) );
	int nhAE = hAE_AFCD->Size();
	const int * i_hAE_AFCD = hAE_AFCD->GetI();
	const int * j_hAE_AFCD = hAE_AFCD->GetJ();

	// SparseMatrix to find the global PV dof for jform+1.
	SparseMatrix * AE_PVdof = pCDofHandler->GetEntityRangeTSpaceDofTable( codim_dom );
	const int * const i_AE_PVdof = AE_PVdof->GetI();
	const int * const j_AE_PVdof = AE_PVdof->GetJ();
	
#ifdef ELAG_DEBUG
	{
		const int * it = i_AE_PVdof;
		for(int i = 0; i < nhAE; ++i, ++it)
			elag_assert(*it == i);
	}
#endif

	// SparseMatrix to find the global NullSpace dof for jform+1 (this new dofs of jform will go in RangeTSpace)
	SparseMatrix * AE_NullSpaceDof = pCDofHandler->GetEntityNullSpaceDofTable( codim_dom );
	const int * const i_AE_NullSpaceDof = AE_NullSpaceDof->GetI();
	const int * const j_AE_NullSpaceDof = AE_NullSpaceDof->GetJ();

	int uStart, uEnd, pStart, pEnd;
	int uBdrStart, uBdrEnd;
	int uAllStart, uAllEnd;

	MultiVector rhs, sol;
	int local_offsets[4];
	int nrhs_ext, nrhs_RangeT, nrhs_Null, nlocalbasis;
	MultiVector rhs_view_u,rhs_view_p,rhs_view_l;
	MultiVector sol_view_u,sol_view_p,sol_view_l;
	
	Array<int> coarseUDof_on_Bdr, coarseUDof_InternalRangeT, coarseUDof_InternalNull;	
	Array<int> coarsePDof;

	Array<int> fineUDof_Internal, fineUDof_Bdr, fineUDof, finePDof;

	int estimateNumberCoarseDofs(0);                                 // Upper bound on the number of coarse dofs,
	                                                                 // assuming that targets are linearly independent from all other types of
	                                                                 // dofs.
	estimateNumberCoarseDofs = R_t->Size()+P[jform+1]->Width();       // Traces dofs + Derivative dofs + number of AE
	estimateNumberCoarseDofs += nhAE*(targets[jform]->NumberOfVectors() - 1);   // Maximum allowed number of Bubble AE dofs - number of AE

	coarserSequence->D[jform] = new SparseMatrix( P[jform+1]->Width(), estimateNumberCoarseDofs );
	ElementalMatricesContainer mass(nhAE);

	DenseMatrix subm;
	Vector subv;

	int nNullSpaceDofs(0), nRangeTDofs(0);

	int coarseDofCounter(R_t->Size());	//The dofs we add are bubbles on the domain,
	                                    // so their global numbering is higher then the dof of the traces

	SVD_Calculator svd;
	svd.setFlagON();
	const int nTargets = targets[jform]->NumberOfVectors();
	const int max_nfDof_per_AE = dofAgg[jform  ]->GetAEntityDofTable( codim_dom ).MaxRowSize();
	if(nTargets)
		svd.AllocateOptimalSize(max_nfDof_per_AE, nTargets);
	Vector sv(nTargets);

	for(int iAE(0); iAE < nhAE; ++iAE)
	{
		// (1) Compute the offsets
		dofAgg[jform  ]->GetAgglomerateInternalDofRange(codim_dom, iAE, uStart, uEnd);
		dofAgg[jform+1]->GetAgglomerateInternalDofRange(codim_dom, iAE, pStart, pEnd);
		local_offsets[0] = 0;
		local_offsets[1] = uEnd-uStart;
		local_offsets[2] = local_offsets[1]+pEnd-pStart;
		local_offsets[3] = local_offsets[2] + 1;

		dofAgg[jform]->GetAgglomerateDofRange(codim_dom, iAE, uAllStart, uAllEnd);
		dofAgg[jform]->GetAgglomerateBdrDofRange(codim_dom, iAE, uBdrStart, uBdrEnd);

		dofAgg[jform]->GetViewAgglomerateInternalDofGlobalNumering(codim_dom, iAE, fineUDof_Internal);
		dofAgg[jform+1]->GetViewAgglomerateInternalDofGlobalNumering(codim_dom, iAE, finePDof);

		dofAgg[jform]->GetViewAgglomerateDofGlobalNumering(codim_dom, iAE, fineUDof);
		dofAgg[jform]->GetViewAgglomerateBdrDofGlobalNumering(codim_dom, iAE, fineUDof_Bdr);

		DenseMatrix loc_pv_dof(local_offsets[2]-local_offsets[1], 1);
		coarsePDof.MakeRef(const_cast<int *>(j_AE_PVdof+i_AE_PVdof[iAE]), 1);
		P[jform+1]->GetSubMatrix(finePDof,coarsePDof,loc_pv_dof);


		// (2) Get local matrices and allocate solver
		SparseMatrix * Mloc  = ExtractSubMatrix(*M_d, uStart, uEnd, uStart, uEnd);
		SparseMatrix * Mloc_ib  = ExtractSubMatrix(*M_d, uStart, uEnd, uBdrStart, uBdrEnd);
		SparseMatrix * Bloc  = ExtractSubMatrix(*B_d, pStart, pEnd, uStart, uEnd);
		SparseMatrix * Wloc  = ExtractSubMatrix(*W_d, pStart, pEnd, pStart, pEnd);

		Vector pvloc(loc_pv_dof.Data(), loc_pv_dof.Height());
		Vector tloc(pvloc.Size());
		Wloc->Mult(pvloc,tloc);
		SparseMatrix * Tloc  = createSparseMatrixRepresentationOfScalarProduct(tloc.GetData(), tloc.Size() );

		MA57BlockOperator solver(3);
		solver.SetBlock(0,0,*Mloc);
		solver.SetBlock(1,0,*Bloc);
		solver.SetBlock(2,1,*Tloc);
		int test = solver.Compute();

		if(test != 0)
		{
			DeRhamSequence_os<<"*************************************\n";
			DeRhamSequence_os<<"hFacet Extension: jform = " << jform << "\n";
			DeRhamSequence_os<<"Error in the Factorization of iAE = " << iAE << "\n";
			Mloc->PrintMatlab( DeRhamSequence_os << "Aloc \n");
			Bloc->PrintMatlab( DeRhamSequence_os << "Bloc \n");
			Tloc->PrintMatlab( DeRhamSequence_os << "Tloc \n");
			DeRhamSequence_os<<"*************************************\n";
		}

		// (3) Solve the harmonic extension
		nrhs_ext = i_hAE_AFCD[iAE+1] - i_hAE_AFCD[iAE];
		nlocalbasis = nrhs_ext;
		coarseUDof_on_Bdr.MakeRef(const_cast<int *>(j_hAE_AFCD+i_hAE_AFCD[iAE]), i_hAE_AFCD[iAE+1]-i_hAE_AFCD[iAE]);

		// Get local rhs
		rhs.SetSizeAndNumberOfVectors(local_offsets[3], nrhs_ext);
		sol.SetSizeAndNumberOfVectors(local_offsets[3], nrhs_ext);

		rhs = 0.0;
		rhs.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);
		rhs.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
		GetRows(*minusRtM_d,  coarseUDof_on_Bdr,  uStart, uEnd, rhs_view_u);
		GetRows(*minusRtBt_d, coarseUDof_on_Bdr, pStart, pEnd, rhs_view_p);

		solver.Mult(rhs, sol);

		sol.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);
		sol.GetRangeView(local_offsets[2], local_offsets[3], rhs_view_l);

		subm.SetSize(local_offsets[1], nrhs_ext);
		sol.CopyToDenseMatrix(0,nrhs_ext,local_offsets[0],local_offsets[1], subm);
		P[jform]->SetSubMatrix(fineUDof_Internal,coarseUDof_on_Bdr, subm);

		subv.SetSize(nrhs_ext);
		for(int isol(0); isol <nrhs_ext; ++isol)
		{
			if(fabs(rhs_view_l.GetDataFromVector(isol)[0]) > smallEntry)
				subv(isol) =  -rhs_view_l.GetDataFromVector(isol)[0];
			else
				subv(isol) = 0.0;
		}

		// Coarsen the ExteriorDerivative
		coarserSequence->D[jform]->SetRow(j_AE_PVdof[ i_AE_PVdof[iAE] ], coarseUDof_on_Bdr, subv);

		// (4) Solve for RangeT Dofs
		nrhs_RangeT = i_AE_NullSpaceDof[iAE+1] - i_AE_NullSpaceDof[iAE];
		nlocalbasis += nrhs_RangeT;
		coarseUDof_InternalRangeT.SetSize(nrhs_RangeT);
		uCDofHandler->SetNumberOfInteriorDofsRangeTSpace(codim_dom, iAE, nrhs_RangeT);
		nRangeTDofs += nrhs_RangeT;
		if(nrhs_RangeT)
		{
			for(int i(0); i<nrhs_RangeT; ++i)
				coarseUDof_InternalRangeT[i] = coarseDofCounter++;
			coarsePDof.MakeRef(const_cast<int *>(j_AE_NullSpaceDof+i_AE_NullSpaceDof[iAE]), nrhs_RangeT);

			subm.SetSize(local_offsets[2]-local_offsets[1], nrhs_RangeT);
			P[jform+1]->GetSubMatrix(finePDof,coarsePDof,subm);

			// Get local rhs
			rhs.SetSizeAndNumberOfVectors(local_offsets[3], nrhs_RangeT);
			sol.SetSizeAndNumberOfVectors(local_offsets[3], nrhs_RangeT);

			rhs = 0.0;
			rhs.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
			MultiVector subm_as_mv(subm.Data(), nrhs_RangeT, local_offsets[2]-local_offsets[1]);
			MatrixTimesMultiVector(*Wloc, subm_as_mv, rhs_view_p);
			solver.Mult(rhs, sol);

			subm.SetSize(local_offsets[1], nrhs_RangeT);
			sol.CopyToDenseMatrix(0,nrhs_RangeT,local_offsets[0],local_offsets[1], subm);
			P[jform]->SetSubMatrix(fineUDof_Internal,coarseUDof_InternalRangeT, subm);

			for(int i(0); i<nrhs_RangeT; ++i)
				coarserSequence->D[jform]->Set(coarsePDof[i], coarseUDof_InternalRangeT[i], 1.);

			for(int i(0); i<nrhs_RangeT; ++i)
				uCDofHandler->SetDofType(coarseUDof_InternalRangeT[i], DofHandlerALG::RangeTSpace);

		}

		// (5) Solve for Null Dofs
		if( local_offsets[1] > nrhs_RangeT && nTargets)
		{
			nrhs_Null = nTargets;
			MultiVector localTargets_Interior(nrhs_Null , fineUDof_Internal.Size() );
			MultiVector localTargets_Bdr(nrhs_Null , fineUDof_Bdr.Size() );
			targets[jform]->GetSubMultiVector(fineUDof_Internal, localTargets_Interior);
			targets[jform]->GetSubMultiVector(fineUDof_Bdr, localTargets_Bdr);

			// Get local rhs
			rhs.SetSizeAndNumberOfVectors(local_offsets[3], nrhs_Null);
			sol.SetSizeAndNumberOfVectors(local_offsets[3], nrhs_Null);

			rhs = 0.0;
			rhs.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);
			MatrixTimesMultiVector(-1., *Mloc_ib, localTargets_Bdr, rhs_view_u);

			rhs.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
			MatrixTimesMultiVector(*Bloc, localTargets_Interior, rhs_view_p);
			solver.Mult(rhs, sol);

			sol.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);
			add(1., localTargets_Interior, -1., rhs_view_u, localTargets_Interior);

#if ELAG_DEBUG
{
		MultiVector dummy( nrhs_Null , local_offsets[2] - local_offsets[1]);
		MatrixTimesMultiVector(*Bloc, localTargets_Interior, dummy);

		if(dummy.Normlinf() > 1e-6 )
		{
			std::cout << "|| D u_loc ||_inf = " << dummy.Normlinf() << "\n";
			mfem_error("Adding a non divergence free bubble \n");
		}
}
#endif
			svd.ComputeON(localTargets_Interior, sv);

			double s_max_tol = tolSVD;

			nrhs_Null = 0;
			coarseUDof_InternalNull.SetSize( nrhs_Null );
			for(; nrhs_Null < sv.Size(); ++nrhs_Null)
			{
				if(sv(nrhs_Null) < s_max_tol)
					break;

				uCDofHandler->SetDofType(coarseDofCounter, DofHandlerALG::NullSpace);
				coarseUDof_InternalNull.Append(coarseDofCounter++);
			}

			nlocalbasis += nrhs_Null;
			nNullSpaceDofs += nrhs_Null;

			uCDofHandler->SetNumberOfInteriorDofsNullSpace(codim_dom, iAE, nrhs_Null);

			subm.SetSize(local_offsets[1] - local_offsets[0], nrhs_Null);
			localTargets_Interior.CopyToDenseMatrix(0, nrhs_Null, local_offsets[0], local_offsets[1], subm);

			P[jform]->SetSubMatrix(fineUDof_Internal,coarseUDof_InternalNull, subm);

		}
		else
		{
			nrhs_Null = 0;
			uCDofHandler->SetNumberOfInteriorDofsNullSpace(codim_dom, iAE, nrhs_Null);
			coarseUDof_InternalNull.SetSize( nrhs_Null );
		}

		// (5) Projector
		{
			int n_internal = nrhs_RangeT + nrhs_Null;
			Array<int> allInteralCDofs( n_internal );
			int * tmp =  std::copy(coarseUDof_InternalRangeT.GetData(), coarseUDof_InternalRangeT.GetData()+nrhs_RangeT,allInteralCDofs.GetData() );
			tmp =  std::copy(coarseUDof_InternalNull.GetData(), coarseUDof_InternalNull.GetData()+nrhs_Null, tmp);
			DenseMatrix localBasis(fineUDof_Internal.Size(), n_internal );
			P[jform]->GetSubMatrix(fineUDof_Internal, allInteralCDofs, localBasis);
			MultiVector myVect(localBasis.Data(), n_internal, local_offsets[1] - local_offsets[0]);
			Pi[jform]->CreateDofFunctional(codim_dom, iAE, myVect, *Mloc);
		}
		// (6) Coarsen part of the mass matrix
		{
			Array<int> allCDofs(nlocalbasis);
			int * tmp = std::copy(coarseUDof_on_Bdr.GetData(), coarseUDof_on_Bdr.GetData()+coarseUDof_on_Bdr.Size(), allCDofs.GetData() );
			tmp =  std::copy(coarseUDof_InternalRangeT.GetData(), coarseUDof_InternalRangeT.GetData()+coarseUDof_InternalRangeT.Size(), tmp);
			tmp =  std::copy(coarseUDof_InternalNull.GetData(), coarseUDof_InternalNull.GetData()+coarseUDof_InternalNull.Size(), tmp);
			DenseMatrix localBasis(fineUDof.Size(), nlocalbasis);
			P[jform]->GetSubMatrix(fineUDof, allCDofs, localBasis);
			DenseMatrix * cElemMass = new DenseMatrix(nlocalbasis, nlocalbasis);
			SparseMatrix * M_aa = ExtractSubMatrix(*M_d, uAllStart, uAllEnd, uAllStart, uAllEnd);
			WeightedInnerProduct dot2(*M_aa);
			dot2(localBasis, localBasis, *cElemMass );
			cElemMass->Symmetrize();
			mass.SetElementalMatrix(iAE, cElemMass);
			delete M_aa;
		}
		delete Mloc;
		delete Mloc_ib;
		delete Bloc;
		delete Wloc;

		destroySparseMatrixRepresentationOfScalarProduct(Tloc);

	}

	DeRhamSequence_os << "*** Number of dofs that have been extended " << R_t->Size() << "\n";
	DeRhamSequence_os << "*** Number of RangeTSpace Dofs             " << nRangeTDofs << "\n";
	DeRhamSequence_os << "*** Number of NullSpace Dofs             "   << nNullSpaceDofs << "\n";

	delete AE_PVdof;
	delete AE_NullSpaceDof;

	uCDofHandler->BuildEntityDofTable(codim_dom);
	coarserSequence->M(codim_dom, jform) = mass.GetAsSparseMatrix();


	if(	( coarseDofCounter != R_t->Size() + nRangeTDofs + nNullSpaceDofs )                                                  ||
		( uCDofHandler->GetNumberInteriorDofs( codim_dom ) != nRangeTDofs + nNullSpaceDofs )   )
	{
		std::cout << "Number of Interior Dof according to DofHandler: " << uCDofHandler->GetNumberInteriorDofs( codim_dom ) << std::endl;
		std::cout << "Actual of Interior Dof: " <<coarseDofCounter - R_t->Size() << std::endl;
		mfem_error("DeRhamSequence::hFacetExtension(int jform)");
	}



	delete minusRtM_d;
	delete minusRtBt_d;
	delete R_t;
	delete B_d;
	delete hAE_AFCD;
	delete W_d;
	delete M_d;
	delete D_d;

}

void DeRhamSequence::hFacetExtension_new(int jform)
{

	DeRhamSequence_os << "Enter hFacetExtension of " << jform << "\n";
	std::cout << "NEW!\n";

	AgglomeratedTopology::EntityByCodim codim_bdr = static_cast<AgglomeratedTopology::EntityByCodim>(nforms - jform - 1);
	AgglomeratedTopology::EntityByCodim codim_dom = static_cast<AgglomeratedTopology::EntityByCodim>(nforms - jform - 2);

	DofHandlerALG * uCDofHandler = dynamic_cast<DofHandlerALG *>(coarserSequence->dof[jform]);
	DofHandlerALG * pCDofHandler = dynamic_cast<DofHandlerALG *>(coarserSequence->dof[jform+1]);

	if(uCDofHandler == NULL || pCDofHandler == 0)
		mfem_error("The coarse dof handlers are not of DofHandlerALG type");

	/*
	 * Here we do block solvers for the following systems
	 *
	 * 	[ A_ii   B_ai^T    0   ] [ u ]     [A_ib*R_d ]    [ A_ii   B_ai^T    0   ] [ u ]     [ 0 ]
	 * 	[ B_ai     0       T^T ] [ p ] = - [B_ab*R_d ],   [ B_ai     0       T^T ] [ p ] = - [ x ]
	 * 	[ 0        T       0   ] [ l ]     [ 0       ]    [ 0        T       0   ] [ l ]     [ 0 ]
	 *
	 *	where:
	 *
	 *	u is the sparse matrix representing the Projector operator between the coarse and fine ispace.
	 *	p is the sparse matrix of lagrangian multipliers (that can be ignored)
	 *	l is the sparse matrix representing the coarse derivative operator
	 *
	 *	A_ii is the sparse matrix discretizing (u,v) + <\der u, \der v> restricted to the interal dofs of each agglomerate
	 *	A_ib is the sparse rectangular matrix discretizing (u,v) + <\der u, \der v> whose rows are restricted to the interal dofs of each agglomerate,
	 *	     and the columns are restricted to the boundary dofs of each agglomerate.
	 *	B_ai is the sparse rectangular matrix discretizing <\der u, q> whose columns are restricted to the internal dofs of each agglomerate.
	 *	B_ab is the sparse rectangular matrix discretizing <\der u, q> whose columns are restricted to the bdr dofs of each agglomerate.
	 *	T is the sparse matrix discretizing <Q,q> where Q belongs to the coarse PV (jform+1) and q belongs to the fine (iform+1).
	 */
	//		my_os << "Harmonic extension for space " << jform << " on variety of codimension " << codim_dom << std::endl;
	// Agglomerate-based Mass Matrix (primal variable)
	SparseMatrix * M_d = AssembleAgglomerateMatrix(codim_dom, *(M(codim_dom, jform)), dofAgg[jform], dofAgg[jform]);
	// Agglomerate-based Derivative operator ( primal variable --> derivative )
	SparseMatrix * D_d = DistributeAgglomerateMatrix(codim_dom, *(D[jform]), dofAgg[jform+1], dofAgg[jform] );
	// Agglomerate-based Mass Matrix ( derivative )
	SparseMatrix * W_d = AssembleAgglomerateMatrix(codim_dom, *(M(codim_dom, jform+1)), dofAgg[jform+1], dofAgg[jform+1]);

	// Matrix Containing the target derivative on the coarse space
	Vector * pv_d = dofAgg[jform+1]->DistributeGlobalVector(codim_dom, *(pvTraces[jform+1]));
	Vector T_d(pv_d->Size());
	W_d->Mult(*pv_d, T_d);
	delete pv_d;

	SparseMatrix * B_d = Mult(*W_d, *D_d);

	// Matrix Containing the lifting of the bc on the boundary
	SparseMatrix * R_t = TransposeAbstractSparseMatrix(*P[jform], 1);
	SparseMatrix * Dt = Transpose(*D[jform]);
	SparseMatrix * M_gd = Assemble(codim_dom, *M_d, dofAgg[jform],  NULL  );
	SparseMatrix * W_gd = Assemble(codim_dom, *W_d, dofAgg[jform+1], NULL);
	R_t->SetWidth( M_gd->Size() );

	SparseMatrix * minusRtM_d = Mult(*R_t, *M_gd);
	*minusRtM_d *= -1.0;
	R_t->SetWidth( Dt->Size() );
	SparseMatrix * minusRtDt = Mult(*R_t, *Dt);
	*minusRtDt *= -1.0;
	minusRtDt->SetWidth( W_gd->Size() );
	SparseMatrix * minusRtBt_d = Mult(*minusRtDt, *W_gd);

	delete minusRtDt;
	delete Dt;
	delete W_gd;
	delete M_gd;


	// hyperAgglomeratedElement_AFacetCDofs
	SparseMatrix * hAE_AFCD = Mult(coarserSequence->topo->B(codim_dom), const_cast<SparseMatrix &>(coarserSequence->dof[jform]->GetEntityDofTable(codim_bdr)) );
	int nhAE = hAE_AFCD->Size();
	const int * i_hAE_AFCD = hAE_AFCD->GetI();
	const int * j_hAE_AFCD = hAE_AFCD->GetJ();

	// SparseMatrix to find the global PV dof for jform+1.
	SparseMatrix * AE_PVdof = pCDofHandler->GetEntityRangeTSpaceDofTable( codim_dom );
	const int * const i_AE_PVdof = AE_PVdof->GetI();
	const int * const j_AE_PVdof = AE_PVdof->GetJ();

	// SparseMatrix to find the global NullSpace dof for jform+1 (this new dofs of jform will go in RangeTSpace)
	SparseMatrix * AE_NullSpaceDof = pCDofHandler->GetEntityNullSpaceDofTable( codim_dom );
	const int * const i_AE_NullSpaceDof = AE_NullSpaceDof->GetI();
	const int * const j_AE_NullSpaceDof = AE_NullSpaceDof->GetJ();

	int uStart, uEnd, pStart, pEnd;
	int uBdrStart, uBdrEnd;
	int uAllStart, uAllEnd;

	MultiVector rhs, sol;
	int local_offsets[4];
	int nrhs_ext, nrhs_RangeT, nrhs_Null, nlocalbasis;
	MultiVector rhs_view_u,rhs_view_p,rhs_view_l;
	MultiVector sol_view_u,sol_view_p,sol_view_l;

	Array<int> coarseUDof_on_Bdr, coarseUDof_InternalRangeT, coarseUDof_InternalNull;
	Array<int> coarsePDof;

	Array<int> fineUDof_Internal, fineUDof_Bdr, fineUDof, finePDof;

	int estimateNumberCoarseDofs(0);                                 // Upper bound on the number of coarse dofs,
	                                                                 // assuming that targets are linearly independent from all other types of
	                                                                 // dofs.
	estimateNumberCoarseDofs = R_t->Size()+P[jform+1]->Width();       // Traces dofs + Derivative dofs + number of AE
	estimateNumberCoarseDofs += nhAE*(targets[jform]->NumberOfVectors() - 1);   // Maximum allowed number of Bubble AE dofs - number of AE

	coarserSequence->D[jform] = new SparseMatrix( P[jform+1]->Width(), estimateNumberCoarseDofs );
	ElementalMatricesContainer mass(nhAE);

	DenseMatrix subm;
	Vector subv;

	int nNullSpaceDofs(0), nRangeTDofs(0);

	int coarseDofCounter(R_t->Size());	//The dofs we add are bubbles on the domain,
	                                    // so their global numbering is higher then the dof of the traces

	SVD_Calculator svd;
	svd.setFlagON();
	const int nTargets = targets[jform]->NumberOfVectors();
	const int max_nfDof_per_AE = dofAgg[jform  ]->GetAEntityDofTable( codim_dom ).MaxRowSize();
	if(nTargets)
		svd.AllocateOptimalSize(max_nfDof_per_AE, nTargets);
	Vector sv(nTargets);

	for(int iAE(0); iAE < nhAE; ++iAE)
	{
		// (1) Compute the offsets
		dofAgg[jform  ]->GetAgglomerateInternalDofRange(codim_dom, iAE, uStart, uEnd);
		dofAgg[jform+1]->GetAgglomerateInternalDofRange(codim_dom, iAE, pStart, pEnd);
		local_offsets[0] = 0;
		local_offsets[1] = uEnd-uStart;
		local_offsets[2] = local_offsets[1]+pEnd-pStart;
		local_offsets[3] = local_offsets[2] + 1;

		dofAgg[jform]->GetAgglomerateDofRange(codim_dom, iAE, uAllStart, uAllEnd);
		dofAgg[jform]->GetAgglomerateBdrDofRange(codim_dom, iAE, uBdrStart, uBdrEnd);

		dofAgg[jform]->GetViewAgglomerateInternalDofGlobalNumering(codim_dom, iAE, fineUDof_Internal);
		dofAgg[jform+1]->GetViewAgglomerateInternalDofGlobalNumering(codim_dom, iAE, finePDof);

		dofAgg[jform]->GetViewAgglomerateDofGlobalNumering(codim_dom, iAE, fineUDof);
		dofAgg[jform]->GetViewAgglomerateBdrDofGlobalNumering(codim_dom, iAE, fineUDof_Bdr);

		// (2) Get local matrices and allocate solver
		SparseMatrix * Mloc  = ExtractSubMatrix(*M_d, uStart, uEnd, uStart, uEnd);
		SparseMatrix * Mloc_ib  = ExtractSubMatrix(*M_d, uStart, uEnd, uBdrStart, uBdrEnd);
		SparseMatrix * Bloc  = ExtractSubMatrix(*B_d, pStart, pEnd, uStart, uEnd);
		SparseMatrix * Tloc  = createSparseMatrixRepresentationOfScalarProduct(T_d.GetData()+pStart, pEnd - pStart);
		SparseMatrix * Wloc  = ExtractSubMatrix(*W_d, pStart, pEnd, pStart, pEnd);

		MA57BlockOperator solver(3);
		solver.SetBlock(0,0,*Mloc);
		solver.SetBlock(1,0,*Bloc);
		solver.SetBlock(2,1,*Tloc);
		int test = solver.Compute();

		if(test != 0)
		{
			DeRhamSequence_os<<"*************************************\n";
			DeRhamSequence_os<<"hFacet Extension: jform = " << jform << "\n";
			DeRhamSequence_os<<"Error in the Factorization of iAE = " << iAE << "\n";
			Mloc->PrintMatlab( DeRhamSequence_os << "Aloc \n");
			Bloc->PrintMatlab( DeRhamSequence_os << "Bloc \n");
			Tloc->PrintMatlab( DeRhamSequence_os << "Tloc \n");
			DeRhamSequence_os<<"*************************************\n";
		}

		// (3) Solve the harmonic extension
		nrhs_ext = i_hAE_AFCD[iAE+1] - i_hAE_AFCD[iAE];
		nlocalbasis = nrhs_ext;
		coarseUDof_on_Bdr.MakeRef(const_cast<int *>(j_hAE_AFCD+i_hAE_AFCD[iAE]), i_hAE_AFCD[iAE+1]-i_hAE_AFCD[iAE]);

		// Get local rhs
		rhs.SetSizeAndNumberOfVectors(local_offsets[3], nrhs_ext);
		sol.SetSizeAndNumberOfVectors(local_offsets[3], nrhs_ext);

		rhs = 0.0;
		rhs.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);
		rhs.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
		GetRows(*minusRtM_d,  coarseUDof_on_Bdr,  uStart, uEnd, rhs_view_u);
		GetRows(*minusRtBt_d, coarseUDof_on_Bdr, pStart, pEnd, rhs_view_p);

		solver.Mult(rhs, sol);

		sol.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);
		sol.GetRangeView(local_offsets[2], local_offsets[3], rhs_view_l);

		subm.SetSize(local_offsets[1], nrhs_ext);
		sol.CopyToDenseMatrix(0,nrhs_ext,local_offsets[0],local_offsets[1], subm);
		P[jform]->SetSubMatrix(fineUDof_Internal,coarseUDof_on_Bdr, subm);

		subv.SetSize(nrhs_ext);
		for(int isol(0); isol <nrhs_ext; ++isol)
		{
			if(fabs(rhs_view_l.GetDataFromVector(isol)[0]) > smallEntry)
				subv(isol) =  -rhs_view_l.GetDataFromVector(isol)[0];
			else
				subv(isol) = 0.0;
		}

		// Coarsen the ExteriorDerivative
		coarserSequence->D[jform]->SetRow(j_AE_PVdof[ i_AE_PVdof[iAE] ], coarseUDof_on_Bdr, subv);

		// (4) Solve for RangeT Dofs
		nrhs_RangeT = i_AE_NullSpaceDof[iAE+1] - i_AE_NullSpaceDof[iAE];
		nlocalbasis += nrhs_RangeT;
		coarseUDof_InternalRangeT.SetSize(nrhs_RangeT);
		uCDofHandler->SetNumberOfInteriorDofsRangeTSpace(codim_dom, iAE, nrhs_RangeT);
		nRangeTDofs += nrhs_RangeT;
		if(nrhs_RangeT)
		{
			for(int i(0); i<nrhs_RangeT; ++i)
				coarseUDof_InternalRangeT[i] = coarseDofCounter++;
			coarsePDof.MakeRef(const_cast<int *>(j_AE_NullSpaceDof+i_AE_NullSpaceDof[iAE]), nrhs_RangeT);

			subm.SetSize(local_offsets[2]-local_offsets[1], nrhs_RangeT);
			P[jform+1]->GetSubMatrix(finePDof,coarsePDof,subm);

			// Get local rhs
			rhs.SetSizeAndNumberOfVectors(local_offsets[3], nrhs_RangeT);
			sol.SetSizeAndNumberOfVectors(local_offsets[3], nrhs_RangeT);

			rhs = 0.0;
			rhs.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
			MultiVector subm_as_mv(subm.Data(), nrhs_RangeT, local_offsets[2]-local_offsets[1]);
			MatrixTimesMultiVector(*Wloc, subm_as_mv, rhs_view_p);
			solver.Mult(rhs, sol);

			subm.SetSize(local_offsets[1], nrhs_RangeT);
			sol.CopyToDenseMatrix(0,nrhs_RangeT,local_offsets[0],local_offsets[1], subm);
			P[jform]->SetSubMatrix(fineUDof_Internal,coarseUDof_InternalRangeT, subm);

			for(int i(0); i<nrhs_RangeT; ++i)
				coarserSequence->D[jform]->Set(coarsePDof[i], coarseUDof_InternalRangeT[i], 1.);

			for(int i(0); i<nrhs_RangeT; ++i)
				uCDofHandler->SetDofType(coarseUDof_InternalRangeT[i], DofHandlerALG::RangeTSpace);

		}

		// (5) Solve for Null Dofs
		if( local_offsets[1] > nrhs_RangeT && nTargets)
		{
			nrhs_Null = nTargets;
			MultiVector localTargets_Interior(nrhs_Null , fineUDof_Internal.Size() );
			MultiVector localTargets_Bdr(nrhs_Null , fineUDof_Bdr.Size() );
			targets[jform]->GetSubMultiVector(fineUDof_Internal, localTargets_Interior);
			targets[jform]->GetSubMultiVector(fineUDof_Bdr, localTargets_Bdr);

			// Get local rhs
			rhs.SetSizeAndNumberOfVectors(local_offsets[3], nrhs_Null);
			sol.SetSizeAndNumberOfVectors(local_offsets[3], nrhs_Null);

			rhs = 0.0;
			rhs.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);
			MatrixTimesMultiVector(-1., *Mloc_ib, localTargets_Bdr, rhs_view_u);

			rhs.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
			MatrixTimesMultiVector(*Bloc, localTargets_Interior, rhs_view_p);
			solver.Mult(rhs, sol);

			sol.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);
			add(1., localTargets_Interior, -1., rhs_view_u, localTargets_Interior);

#if elemAGG_Debug
{
		MultiVector dummy( nrhs_Null , local_offsets[2] - local_offsets[1]);
		MatrixTimesMultiVector(*Bloc, localTargets_Interior, dummy);

		if(dummy.Normlinf() > 1e-6 )
		{
			std::cout << "|| D u_loc ||_inf = " << dummy.Normlinf() << "\n";
			mfem_error("Adding a non divergence free bubble \n");
		}
}
#endif
			svd.ComputeON(localTargets_Interior, sv);

			double s_max_tol = tolSVD;

			nrhs_Null = 0;
			coarseUDof_InternalNull.SetSize( nrhs_Null );
			for(; nrhs_Null < sv.Size(); ++nrhs_Null)
			{
				if(sv(nrhs_Null) < s_max_tol)
					break;

				uCDofHandler->SetDofType(coarseDofCounter, DofHandlerALG::NullSpace);
				coarseUDof_InternalNull.Append(coarseDofCounter++);
			}

			nlocalbasis += nrhs_Null;
			nNullSpaceDofs += nrhs_Null;

			uCDofHandler->SetNumberOfInteriorDofsNullSpace(codim_dom, iAE, nrhs_Null);

			subm.SetSize(local_offsets[1] - local_offsets[0], nrhs_Null);
			localTargets_Interior.CopyToDenseMatrix(0, nrhs_Null, local_offsets[0], local_offsets[1], subm);

			P[jform]->SetSubMatrix(fineUDof_Internal,coarseUDof_InternalNull, subm);

		}
		else
		{
			nrhs_Null = 0;
			uCDofHandler->SetNumberOfInteriorDofsNullSpace(codim_dom, iAE, nrhs_Null);
			coarseUDof_InternalNull.SetSize( nrhs_Null );
		}

		// (5) Projector
		{
			int n_internal = nrhs_RangeT + nrhs_Null;
			Array<int> allInteralCDofs( n_internal );
			int * tmp =  std::copy(coarseUDof_InternalRangeT.GetData(), coarseUDof_InternalRangeT.GetData()+nrhs_RangeT,allInteralCDofs.GetData() );
			tmp =  std::copy(coarseUDof_InternalNull.GetData(), coarseUDof_InternalNull.GetData()+nrhs_Null, tmp);
			DenseMatrix localBasis(fineUDof_Internal.Size(), n_internal );
			P[jform]->GetSubMatrix(fineUDof_Internal, allInteralCDofs, localBasis);
			MultiVector myVect(localBasis.Data(), n_internal, local_offsets[1] - local_offsets[0]);
			Pi[jform]->CreateDofFunctional(codim_dom, iAE, myVect, *Mloc);
		}
		// (6) Coarsen part of the mass matrix
		{
			Array<int> allCDofs(nlocalbasis);
			int * tmp = std::copy(coarseUDof_on_Bdr.GetData(), coarseUDof_on_Bdr.GetData()+coarseUDof_on_Bdr.Size(), allCDofs.GetData() );
			tmp =  std::copy(coarseUDof_InternalRangeT.GetData(), coarseUDof_InternalRangeT.GetData()+coarseUDof_InternalRangeT.Size(), tmp);
			tmp =  std::copy(coarseUDof_InternalNull.GetData(), coarseUDof_InternalNull.GetData()+coarseUDof_InternalNull.Size(), tmp);
			DenseMatrix localBasis(fineUDof.Size(), nlocalbasis);
			P[jform]->GetSubMatrix(fineUDof, allCDofs, localBasis);
			DenseMatrix * cElemMass = new DenseMatrix(nlocalbasis, nlocalbasis);
			SparseMatrix * M_aa = ExtractSubMatrix(*M_d, uAllStart, uAllEnd, uAllStart, uAllEnd);
			WeightedInnerProduct dot2(*M_aa);
			dot2(localBasis, localBasis, *cElemMass );
			cElemMass->Symmetrize();
			mass.SetElementalMatrix(iAE, cElemMass);
			delete M_aa;
		}
		delete Mloc;
		delete Mloc_ib;
		delete Bloc;
		delete Wloc;

		destroySparseMatrixRepresentationOfScalarProduct(Tloc);

	}

	DeRhamSequence_os << "*** Number of dofs that have been extended " << R_t->Size() << "\n";
	DeRhamSequence_os << "*** Number of RangeTSpace Dofs             " << nRangeTDofs << "\n";
	DeRhamSequence_os << "*** Number of NullSpace Dofs             "   << nNullSpaceDofs << "\n";

	delete AE_PVdof;
	delete AE_NullSpaceDof;

	uCDofHandler->BuildEntityDofTable(codim_dom);
	coarserSequence->M(codim_dom, jform) = mass.GetAsSparseMatrix();


	if(	( coarseDofCounter != R_t->Size() + nRangeTDofs + nNullSpaceDofs )                                                  ||
		( uCDofHandler->GetNumberInteriorDofs( codim_dom ) != nRangeTDofs + nNullSpaceDofs )   )
	{
		std::cout << "Number of Interior Dof according to DofHandler: " << uCDofHandler->GetNumberInteriorDofs( codim_dom ) << std::endl;
		std::cout << "Actual of Interior Dof: " <<coarseDofCounter - R_t->Size() << std::endl;
		mfem_error("DeRhamSequence::hFacetExtension(int jform)");
	}



	delete minusRtM_d;
	delete minusRtBt_d;
	delete R_t;
	delete B_d;
	delete hAE_AFCD;
	delete W_d;
	delete M_d;
	delete D_d;

}


void DeRhamSequence::hRidgeExtension(int jform)
{
	DeRhamSequence_os << "Enter hRidgeExtension of " << jform << "\n";

//	AgglomeratedTopology::EntityByCodim codim_bdr_ridge = static_cast<AgglomeratedTopology::EntityByCodim>(nforms - jform - 1);
//	AgglomeratedTopology::EntityByCodim codim_bdr_facet = static_cast<AgglomeratedTopology::EntityByCodim>(nforms - jform - 2);
	AgglomeratedTopology::EntityByCodim codim_dom       = static_cast<AgglomeratedTopology::EntityByCodim>(nforms - jform - 3);

	int nAE = coarserSequence->topo->GetNumberLocalEntities( codim_dom );

	DofHandlerALG * uCDofHandler = dynamic_cast<DofHandlerALG *>(coarserSequence->dof[jform]);
	DofHandlerALG * pCDofHandler = dynamic_cast<DofHandlerALG *>(coarserSequence->dof[jform+1]);

	if(uCDofHandler == NULL || pCDofHandler == 0)
		mfem_error("The coarse dof handlers are not of DofHandlerALG type");

	// (1) Extend all the traces (RIDGE-based or FACET-based) by using the correct exterior derivative from the previous space (THIS Dof are already labeled).
	// (2) For each dof in the NullSpace of jform+1 that vanished on the boundary build a shape function in the RangeT Space of jform.
	// (3) If anything is left of the targets build bubble shape functions for Null Space of jform.
#if ELAG_DEBUG
	{
		SparseMatrix * DD = Mult(*(D[jform+1]), *(D[jform]) );
		double DDnorm = DD->MaxNorm();
		try
		{
			elag_assert(DDnorm < 1e-9);
		}
		catch(int)
		{
			std::cout << "DDnorm " << DDnorm << "\n";
		}
		delete DD;
	}
#endif

	// Agglomerate-based Mass Matrix (primal variable)
	SparseMatrix * M_d = AssembleAgglomerateMatrix(codim_dom, *(M(codim_dom, jform)), dofAgg[jform], dofAgg[jform]);
	// Agglomerate-based Derivative operator ( primal variable --> derivative )
	SparseMatrix * D_d = DistributeAgglomerateMatrix(codim_dom, *(D[jform]), dofAgg[jform+1], dofAgg[jform] );
	SparseMatrix * D2_d = DistributeAgglomerateMatrix(codim_dom, *(D[jform+1]), dofAgg[jform+2], dofAgg[jform+1] );
	// Agglomerate-based Mass Matrix ( derivative )
	SparseMatrix * W_d = AssembleAgglomerateMatrix(codim_dom, *(M(codim_dom, jform+1)), dofAgg[jform+1], dofAgg[jform+1]);
	SparseMatrix * W2_d = AssembleAgglomerateMatrix(codim_dom, *(M(codim_dom, jform+2)), dofAgg[jform+2], dofAgg[jform+2]);

#if ELAG_DEBUG
	{
		SparseMatrix * DD = Mult(*D2_d, *D_d );
		double DDnorm = DD->MaxNorm();
		try
		{
			elag_assert(DDnorm < 1e-9);
		}
		catch(int)
		{
			std::cout << "DDnorm " << DDnorm << "\n";
		}
		delete DD;
	}
#endif
	// B matrix
	SparseMatrix * B_d = Mult(*W_d, *D_d);

	// minusC block
	SparseMatrix * D2T_d = Transpose(*D2_d);
	SparseMatrix * tmp   = Mult(*D2T_d, *W2_d);
	SparseMatrix * minusC_d = Mult(*tmp, *D2_d);
	(*minusC_d) *= -1.0;
	delete tmp;
	delete D2T_d;
	delete D2_d;
	delete W2_d;

	// Matrix Containing the target derivative on the coarse space
	SparseMatrix * PDc = MultAbstractSparseMatrix( *(P[jform+1]), *(coarserSequence->D[jform]) );

#if ELAG_DEBUG
	{
		SparseMatrix * D2PDc = Mult(*(D[jform+1]), *PDc );
		double norm = D2PDc->MaxNorm();
		try
		{
			elag_assert(norm < 1e-9);
		}
		catch(int)
		{
			std::string fname = AppendProcessId(comm, "Ridge_D2PDc", "mtx");
			std::ofstream fid(fname.c_str());
			D2PDc->PrintMatlab(fid);
			fid.close();

			fname = AppendProcessId(comm, "deRS", "log");
			fid.open(fname.c_str());
			fid << DeRhamSequence_os.str();
			fid.close();

			std::cout << "NORM: = " << norm << "\n";
		}
		delete D2PDc;
	}
#endif

	// Matrix Containing the lifting of the bc on the boundary
	SparseMatrix * R_t = TransposeAbstractSparseMatrix(*P[jform], 1);
	SparseMatrix * M_gd = Assemble(codim_dom, *M_d, dofAgg[jform],  NULL  );
	SparseMatrix * W_gd = Assemble(codim_dom, *W_d, dofAgg[jform+1], NULL);
	R_t->SetWidth( M_gd->Size() );
	SparseMatrix * minusRtM_d = Mult(*R_t, *M_gd);
	*minusRtM_d *= -1.0;
	SparseMatrix * Dt = Transpose(*D[jform]);
	R_t->SetWidth( Dt->Size() );
	SparseMatrix * minusRtDt = Mult(*R_t, *Dt);
	*minusRtDt *= -1.0;
	minusRtDt->SetWidth( W_gd->Size() );
	SparseMatrix * minusRtBt_d = Mult(*minusRtDt, *W_gd);

	delete minusRtDt;
	delete Dt;
	delete W_gd;
	delete M_gd;

	// SparseMatrix to find the global NullSpace dof for jform+1 (this new dofs of jform will go in RangeTSpace)
	SparseMatrix * AE_NullSpaceDof = pCDofHandler->GetEntityNullSpaceDofTable( codim_dom );
	const int * const i_AE_NullSpaceDof = AE_NullSpaceDof->GetI();
	const int * const j_AE_NullSpaceDof = AE_NullSpaceDof->GetJ();

	int uInternalStart, uInternalEnd, pInternalStart, pInternalEnd;
	int uBdrStart, uBdrEnd;
	int uAllStart, uAllEnd, pIBStart, pIBEnd;
	MultiVector rhs, sol;
	int local_offsets[3];
	int nrhs_ext, nrhs_RangeT, nrhs_Null, nlocalbasis;
	MultiVector rhs_view_u,rhs_view_p,rhs_view_l;
	MultiVector sol_view_u,sol_view_p,sol_view_l;
	Array<int> coarseUDof_on_Bdr, coarseUDof_InternalRangeT, coarseUDof_InternalNull, coarsePDof_Internal;
	Array<int> fineUDof_Internal, fineUDof_Bdr, finePDof_Internal;
	Array<int> fineUDof, finePDof;

	int estimateNumberCoarseDofs(0);                                 // Upper bound on the number of coarse dofs,
	                                                                 // assuming that targets are linearly independent from all other types of
	                                                                 // dofs.
	estimateNumberCoarseDofs = R_t->Size()+P[jform+1]->Width();       // Traces dofs + Derivative dofs + number of AE
	estimateNumberCoarseDofs += nAE*(targets[jform]->NumberOfVectors() - 1);   // Maximum allowed number of Bubble AE dofs - number of AE

	ElementalMatricesContainer mass(nAE);

	DenseMatrix subm;
	Vector subv;

	int coarseDofCounter(R_t->Size());	//The dofs we add are bubbles on the domain,
	                                    // so their global numbering is higher then the dof of the traces

	int nNullSpaceDofs(0), nRangeTDofs(0);

	SVD_Calculator svd;
	svd.setFlagON();
	const int nTargets = targets[jform]->NumberOfVectors();
	const int max_nfDof_per_AE = dofAgg[jform  ]->GetAEntityDofTable( codim_dom ).MaxRowSize();
	if(nTargets)
		svd.AllocateOptimalSize(max_nfDof_per_AE, nTargets);
	Vector sv(nTargets);

	for(int iAE(0); iAE < nAE; ++iAE)
	{
		dofAgg[jform  ]->GetAgglomerateInternalDofRange(codim_dom, iAE, uInternalStart, uInternalEnd);
		dofAgg[jform+1]->GetAgglomerateInternalDofRange(codim_dom, iAE, pInternalStart, pInternalEnd);
		local_offsets[0] = 0;
		local_offsets[1] = uInternalEnd-uInternalStart;
		local_offsets[2] = local_offsets[1]+pInternalEnd-pInternalStart;

		dofAgg[jform  ]->GetAgglomerateDofRange(codim_dom, iAE, uAllStart, uAllEnd);
		dofAgg[jform  ]->GetAgglomerateBdrDofRange(codim_dom, iAE, uBdrStart, uBdrEnd);

		dofAgg[jform+1]->GetAgglomerateDofRange(codim_dom, iAE, pIBStart, pIBEnd);

		uCDofHandler->GetDofsOnBdr(codim_dom, iAE, coarseUDof_on_Bdr);
		nrhs_ext = coarseUDof_on_Bdr.Size();
		nrhs_RangeT = 0;
		nrhs_Null = 0;

		dofAgg[jform]->GetViewAgglomerateInternalDofGlobalNumering(codim_dom, iAE, fineUDof_Internal);
		dofAgg[jform+1]->GetViewAgglomerateInternalDofGlobalNumering(codim_dom, iAE, finePDof_Internal);

		dofAgg[jform]->GetViewAgglomerateDofGlobalNumering(codim_dom, iAE, fineUDof);
		dofAgg[jform+1]->GetViewAgglomerateDofGlobalNumering(codim_dom, iAE, finePDof);

		dofAgg[jform]->GetViewAgglomerateBdrDofGlobalNumering(codim_dom, iAE, fineUDof_Bdr);


		// Get local matrices
		SparseMatrix * Mloc   = ExtractSubMatrix(*M_d, uInternalStart, uInternalEnd, uInternalStart, uInternalEnd);
		SparseMatrix * Mloc_ib= ExtractSubMatrix(*M_d, uInternalStart, uInternalEnd, uBdrStart, uBdrEnd);
		SparseMatrix * Bloc   = ExtractSubMatrix(*B_d, pInternalStart, pInternalEnd, uInternalStart, uInternalEnd);
		SparseMatrix * mCloc  = ExtractSubMatrix(*minusC_d, pInternalStart, pInternalEnd, pInternalStart, pInternalEnd);
		SparseMatrix * Wloc_iA = ExtractSubMatrix(*W_d, pInternalStart, pInternalEnd, pIBStart, pIBEnd);
		SparseMatrix * Wloc = ExtractSubMatrix(*W_d, pInternalStart, pInternalEnd, pInternalStart, pInternalEnd);

		// Get local rhs
		rhs.SetSizeAndNumberOfVectors(local_offsets[2], nrhs_ext);
		sol.SetSizeAndNumberOfVectors(local_offsets[2], nrhs_ext);

		rhs = 0.0;
		rhs.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);
		rhs.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
		GetRows(*minusRtM_d,  coarseUDof_on_Bdr,  uInternalStart, uInternalEnd, rhs_view_u);
		GetRows(*minusRtBt_d, coarseUDof_on_Bdr,  pInternalStart, pInternalEnd, rhs_view_p);

		subm.SetSize(pIBEnd - pIBStart, nrhs_ext);
		PDc->GetSubMatrix(finePDof, coarseUDof_on_Bdr, subm );
		MultiVector localDerivative(subm.Data(), nrhs_ext, pIBEnd - pIBStart );
		MatrixTimesMultiVector(1., *Wloc_iA, localDerivative, rhs_view_p);


		MA57BlockOperator solver(2);
		solver.SetBlock(0,0,*Mloc);
		solver.SetBlock(1,0,*Bloc);
		solver.SetBlock(1,1,*mCloc);
		if(local_offsets[1] != 0)
		{
			solver.Compute();
			solver.Mult(rhs, sol);

#ifdef ELAG_DEBUG
{
		sol.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
		MultiVector tmp(nrhs_ext, local_offsets[2]-local_offsets[1]);
		tmp = 0.;
		MatrixTimesMultiVector(1.,*mCloc, rhs_view_p, tmp);

		if( tmp.Normlinf() > 1e-6 )
			std::cout << "**2Warning hRidge: Form "<< jform << " Agglomerated Element " << iAE <<" of codim "<< codim_dom <<": ||C u ||_inf = " << tmp.Normlinf() << "\n";

}
#endif

		}

		sol.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);

		subm.SetSize(local_offsets[1], nrhs_ext);
		sol.CopyToDenseMatrix(0,nrhs_ext,local_offsets[0],local_offsets[1], subm);
		P[jform]->SetSubMatrix(fineUDof_Internal, coarseUDof_on_Bdr, subm);

		// (4) Solve for RangeT Dofs
		nrhs_RangeT = i_AE_NullSpaceDof[iAE+1] - i_AE_NullSpaceDof[iAE];
		nlocalbasis += nrhs_RangeT;
		nRangeTDofs += nrhs_RangeT;
		coarseUDof_InternalRangeT.SetSize(nrhs_RangeT);
		uCDofHandler->SetNumberOfInteriorDofsRangeTSpace(codim_dom, iAE, nrhs_RangeT);
		if(nrhs_RangeT)
		{
			for(int i(0); i<nrhs_RangeT; ++i)
				coarseUDof_InternalRangeT[i] = coarseDofCounter++;

			coarsePDof_Internal.MakeRef(const_cast<int *>(j_AE_NullSpaceDof+i_AE_NullSpaceDof[iAE]), nrhs_RangeT);

			subm.SetSize(local_offsets[2]-local_offsets[1], nrhs_RangeT);
			P[jform+1]->GetSubMatrix(finePDof_Internal,coarsePDof_Internal,subm);

			// Get local rhs
			rhs.SetSizeAndNumberOfVectors(local_offsets[2], nrhs_RangeT);
			sol.SetSizeAndNumberOfVectors(local_offsets[2], nrhs_RangeT);

			rhs = 0.0;
			rhs.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
			MultiVector subm_as_mv(subm.Data(), nrhs_RangeT, local_offsets[2]-local_offsets[1]);
			MatrixTimesMultiVector(*Wloc, subm_as_mv, rhs_view_p);

			if(local_offsets[1] != 0)
				solver.Mult(rhs, sol);

			subm.SetSize(local_offsets[1], nrhs_RangeT);
			sol.CopyToDenseMatrix(0,nrhs_RangeT,local_offsets[0],local_offsets[1], subm);
			P[jform]->SetSubMatrix(fineUDof_Internal, coarseUDof_InternalRangeT, subm);

			for(int i(0); i<nrhs_RangeT; ++i)
				coarserSequence->D[jform]->Set(coarsePDof_Internal[i], coarseUDof_InternalRangeT[i], 1.);

			for(int i(0); i<nrhs_RangeT; ++i)
				uCDofHandler->SetDofType(coarseUDof_InternalRangeT[i], DofHandlerALG::RangeTSpace);
		}


		// (5) Solve for Null Dofs
		if( local_offsets[1] > nrhs_RangeT && nTargets)
		{
//			my_os << "Entering the Null Dofs solver \n";
			nrhs_Null = nTargets;
			MultiVector localTargets_Interior(nrhs_Null , fineUDof_Internal.Size() );
			MultiVector localTargets_Bdr(nrhs_Null , fineUDof_Bdr.Size() );
			targets[jform]->GetSubMultiVector(fineUDof_Internal, localTargets_Interior);
			targets[jform]->GetSubMultiVector(fineUDof_Bdr, localTargets_Bdr);

			// Get local rhs
			rhs.SetSizeAndNumberOfVectors(local_offsets[2], nrhs_Null);
			sol.SetSizeAndNumberOfVectors(local_offsets[2], nrhs_Null);

			rhs = 0.0;
			rhs.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);
			MatrixTimesMultiVector(-1., *Mloc_ib, localTargets_Bdr, rhs_view_u);
			rhs.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
			MatrixTimesMultiVector(*Bloc, localTargets_Interior, rhs_view_p);

			if(local_offsets[1] != 0)
				solver.Mult(rhs, sol);


			sol.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);
			add(1., localTargets_Interior, -1., rhs_view_u, localTargets_Interior);

//			localTargets_Interior.RoundSmallEntriesToZero(smallEntry);

#ifdef ELAG_DEBUG
{
		MultiVector dummy( nrhs_Null , local_offsets[2] - local_offsets[1]);
		MatrixTimesMultiVector(*Bloc, localTargets_Interior, dummy);

		if(dummy.Normlinf() > 1e-9 )
			mfem_error("Adding a non divergence free bubble \n");
}
#endif
			svd.ComputeON(localTargets_Interior, sv);

			double s_max_tol = tolSVD;

			nrhs_Null = 0;
			coarseUDof_InternalNull.SetSize( nrhs_Null );
			for(; nrhs_Null < sv.Size(); ++nrhs_Null)
			{
				if(sv(nrhs_Null) < s_max_tol)
					break;

				uCDofHandler->SetDofType(coarseDofCounter, DofHandlerALG::NullSpace);
				coarseUDof_InternalNull.Append(coarseDofCounter++);
			}

			nlocalbasis += nrhs_Null;
			nNullSpaceDofs += nrhs_Null;

			uCDofHandler->SetNumberOfInteriorDofsNullSpace(codim_dom, iAE, nrhs_Null);

			subm.SetSize(local_offsets[1] - local_offsets[0], nrhs_Null);
			localTargets_Interior.CopyToDenseMatrix(0, nrhs_Null, local_offsets[0], local_offsets[1], subm);

			P[jform]->SetSubMatrix(fineUDof_Internal,coarseUDof_InternalNull, subm);

		}
		else
		{
			nrhs_Null = 0;
			uCDofHandler->SetNumberOfInteriorDofsNullSpace(codim_dom, iAE, nrhs_Null);
			coarseUDof_InternalNull.SetSize( nrhs_Null );
		}

		{
			int n_internal = nrhs_RangeT + nrhs_Null;
			Array<int> allInteralCDofs( n_internal );
			int * tmp =  std::copy(coarseUDof_InternalRangeT.GetData(), coarseUDof_InternalRangeT.GetData()+nrhs_RangeT,allInteralCDofs.GetData() );
			tmp =  std::copy(coarseUDof_InternalNull.GetData(), coarseUDof_InternalNull.GetData()+nrhs_Null, tmp);
			DenseMatrix localBasis(fineUDof_Internal.Size(), n_internal );
			P[jform]->GetSubMatrix(fineUDof_Internal, allInteralCDofs, localBasis);
			MultiVector myVect(localBasis.Data(), n_internal, local_offsets[1] - local_offsets[0]);
			Pi[jform]->CreateDofFunctional(codim_dom, iAE, myVect, *Mloc);
		}

		// (6) Coarsen part of the mass matrix
		nlocalbasis = nrhs_ext + nrhs_RangeT + nrhs_Null;
		Array<int> allCDofs(nlocalbasis);
		int * tmp = std::copy(coarseUDof_on_Bdr.GetData(), coarseUDof_on_Bdr.GetData()+nrhs_ext, allCDofs.GetData() );
		tmp =  std::copy(coarseUDof_InternalRangeT.GetData(), coarseUDof_InternalRangeT.GetData()+nrhs_RangeT, tmp);
		tmp =  std::copy(coarseUDof_InternalNull.GetData(), coarseUDof_InternalNull.GetData()+nrhs_Null, tmp);
		DenseMatrix localBasis(fineUDof.Size(), nlocalbasis);
		P[jform]->GetSubMatrix(fineUDof, allCDofs, localBasis);
		DenseMatrix * cElemMass = new DenseMatrix(nlocalbasis, nlocalbasis);
		SparseMatrix * M_aa = ExtractSubMatrix(*M_d, uAllStart, uAllEnd, uAllStart, uAllEnd);
		WeightedInnerProduct dot2(*M_aa);
		dot2(localBasis, localBasis,*cElemMass );
		cElemMass->Symmetrize();
		mass.SetElementalMatrix(iAE, cElemMass);
		delete M_aa;


		delete Mloc;
		delete Mloc_ib;
		delete Bloc;
		delete mCloc;
		delete Wloc_iA;
		delete Wloc;


	}

	delete AE_NullSpaceDof;

	DeRhamSequence_os << "*** Number of dofs that have been extended " << R_t->Size() << "\n";
	DeRhamSequence_os << "*** Number of RangeTSpace Dofs             " << nRangeTDofs << "\n";
	DeRhamSequence_os << "*** Number of NullSpace Dofs             "   << nNullSpaceDofs << "\n";

	uCDofHandler->BuildEntityDofTable(codim_dom);
	coarserSequence->M(codim_dom, jform) = mass.GetAsSparseMatrix();

	if(uCDofHandler->GetNumberInteriorDofs( codim_dom ) != coarseDofCounter - R_t->Size() )
	{
		std::cout << "Number of Interior Dof according to DofHandler: " << uCDofHandler->GetNumberInteriorDofs( codim_dom ) << std::endl;
		std::cout << "Actual of Interior Dof: " <<coarseDofCounter - R_t->Size() << std::endl;
		mfem_error("DeRhamSequence::hRidgeExtension(int jform)");
	}



	delete minusRtM_d;
	delete minusRtBt_d;
	delete R_t;
	delete B_d;

	delete W_d;
	delete M_d;
	delete D_d;
	delete minusC_d;
	delete PDc;
}

void DeRhamSequence::hPeakExtension(int jform)
{
	DeRhamSequence_os << "Enter hPeakExtension of " << jform << "\n";

//	AgglomeratedTopology::EntityByCodim codim_bdr_peak  = static_cast<AgglomeratedTopology::EntityByCodim>(nforms - jform - 1);
//	AgglomeratedTopology::EntityByCodim codim_bdr_ridge = static_cast<AgglomeratedTopology::EntityByCodim>(nforms - jform - 2);
//	AgglomeratedTopology::EntityByCodim codim_bdr_facet = static_cast<AgglomeratedTopology::EntityByCodim>(nforms - jform - 3);
	AgglomeratedTopology::EntityByCodim codim_dom       = static_cast<AgglomeratedTopology::EntityByCodim>(nforms - jform - 4);

	int nAE = coarserSequence->topo->GetNumberLocalEntities( codim_dom );

	DofHandlerALG * uCDofHandler = dynamic_cast<DofHandlerALG *>(coarserSequence->dof[jform]);
	DofHandlerALG * pCDofHandler = dynamic_cast<DofHandlerALG *>(coarserSequence->dof[jform+1]);

	if(uCDofHandler == NULL || pCDofHandler == 0)
		mfem_error("The coarse dof handlers are not of DofHandlerALG type");

	// (1) Extend all the traces (RIDGE-based or FACET-based) by using the correct exterior derivative from the previous space (THIS Dof are already labeled).
	// (2) For each dof in the NullSpace of jform+1 that vanished on the boundary build a shape function in the RangeT Space of jform.
	// (3) If anything is left of the targets build bubble shape functions for Null Space of jform.

#if ELAG_DEBUG
	{
		SparseMatrix * DD = Mult(*(D[jform+1]), *(D[jform]) );
		double DDnorm = DD->MaxNorm();
		try
		{
			elag_assert(DDnorm < 1e-9);
		}
		catch(int)
		{
			std::cout << "DDnorm " << DDnorm << "\n";
		}
		delete DD;
	}
#endif

	// Agglomerate-based Mass Matrix (primal variable)
	SparseMatrix * M_d = AssembleAgglomerateMatrix(codim_dom, *(M(codim_dom, jform)), dofAgg[jform], dofAgg[jform]);
	// Agglomerate-based Derivative operator ( primal variable --> derivative )
	SparseMatrix * D_d = DistributeAgglomerateMatrix(codim_dom, *(D[jform]), dofAgg[jform+1], dofAgg[jform] );
	SparseMatrix * D2_d = DistributeAgglomerateMatrix(codim_dom, *(D[jform+1]), dofAgg[jform+2], dofAgg[jform+1] );
	// Agglomerate-based Mass Matrix ( derivative )
	SparseMatrix * W_d = AssembleAgglomerateMatrix(codim_dom, *(M(codim_dom, jform+1)), dofAgg[jform+1], dofAgg[jform+1]);
	SparseMatrix * W2_d = AssembleAgglomerateMatrix(codim_dom, *(M(codim_dom, jform+2)), dofAgg[jform+2], dofAgg[jform+2]);

#if ELAG_DEBUG
	{
		SparseMatrix * DD = Mult(*D2_d, *D_d );
		double DDnorm = DD->MaxNorm();
		try
		{
			elag_assert(DDnorm < 1e-9);
		}
		catch(int)
		{
			std::cout << "DDnorm " << DDnorm << "\n";
		}
		delete DD;
	}
#endif

	// B matrix
	SparseMatrix * B_d = Mult(*W_d, *D_d);

	// minusC block
	SparseMatrix * D2T_d = Transpose(*D2_d);
	SparseMatrix * tmp   = Mult(*D2T_d, *W2_d);
	SparseMatrix * minusC_d = Mult(*tmp, *D2_d);
	(*minusC_d) *= -1.0;
	delete tmp;
	delete D2T_d;
	delete D2_d;
	delete W2_d;

	// Matrix Containing the target derivative on the coarse space
	SparseMatrix * PDc = MultAbstractSparseMatrix( *(P[jform+1]), *(coarserSequence->D[jform]) );

#if ELAG_DEBUG
	{
		SparseMatrix * D2PDc = Mult(*(D[jform+1]), *PDc );
		double norm = D2PDc->MaxNorm();

		try
		{
			elag_assert(norm < 1e-9);
		}
		catch(int)
		{
			std::string fname = AppendProcessId(comm, "Ridge_D2PDc", "mtx");
			std::ofstream fid(fname.c_str());
			D2PDc->PrintMatlab(fid);
			fid.close();

			fname = AppendProcessId(comm, "deRS", "log");
			fid.open(fname.c_str());
			fid << DeRhamSequence_os.str();
			fid.close();

			std::cout << "NORM: = " << norm << "\n";
		}
		delete D2PDc;
	}
#endif

	// Matrix Containing the lifting of the bc on the boundary
	SparseMatrix * R_t = TransposeAbstractSparseMatrix(*P[jform], 1);
	SparseMatrix * M_gd = Assemble(codim_dom, *M_d, dofAgg[jform],  NULL  );
	SparseMatrix * W_gd = Assemble(codim_dom, *W_d, dofAgg[jform+1], NULL);
	R_t->SetWidth( M_gd->Size() );
	SparseMatrix * minusRtM_d = Mult(*R_t, *M_gd);
	*minusRtM_d *= -1.0;
	SparseMatrix * Dt = Transpose(*D[jform]);
	R_t->SetWidth( Dt->Size() );
	SparseMatrix * minusRtDt = Mult(*R_t, *Dt);
	*minusRtDt *= -1.0;
	SparseMatrix * minusRtBt_d = Mult(*minusRtDt, *W_gd);

	delete minusRtDt;
	delete Dt;
	delete W_gd;
	delete M_gd;

	// SparseMatrix to find the global NullSpace dof for jform+1 (this new dofs of jform will go in RangeTSpace)
	SparseMatrix * AE_NullSpaceDof = pCDofHandler->GetEntityNullSpaceDofTable( codim_dom );
	const int * const i_AE_NullSpaceDof = AE_NullSpaceDof->GetI();
	const int * const j_AE_NullSpaceDof = AE_NullSpaceDof->GetJ();

	int uInternalStart, uInternalEnd, pInternalStart, pInternalEnd;
	int uBdrStart, uBdrEnd;
	int uAllStart, uAllEnd, pIBStart, pIBEnd;
	MultiVector rhs, sol;
	int local_offsets[3];
	int nrhs_ext, nrhs_RangeT, nrhs_Null, nlocalbasis;
	MultiVector rhs_view_u,rhs_view_p,rhs_view_l;
	MultiVector sol_view_u,sol_view_p,sol_view_l;
	Array<int> coarseUDof_on_Bdr, coarseUDof_InternalRangeT, coarseUDof_InternalNull, coarsePDof_Internal;
	Array<int> fineUDof_Internal, fineUDof_Bdr, finePDof_Internal;
	Array<int> fineUDof, finePDof;

	int estimateNumberCoarseDofs(0);                                 // Upper bound on the number of coarse dofs,
	                                                                 // assuming that targets are linearly independent from all other types of
	                                                                 // dofs.
	estimateNumberCoarseDofs = R_t->Size()+P[jform+1]->Width();       // Traces dofs + Derivative dofs + number of AE
	estimateNumberCoarseDofs += nAE*(targets[jform]->NumberOfVectors() - 1);   // Maximum allowed number of Bubble AE dofs - number of AE

	ElementalMatricesContainer mass(nAE);

	DenseMatrix subm;
	Vector subv;

	int coarseDofCounter(R_t->Size());	//The dofs we add are bubbles on the domain,
	                                    // so their global numbering is higher then the dof of the traces

	int nNullSpaceDofs(0), nRangeTDofs(0);

	for(int iAE(0); iAE < nAE; ++iAE)
	{
		dofAgg[jform  ]->GetAgglomerateInternalDofRange(codim_dom, iAE, uInternalStart, uInternalEnd);
		dofAgg[jform+1]->GetAgglomerateInternalDofRange(codim_dom, iAE, pInternalStart, pInternalEnd);
		local_offsets[0] = 0;
		local_offsets[1] = uInternalEnd-uInternalStart;
		local_offsets[2] = local_offsets[1]+pInternalEnd-pInternalStart;

		dofAgg[jform  ]->GetAgglomerateDofRange(codim_dom, iAE, uAllStart, uAllEnd);
		dofAgg[jform  ]->GetAgglomerateBdrDofRange(codim_dom, iAE, uBdrStart, uBdrEnd);

		dofAgg[jform+1]->GetAgglomerateDofRange(codim_dom, iAE, pIBStart, pIBEnd);

		uCDofHandler->GetDofsOnBdr(codim_dom, iAE, coarseUDof_on_Bdr);
		nrhs_ext = coarseUDof_on_Bdr.Size();
		nrhs_RangeT = 0;
		nrhs_Null = 0;

		dofAgg[jform]->GetViewAgglomerateInternalDofGlobalNumering(codim_dom, iAE, fineUDof_Internal);
		dofAgg[jform+1]->GetViewAgglomerateInternalDofGlobalNumering(codim_dom, iAE, finePDof_Internal);

		dofAgg[jform]->GetViewAgglomerateDofGlobalNumering(codim_dom, iAE, fineUDof);
		dofAgg[jform+1]->GetViewAgglomerateDofGlobalNumering(codim_dom, iAE, finePDof);

		dofAgg[jform]->GetViewAgglomerateBdrDofGlobalNumering(codim_dom, iAE, fineUDof_Bdr);


		// Get local matrices
		SparseMatrix * Mloc   = ExtractSubMatrix(*M_d, uInternalStart, uInternalEnd, uInternalStart, uInternalEnd);
		SparseMatrix * Mloc_ib= ExtractSubMatrix(*M_d, uInternalStart, uInternalEnd, uBdrStart, uBdrEnd);
		SparseMatrix * Bloc   = ExtractSubMatrix(*B_d, pInternalStart, pInternalEnd, uInternalStart, uInternalEnd);
		SparseMatrix * mCloc  = ExtractSubMatrix(*minusC_d, pInternalStart, pInternalEnd, pInternalStart, pInternalEnd);
		SparseMatrix * Wloc_iA = ExtractSubMatrix(*W_d, pInternalStart, pInternalEnd, pIBStart, pIBEnd);
		SparseMatrix * Wloc = ExtractSubMatrix(*W_d, pInternalStart, pInternalEnd, pInternalStart, pInternalEnd);

		// Get local rhs
		rhs.SetSizeAndNumberOfVectors(local_offsets[2], nrhs_ext);
		sol.SetSizeAndNumberOfVectors(local_offsets[2], nrhs_ext);

		rhs = 0.0;
		rhs.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);
		rhs.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
		GetRows(*minusRtM_d,  coarseUDof_on_Bdr,  uInternalStart, uInternalEnd, rhs_view_u);
		GetRows(*minusRtBt_d, coarseUDof_on_Bdr,  pInternalStart, pInternalEnd, rhs_view_p);

		subm.SetSize(pIBEnd - pIBStart, nrhs_ext);
		PDc->GetSubMatrix(finePDof, coarseUDof_on_Bdr, subm );
		MultiVector localDerivative(subm.Data(), nrhs_ext, pIBEnd - pIBStart );
		MatrixTimesMultiVector(1., *Wloc_iA, localDerivative, rhs_view_p);


		MA57BlockOperator solver(2);
		solver.SetBlock(0,0,*Mloc);
		solver.SetBlock(1,0,*Bloc);
		solver.SetBlock(1,1,*mCloc);
		if(local_offsets[1] != 0 )
		{
			solver.Compute();
			solver.Mult(rhs, sol);
#ifdef ELAG_DEBUG
{
		sol.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
		MultiVector tmp(nrhs_ext, local_offsets[2]-local_offsets[1]);
		tmp = 0.;
		MatrixTimesMultiVector(1.,*mCloc, rhs_view_p, tmp);

		if( tmp.Normlinf() > 1e-6 )
			std::cout << "**2Warning hPeak: Form "<< jform << " Agglomerated Element " << iAE <<" of codim "<< codim_dom <<": ||C u ||_inf = " << tmp.Normlinf() << "\n";
}
#endif
		}

		sol.GetRangeView(local_offsets[0], local_offsets[1], rhs_view_u);

		subm.SetSize(local_offsets[1], nrhs_ext);
		sol.CopyToDenseMatrix(0,nrhs_ext,local_offsets[0],local_offsets[1], subm);
		P[jform]->SetSubMatrix(fineUDof_Internal, coarseUDof_on_Bdr, subm);

		// (4) Solve for RangeT Dofs
		nrhs_RangeT = i_AE_NullSpaceDof[iAE+1] - i_AE_NullSpaceDof[iAE];
		nlocalbasis += nrhs_RangeT;
		nRangeTDofs += nrhs_RangeT;
		coarseUDof_InternalRangeT.SetSize(nrhs_RangeT);
		uCDofHandler->SetNumberOfInteriorDofsRangeTSpace(codim_dom, iAE, nrhs_RangeT);
		if(nrhs_RangeT)
		{
			for(int i(0); i<nrhs_RangeT; ++i)
				coarseUDof_InternalRangeT[i] = coarseDofCounter++;

			coarsePDof_Internal.MakeRef(const_cast<int *>(j_AE_NullSpaceDof+i_AE_NullSpaceDof[iAE]), nrhs_RangeT);

			subm.SetSize(local_offsets[2]-local_offsets[1], nrhs_RangeT);
			P[jform+1]->GetSubMatrix(finePDof_Internal,coarsePDof_Internal,subm);

			// Get local rhs
			rhs.SetSizeAndNumberOfVectors(local_offsets[2], nrhs_RangeT);
			sol.SetSizeAndNumberOfVectors(local_offsets[2], nrhs_RangeT);

			rhs = 0.0;
			rhs.GetRangeView(local_offsets[1], local_offsets[2], rhs_view_p);
			MultiVector subm_as_mv(subm.Data(), nrhs_RangeT, local_offsets[2]-local_offsets[1]);
			MatrixTimesMultiVector(*Wloc, subm_as_mv, rhs_view_p);

			if(local_offsets[1] != 0 )
				solver.Mult(rhs, sol);

			subm.SetSize(local_offsets[1], nrhs_RangeT);
			sol.CopyToDenseMatrix(0,nrhs_RangeT,local_offsets[0],local_offsets[1], subm);
			P[jform]->SetSubMatrix(fineUDof_Internal, coarseUDof_InternalRangeT, subm);

			for(int i(0); i<nrhs_RangeT; ++i)
				coarserSequence->D[jform]->Set(coarsePDof_Internal[i], coarseUDof_InternalRangeT[i], 1.);

			for(int i(0); i<nrhs_RangeT; ++i)
				uCDofHandler->SetDofType(coarseUDof_InternalRangeT[i], DofHandlerALG::RangeTSpace);
		}


		// (5) There are no Null Dofs in H1
		{
			nrhs_Null = 0;
			uCDofHandler->SetNumberOfInteriorDofsNullSpace(codim_dom, iAE, nrhs_Null);
			coarseUDof_InternalNull.SetSize( nrhs_Null );
		}

		{
			int n_internal = nrhs_RangeT + nrhs_Null;
			Array<int> allInteralCDofs( n_internal );
			int * tmp =  std::copy(coarseUDof_InternalRangeT.GetData(), coarseUDof_InternalRangeT.GetData()+nrhs_RangeT,allInteralCDofs.GetData() );
			tmp =  std::copy(coarseUDof_InternalNull.GetData(), coarseUDof_InternalNull.GetData()+nrhs_Null, tmp);
			DenseMatrix localBasis(fineUDof_Internal.Size(), n_internal );
			P[jform]->GetSubMatrix(fineUDof_Internal, allInteralCDofs, localBasis);
			MultiVector myVect(localBasis.Data(), n_internal, local_offsets[1] - local_offsets[0]);
			Pi[jform]->CreateDofFunctional(codim_dom, iAE, myVect, *Mloc);
		}

		// (6) Coarsen part of the mass matrix
		nlocalbasis = nrhs_ext + nrhs_RangeT + nrhs_Null;
		Array<int> allCDofs(nlocalbasis);
		int * tmp = std::copy(coarseUDof_on_Bdr.GetData(), coarseUDof_on_Bdr.GetData()+nrhs_ext, allCDofs.GetData() );
		tmp =  std::copy(coarseUDof_InternalRangeT.GetData(), coarseUDof_InternalRangeT.GetData()+nrhs_RangeT, tmp);
		tmp =  std::copy(coarseUDof_InternalNull.GetData(), coarseUDof_InternalNull.GetData()+nrhs_Null, tmp);
		DenseMatrix localBasis(fineUDof.Size(), nlocalbasis);
		P[jform]->GetSubMatrix(fineUDof, allCDofs, localBasis);
		DenseMatrix * cElemMass = new DenseMatrix(nlocalbasis, nlocalbasis);
		SparseMatrix * M_aa = ExtractSubMatrix(*M_d, uAllStart, uAllEnd, uAllStart, uAllEnd);
		WeightedInnerProduct dot2(*M_aa);
		dot2(localBasis, localBasis,*cElemMass );
		cElemMass->Symmetrize();
		mass.SetElementalMatrix(iAE, cElemMass);
		delete M_aa;


		delete Mloc;
		delete Mloc_ib;
		delete Bloc;
		delete mCloc;
		delete Wloc_iA;
		delete Wloc;


	}

	delete AE_NullSpaceDof;

	DeRhamSequence_os << "*** Number of dofs that have been extended " << R_t->Size() << "\n";
	DeRhamSequence_os << "*** Number of RangeTSpace Dofs             " << nRangeTDofs << "\n";
	DeRhamSequence_os << "*** Number of NullSpace Dofs             "   << nNullSpaceDofs << "\n";

	uCDofHandler->BuildEntityDofTable(codim_dom);
	coarserSequence->M(codim_dom, jform) = mass.GetAsSparseMatrix();

	if(uCDofHandler->GetNumberInteriorDofs( codim_dom ) != coarseDofCounter - R_t->Size() )
	{
		std::cout << "Number of Interior Dof according to DofHandler: " << uCDofHandler->GetNumberInteriorDofs( codim_dom ) << std::endl;
		std::cout << "Actual of Interior Dof: " <<coarseDofCounter - R_t->Size() << std::endl;
		mfem_error("DeRhamSequence::hPeakExtension(int jform)");
	}



	delete minusRtM_d;
	delete minusRtBt_d;
	delete R_t;
	delete B_d;

	delete W_d;
	delete M_d;
	delete D_d;
	delete minusC_d;
	delete PDc;
}

SparseMatrix * DeRhamSequence::getUnextendedInterpolator(int jform)
{
	int nFineDofs = P[jform]->Size();
	int nCoarseDofs = P[jform]->Width();

	SparseMatrix * out = new SparseMatrix(nFineDofs, nCoarseDofs);
	Array<int> internalAggDofs, internalCDofs;

	int baseCodim = dof[jform]->GetMaxCodimensionBaseForDof();

	Array<double> subm_data;
	DenseMatrix subm;
	for(int codim(baseCodim); codim >= 0; --codim)
	{
		int nentities = coarserSequence->topo->GetNumberLocalEntities(static_cast<AgglomeratedTopology::EntityByCodim>(codim));
		for(int ientity(0); ientity < nentities; ++ientity)
		{
			dofAgg[jform]->GetViewAgglomerateInternalDofGlobalNumering(static_cast<AgglomeratedTopology::EntityByCodim>(codim), ientity, internalAggDofs);
			coarserSequence->dof[jform]->GetInteriorDofs(static_cast<AgglomeratedTopology::EntityByCodim>(codim), ientity, internalCDofs);
			subm_data.SetSize(internalAggDofs.Size() * internalCDofs.Size(), 0.0);
			subm.UseExternalData(subm_data, internalAggDofs.Size(), internalCDofs.Size() );
			P[jform]->GetSubMatrix(internalAggDofs, internalCDofs, subm);
			out->SetSubMatrix( internalAggDofs, internalCDofs, subm );
			subm.ClearExternalData();
		}
	}

	out->Finalize();

	return out;

}

void DeRhamSequence::ShowProjector(int jform)
{
	if(jform > nforms-1)
		mfem_error("ShowProjector has a too high jform");

	showP(jform, *P[jform], topo->Partitioning());
}

void DeRhamSequence::ShowDerProjector(int jform)
{
	if(jform >= nforms-1)
		mfem_error("ShowDerProjector has a too high jform");

	SparseMatrix * dP = Mult(*D[jform], *P[jform]);
	showP(jform+1, *dP, topo->Partitioning());
	delete dP;
}

/*-------------------------------------------*/

DeRhamSequenceAlg::DeRhamSequenceAlg(AgglomeratedTopology * topo, int nforms):
	DeRhamSequence(topo, nforms)
{

}


DeRhamSequenceAlg::~DeRhamSequenceAlg()
{

}

SparseMatrix * DeRhamSequenceAlg::ComputeProjectorFromH1ConformingSpace(int jform)
{
	SparseMatrix * pi;
	SparseMatrix * pi_fine = finerSequence->ComputeProjectorFromH1ConformingSpace(jform);
	SparseMatrix * conformingSpaceInterpolator;

	if(jform == nforms-1)
	{
		conformingSpaceInterpolator = finerSequence->GetP(0);
	}
	else
	{
		SparseMatrix * id = createSparseIdentityMatrix(nforms-1);
		conformingSpaceInterpolator = Kron(*(finerSequence->GetP(0)), *id);
		delete id;
	}

		finerSequence->GetPi(jform)->ComputeProjector();
		const SparseMatrix & jspaceProjector = finerSequence->GetPi(jform)->GetProjectorMatrix();

		SparseMatrix * tmp = Mult(*pi_fine, *conformingSpaceInterpolator);
		pi   = Mult(const_cast<SparseMatrix &>(jspaceProjector), *tmp );

	if( jform != nforms-1)
		delete conformingSpaceInterpolator;

	delete tmp;
	delete pi_fine;
	return pi;
}

void DeRhamSequenceAlg::computePVTraces(AgglomeratedTopology::EntityByCodim icodim, Vector & PVinAgg)
{
	int jform = nforms - 1 - icodim;
	int nDofs = dof[jform]->GetNDofs();
	int nAE   = dofAgg[jform]->GetNumberCoarseEntities(icodim);

	if( PVinAgg.Size() != nDofs )
		mfem_error("computePVTraces(Topology::entity icodim, Vector & PVinAgg) #1 \n");

	// j_entity_dof[ i_entity_dof[i] ] is the PV dof on entity i
	const SparseMatrix & entity_dof = dof[jform]->GetEntityDofTable(icodim);
	const int * const i_entity_dof = entity_dof.GetI();
	const int * const j_entity_dof = entity_dof.GetJ();

	const SparseMatrix & AE_e = topo->AEntityEntity(icodim);
//	const int * const i_AE_e = AE_e.GetI();
	const int * j_AE_e = AE_e.GetJ();
	const double * v_AE_e = AE_e.GetData();
	const int nnz_AE_e = AE_e.NumNonZeroElems();

	if(AE_e.Size() != nAE)
		mfem_error("computePVTraces(Topology::entity icodim, Vector & PVinAgg) #2 \n");

	if(AE_e.Width() != entity_dof.Size() )
		mfem_error("computePVTraces(Topology::entity icodim, Vector & PVinAgg) #3 \n");

	PVinAgg = 0.0;

	for(const int * const j_AE_e_end = j_AE_e+nnz_AE_e; j_AE_e != j_AE_e_end; ++j_AE_e, ++v_AE_e)
		PVinAgg( j_entity_dof[ i_entity_dof[*j_AE_e]] ) = *v_AE_e;


}

void DeRhamSequenceAlg::showP(int jform, SparseMatrix & Pc, Array<int> & parts_c)
{
	SparseMatrix * PP = Mult(*(finerSequence->GetP(jform)), Pc);
	Array<int> & parts_f = topo->finerTopology->Partitioning();
	int nfpart = parts_f.Size();
	Array<int> fParts(nfpart);

	for(int i(0); i <nfpart; ++i)
		fParts[i] = parts_c[ parts_f[i] ];

	finerSequence->showP(jform, *PP, fParts);
	delete PP;
}

void DeRhamSequenceAlg::show(int jform, MultiVector & v)
{
	MultiVector vFine(v.NumberOfVectors(), finerSequence->GetP(jform)->Size() );
	MatrixTimesMultiVector( *(finerSequence->GetP(jform) ), v, vFine);
	finerSequence->show(jform, vFine);
}

void DeRhamSequenceAlg::ProjectCoefficient(int jform, Coefficient & c, Vector & v)
{
	if(jform != 0 && jform != nforms-1)
		mfem_error("DeRhamSequenceAlg::ProjectCoefficient");

	v.SetSize( finerSequence->GetP(jform)->Width() );

	Vector vf( finerSequence->GetP(jform)->Size() );
	finerSequence->ProjectCoefficient(jform, c, vf);
	finerSequence->GetPi(jform)->GetProjectorMatrix().Mult(vf, v);
}

void DeRhamSequenceAlg::ProjectVectorCoefficient(int jform, VectorCoefficient & c, Vector & v)
{
	if(jform == 0 || jform == nforms-1)
		mfem_error("DeRhamSequenceAlg::ProjectCoefficient");

	v.SetSize( finerSequence->GetP(jform)->Width() );

	Vector vf( finerSequence->GetP(jform)->Size() );
	finerSequence->ProjectVectorCoefficient(jform, c, vf);
	finerSequence->GetPi(jform)->GetProjectorMatrix().Mult(vf, v);
}

std::stringstream DeRhamSequence::DeRhamSequence_os;
