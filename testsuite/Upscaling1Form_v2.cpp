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

//                       Elag Upscaling - Parallel Version
//
//
// Sample runs:  mpirun -np 4 ./Upscaling1Form --datafile=data.getpot
//               mpirun -np 4 ./Upscaling1Form --datafile=data2D.getpot


#include <fstream>
#include <sstream>

#include <mpi.h>

#include "../src/elag.hpp"

enum {AMGe = 0, ADS, NSOLVERS};
enum {ASSEMBLY = 0, PRECONDITIONER_AMGe, PRECONDITIONER_ADS, SOLVER_AMGe, SOLVER_ADS, NSTAGES};

SparseMatrix * myRAP(SparseMatrix * Rt,  SparseMatrix * A, SparseMatrix * P)
{
	SparseMatrix * R = Transpose(*Rt);
	SparseMatrix * RA = Mult(*R, *A);
	delete R;
	SparseMatrix * RAP = Mult(*RA, *P);
	delete RA;
	return RAP;
}

int main (int argc, char *argv[])
{
	int num_procs, myid;

	// 1. Initialize MPI
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myid);

	GetPot cmm_line(argc, argv);

	std::string dataFile(cmm_line("--datafile", "data.getpot"));
	GetPot data(dataFile.c_str());

	StopWatch chrono;

	//PROGRAM OPTIONS
	int ser_ref_levels = data("nref_serial", 0);
	int par_ref_levels = data("nref_parallel", 2);
	int coarseringFactor = data("coarseringFactor", 8);
	std::string meshfile = data("mesh", "mesh.mesh3d");
	int feorder = data("feorder", 0);
	int upscalingOrder = data("upscalingorder", 0);
	int nbdr = data("nBdrAttributes", 6);
	Array<int> ess_one(nbdr);
	for(int i(0); i < nbdr; ++i)
		ess_one[i] = data("EssAttributesOne", 0, i);
	Array<int> ess_zeros(nbdr);
	for(int i(0); i < nbdr; ++i)
		ess_zeros[i] = data("EssAttributesZeros", 0, i);
	Array<int> nat_one(nbdr);
	for(int i(0); i < nbdr; ++i)
		nat_one[i] = data("NatAttributesOne", 0, i);
	Array<int> nat_zeros(nbdr);
	for(int i(0); i < nbdr; ++i)
		nat_zeros[i] = data("NatAttributesZeros", 0, i);

	//Overload options from command line
	feorder = cmm_line("--feorder", feorder);
	upscalingOrder = cmm_line("--upscalingorder", 0);

	Array<int> ess_attr(nbdr);
	for(int i(0); i < nbdr; ++i)
		ess_attr[i] = ess_one[i] + ess_zeros[i];

	Array<int> nat_attr(nbdr);
	for(int i(0); i < nbdr; ++i)
		nat_attr[i] = nat_one[i] + nat_zeros[i];

	//DEFAULTED LINEAR SOLVER OPTIONS
	int print_iter = 0;
	int max_num_iter = 500;
	double rtol = 1e-6;
	double atol = 1e-12;

	Mesh *mesh;

	if(myid == 0)
	{
		std::cout << "Read mesh " << meshfile   << "\n";
		std::cout << "Refine mesh in serial "   <<   ser_ref_levels << " times. \n";
		std::cout << "Refine mesh in parallel "   <<   par_ref_levels << " times. \n";
	}

	// 2. Read the (serial) mesh from the given mesh file and uniformely refine it.
	std::ifstream imesh(meshfile.c_str());
	if (!imesh)
	{
		std::cerr << "\nCan not open mesh file: " << argv[1] << '\n' << std::endl;
		return 2;
	}
	mesh = new Mesh(imesh, 1, 1);
	imesh.close();

	for (int l = 0; l < ser_ref_levels; l++)
		mesh->UniformRefinement();

	int nDimensions = mesh->Dimension();

	ParMesh * pmesh = new ParMesh(comm, *mesh);
	delete mesh;

	ConstantCoefficient coeff2Form(1.);
	ConstantCoefficient coeff1Form(1.);

	Vector ess_bc(nbdr), nat_bc(nbdr);
	ess_bc = 0.; nat_bc = 0.;

	for(int i(0); i < nbdr; ++i)
		if(ess_one[i] == 1)
			ess_bc(i) = 1.;

	for(int i(0); i < nbdr; ++i)
		if(nat_one[i] == 1)
			nat_bc(i) = -1.;

	PWConstCoefficient ubdr(ess_bc);
	PWConstCoefficient fbdr(nat_bc);

	Vector allones(nDimensions);
	allones = 1.;
	VectorConstantCoefficient tmp(allones);
	VectorRestrictedCoefficient fbdr3d(tmp, nat_one);

	int nLevels = par_ref_levels+1;
	Array<AgglomeratedTopology *> topology(nLevels);
	topology = static_cast<AgglomeratedTopology *>(NULL);

	chrono.Clear();
	chrono.Start();
	topology.Last() = new AgglomeratedTopology( pmesh, nDimensions );
	for(int ilevel = nLevels-1; ilevel > 0; --ilevel)
		topology[ilevel-1] = topology[ilevel]->UniformRefinement();

//	if(nDimensions == 3)
//		pmesh->ReorientTetMesh();


	chrono.Stop();
	if(myid == 0)
		std::cout<<"Timing ELEM_AGG: Mesh Agglomeration done in " << chrono.RealTime() << " seconds \n";

	//-----------------------------------------------------//

	double tolSVD = 1e-9;
	Array<DeRhamSequence *> sequence(topology.Size() );
	if(nDimensions == 3)
		sequence[0] = new DeRhamSequence3D_FE(topology[0], pmesh, feorder);
	else
		sequence[0] = new DeRhamSequence2D_Hdiv_FE(topology[0], pmesh, feorder);

	Array< MultiVector *> targets( sequence[0]->GetNumberOfForms() );

	int jFormStart = 0;
	sequence[0]->SetjformStart( jFormStart );

	if(nDimensions == 3)
		dynamic_cast<DeRhamSequenceFE *>(sequence[0])->ReplaceMassIntegrator(AgglomeratedTopology::ELEMENT, 2, new VectorFEMassIntegrator(coeff2Form), false);
	else
		dynamic_cast<DeRhamSequenceFE *>(sequence[0])->ReplaceMassIntegrator(AgglomeratedTopology::ELEMENT, 2, new MassIntegrator(coeff2Form), false);

	dynamic_cast<DeRhamSequenceFE *>(sequence[0])->ReplaceMassIntegrator(AgglomeratedTopology::ELEMENT, 1, new VectorFEMassIntegrator(coeff1Form), true);


	Array<Coefficient *> L2coeff;
	Array<VectorCoefficient *> Hdivcoeff;
	Array<VectorCoefficient *> Hcurlcoeff;
	fillVectorCoefficientArray(nDimensions, upscalingOrder, Hcurlcoeff);
	fillVectorCoefficientArray(nDimensions, upscalingOrder, Hdivcoeff);
	fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);


	int jform(0);

	targets[jform] = static_cast<MultiVector *>(NULL);
	++jform;

	if(nDimensions == 3)
	{
	targets[jform] = dynamic_cast<DeRhamSequenceFE *>(sequence[0])->InterpolateVectorTargets(jform, Hcurlcoeff);
	++jform;
	}

	targets[jform] = dynamic_cast<DeRhamSequenceFE *>(sequence[0])->InterpolateVectorTargets(jform, Hdivcoeff);
	++jform;

	targets[jform] = dynamic_cast<DeRhamSequenceFE *>(sequence[0])->InterpolateScalarTargets(jform, L2coeff);
	++jform;


	freeCoeffArray(Hcurlcoeff);
	freeCoeffArray(Hdivcoeff);
	freeCoeffArray(L2coeff);


	sequence[0]->SetTargets( targets );

	chrono.Clear();
	chrono.Start();
	for(int i(0); i < nLevels-1; ++i)
	{
		sequence[i]->SetSVDTol( tolSVD );
		StopWatch chronoInterior;
		chronoInterior.Clear();
		chronoInterior.Start();
		sequence[i+1] = sequence[i]->Coarsen();
		chronoInterior.Stop();
		if(myid == 0)
			std::cout<<"Timing ELEM_AGG_LEVEL"<<i<<": Coarsering done in " << chronoInterior.RealTime() << " seconds \n";
	}
	chrono.Stop();

	if(myid == 0)
		std::cout<<"Timing ELEM_AGG: Coarsering done in " << chrono.RealTime() << " seconds \n";

	int form = 1;

	//	testUpscalingHdiv(sequence);
	FiniteElementSpace * fespace = sequence[0]->FemSequence()->GetFeSpace(form);
	LinearForm b(fespace);
	if(nDimensions == 3)
		b.AddBoundaryIntegrator( new VectorFEBoundaryTangentLFIntegrator(fbdr3d) );
	else
		b.AddBoundaryIntegrator( new VectorFEBoundaryFluxLFIntegrator(fbdr));
	b.Assemble();

	GridFunction lift(fespace);
	lift.ProjectBdrCoefficient(ubdr, ess_attr);

	DenseMatrix errors_L2_2(nLevels, nLevels);
	errors_L2_2 = 0.0;
	Vector norm_L2_2(nLevels);
	norm_L2_2 = 0.;

	DenseMatrix errors_div_2(nLevels, nLevels);
	errors_div_2 = 0.0;
	Vector norm_div_2(nLevels);
	norm_div_2 = 0.;

	DenseMatrix timings(nLevels, NSTAGES);
	timings = 0.0;

	Array2D<int> iter(nLevels, NSOLVERS);
	iter = 0;
	Array<int> ndofs(nLevels);
	ndofs = 0;
	Array<int> nnz(nLevels);
	nnz = 0;

	double tdiff;

	Array<SparseMatrix *> allP(nLevels-1);
	Array<SparseMatrix *> allD(nLevels);

	for(int i = 0; i < nLevels - 1; ++i)
		allP[i] = sequence[i]->GetP(form);

	for(int i = 0; i < nLevels; ++i)
		allD[i] = sequence[i]->GetDerivativeOperator(form);

	Array<SparseMatrix *> Ml(nLevels);
	Array<SparseMatrix *> Wl(nLevels);

	for(int k(0); k < nLevels; ++k)
	{
		chrono.Clear();
		chrono.Start();
		Ml[k] = sequence[k]->ComputeMassOperator(form);
		Wl[k] = sequence[k]->ComputeMassOperator(form+1);
		chrono.Stop();
		tdiff = chrono.RealTime();
		if(myid == 0)
			std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in " << tdiff << "s. \n";
		timings(k, ASSEMBLY) += tdiff;
	}

	Array<Vector *> rhs(nLevels);
	Array<Vector *> ess_data(nLevels);
	rhs[0] = &b;
	ess_data[0] = &lift;
	for(int i = 0; i < nLevels-1; ++i)
	{
		rhs[i+1] = new Vector( sequence[i+1]->GetNumberOfDofs(form) );
		ess_data[i+1] = new Vector( sequence[i+1]->GetNumberOfDofs(form) );
		sequence[i]->GetP(form)->MultTranspose(*(rhs[i]), *(rhs[i+1]) );
		sequence[i]->GetPi(form)->ComputeProjector();
		sequence[i]->GetPi(form)->GetProjectorMatrix().Mult(*(ess_data[i]), *(ess_data[i+1]) );
	}

	Array<Vector *> sol_AMGe(nLevels);
	Array<Vector *> sol_ADS(nLevels);
	Array<Vector *> help(nLevels);

	sol_AMGe = static_cast<Vector *>(NULL);
	sol_ADS  = static_cast<Vector *>(NULL);
	help     = static_cast<Vector *>(NULL);

	for(int k(0); k < nLevels; ++k)
	{
		sol_AMGe[k]   = new Vector( sequence[k]->GetNumberOfDofs(form) );
		*(sol_AMGe[k]) = 0.;
		sol_ADS[k]   = new Vector( sequence[k]->GetNumberOfDofs(form) );
		*(sol_ADS[k]) = 0.;
		help[k]  = new Vector( sequence[k]->GetNumberOfDofs(form) );
		*(help[k]) = 0.;
	}

	for(int k(0); k < nLevels; ++k)
	{
		chrono.Clear();
		chrono.Start();
		SparseMatrix * M = Ml[k];
		SparseMatrix * W = Wl[k];
		SparseMatrix * D = allD[k];
		SparseMatrix * DtWD = myRAP(D, W, D);
		SparseMatrix * A = Add(*M, *DtWD);

		int nlocdofs = A->Height();
		Array<int> marker( nlocdofs );
		marker = 0;
		sequence[k]->GetDofHandler(form)->MarkDofsOnSelectedBndr(ess_attr, marker);

		for(int mm = 0; mm < nlocdofs; ++mm)
			if(marker[mm])
				A->EliminateRowCol(mm, ess_data[k]->operator ()(mm), *(rhs[k]) );

		const SharingMap & form1_dofTrueDof( sequence[k]->GetDofHandler(1)->GetDofTrueDof() );

		Vector prhs( form1_dofTrueDof.GetTrueLocalSize() );
		form1_dofTrueDof.Assemble(*(rhs[k]), prhs);
		HypreParMatrix * pA = Assemble(form1_dofTrueDof, *A, form1_dofTrueDof);

		elag_assert(prhs.Size() == pA->Height() );

		chrono.Stop();
		tdiff = chrono.RealTime();
		if(myid == 0)
			std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in " << tdiff << "s. \n";
		timings(k, ASSEMBLY) += tdiff;

		delete DtWD;
		delete A;

		ndofs[k] = pA->GetGlobalNumRows();
		nnz[k] = pA->NNZ();

#if 0
		//AMGe SOLVER
		{
			Array<SparseMatrix *> P(allP_bc.GetData()+k, nLevels-1-k);
			Array<SparseMatrix *> C(allC_bc.GetData()+k, nLevels-1-k);

			chrono.Clear();
			chrono.Start();
			MA57Allocator cSolver;
			AuxiliarySpaceSmootherAllocator<GSSmootherAllocator, GSSmootherAllocator>
			pre_smoother(C, new GSSmootherAllocator(0,2),new GSSmootherAllocator(0,2), MMaux);
			AuxiliarySpaceSmootherAllocator<GSSmootherAllocator, GSSmootherAllocator>
			post_smoother(C, new GSSmootherAllocator(0,2),new GSSmootherAllocator(0,2), MMaux);
			MLPreconditioner prec(nLevels-k);
			prec.SetMatrix(*A);
			prec.SetProlongators(P);
			prec.Compute(cSolver, pre_smoother, post_smoother);
			chrono.Stop();
			tdiff = chrono.RealTime();
			std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Preconditioner done in " << tdiff << "s. \n";
			timings(k, PRECONDITIONER_AMGe) = tdiff;

			CGSolver pcg(comm);
			pcg.SetPrintLevel(print_iter);
			pcg.SetMaxIter(max_num_iter);
			pcg.SetRelTol(rtol);
			pcg.SetAbsTol(atol);
			pcg.SetOperator(*A );
			pcg.SetPreconditioner( prec );
			chrono.Clear();
			chrono.Start();
			pcg.Mult(*(rhs[k]), (*sol_AMGe[k]) );
			chrono.Stop();
			tdiff = chrono.RealTime();
			std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Solver done in " << tdiff << "s. \n";
			timings(k,SOLVER_AMGe) = tdiff;
			if(pcg.GetConverged())
				std::cout << "PCG converged in " << pcg.GetNumIterations() << " with a final residual norm " << pcg.GetFinalNorm() << "\n";
			else
				std::cout << "PCG did not converge in " << pcg.GetNumIterations() << ". Final residual norm is " << pcg.GetFinalNorm() << "\n";

			iter(k,AMGe) = pcg.GetNumIterations();
		}
#else
		{
			(*sol_AMGe[k]) = 0.;
			timings(k, PRECONDITIONER_AMGe) = 0.;
			timings(k,SOLVER_AMGe) = 0.;
			iter(k,AMGe) = 0;
		}
#endif
		//ADS Solver
		{
			HypreExtension::HypreAMSData data;

			chrono.Clear();
			chrono.Start();
//			HypreSmoother prec(*pA, HypreSmoother::Jacobi);
			HypreExtension::HypreAMS prec(*pA, sequence[k], data);
			Vector tmp1(pA->Height()), tmp2(pA->Height() );
			tmp1 = 1.; tmp2 = 2.;
			prec.Mult(tmp1, tmp2);
			chrono.Stop();
			tdiff = chrono.RealTime();
			std::cout << "Timing ADS_LEVEL " << k << ": Preconditioner done in " << tdiff << "s. \n";
			timings(k, PRECONDITIONER_ADS) = tdiff;

			Vector psol( pA->Height() );
			psol = 0.;
			CGSolver pcg(comm);
			pcg.SetPrintLevel(print_iter);
			pcg.SetMaxIter(max_num_iter);
			pcg.SetRelTol(rtol);
			pcg.SetAbsTol(atol);
			pcg.SetOperator(*pA );
			pcg.SetPreconditioner( prec );
			chrono.Clear();
			chrono.Start();
			pcg.Mult(prhs, psol );
			chrono.Stop();
			tdiff = chrono.RealTime();
			if(myid == 0)
				std::cout << "Timing ADS_LEVEL " << k << ": Solver done in " << tdiff << "s. \n";
			timings(k,SOLVER_ADS) = tdiff;

			if(myid == 0)
			{
				if(pcg.GetConverged())
					std::cout << "PCG converged in " << pcg.GetNumIterations() << " with a final residual norm " << pcg.GetFinalNorm() << "\n";
				else
					std::cout << "PCG did not converge in " << pcg.GetNumIterations() << ". Final residual norm is " << pcg.GetFinalNorm() << "\n";
			}

			form1_dofTrueDof.Distribute(psol, *(sol_ADS[k]) );
			iter(k,ADS) = pcg.GetNumIterations();
		}

		Array<Vector *> & sol(sol_ADS);
		//ERROR NORMS
		{
			*(help[k]) = *(sol[k]);
			for(int j = k; j > 0; --j)
				allP[j-1]->Mult( *(help[j]), *(help[j-1]) );

			norm_L2_2(k) = Ml[k]->InnerProduct(*(sol[k]), *(sol[k]) );
			Vector dsol( allD[k]->Height() );
			allD[k]->Mult(*(sol[k]), dsol );
			norm_div_2(k) = Wl[k]->InnerProduct(dsol, dsol );

			for(int j(0); j < k; ++j)
			{
				if(help[j]->Size() != sol[j]->Size() || sol[j]->Size() != allD[j]->Width() )
					mfem_error("size don't match \n");

				int size  = sol[j]->Size();
				int dsize = allD[j]->Height();
				Vector u_H( help[j]->GetData(), size );
				Vector u_h( sol[j]->GetData(), size  );
				Vector u_diff( size ), du_diff( dsize );
				u_diff = 0.; du_diff = 0.;

				subtract(u_H, u_h, u_diff);
				allD[j]->Mult(u_diff, du_diff);

				errors_L2_2(k,j) =  Ml[j]->InnerProduct(u_diff, u_diff);
				errors_div_2(k,j) =  Wl[j]->InnerProduct(du_diff, du_diff);
			}
		}

		//VISUALIZE SOLUTION
		if(1)
		{
			MultiVector tmp(sol[k]->GetData(), 1, sol[k]->Size() );
			sequence[k]->show(form, tmp);
		}

		delete pA;
	}

	DenseMatrix errors_L2(nLevels, nLevels);
	errors_L2 = 0.;
	Vector norm_L2(nLevels);
	norm_L2 = 0.;
	DenseMatrix errors_div(nLevels, nLevels);
	errors_div = 0.;
	Vector norm_div(nLevels);
	norm_div = 0.;
//	DenseMatrix timings(nLevels-1, NSTAGES);

	MPI_Reduce(errors_L2_2.Data(), errors_L2.Data(),errors_L2.Height()*errors_L2.Width(), MPI_DOUBLE,MPI_SUM, 0, comm);
	MPI_Reduce(norm_L2_2.GetData(), norm_L2.GetData(), norm_L2.Size(), MPI_DOUBLE,MPI_SUM, 0, comm);
	MPI_Reduce(errors_div_2.Data(), errors_div.Data(),errors_div.Height()*errors_div.Width(), MPI_DOUBLE,MPI_SUM, 0, comm);
	MPI_Reduce(norm_div_2.GetData(), norm_div.GetData(), norm_div.Size(), MPI_DOUBLE,MPI_SUM, 0, comm);

	std::transform(errors_L2.Data(), errors_L2.Data()+errors_L2.Height()*errors_L2.Width(), errors_L2.Data(), (double(*)(double)) sqrt);
	std::transform(norm_L2.GetData(), norm_L2.GetData()+norm_L2.Size(), norm_L2.GetData(), (double(*)(double)) sqrt);
	std::transform(errors_div.Data(), errors_div.Data()+errors_div.Height()*errors_div.Width(), errors_div.Data(), (double(*)(double)) sqrt);
	std::transform(norm_div.GetData(), norm_div.GetData()+norm_div.Size(), norm_div.GetData(), (double(*)(double)) sqrt);

	if(myid == 0)
	{
		std::cout << "\n{\n";
		int w = 14;
		std::cout << "%level" << std::setw(w) << "size" << std::setw(w) << "nnz" << std::setw(w) 
					 << "nit AMGe" << std::setw(w) << "nit ADS"
					 << std::setw(w) << "Assembly" << std::setw(w) << "Prec AMGe" << std::setw(w) << "Prec ADS"
					 << std::setw(w) << "Solver AMGe" << std::setw(w) << "Solver ADS\n";
		for(int i(0); i < nLevels; ++i)
			std::cout << i << std::setw(w) << ndofs[i] << std::setw(w) << nnz[i] << std::setw(w) 
						 << iter(i,AMGe)<< std::setw(w) << iter(i,ADS)
						 << std::setw(w) << timings(i,ASSEMBLY) << std::setw(w) << timings(i,PRECONDITIONER_AMGe)  << std::setw(w) << timings(i,PRECONDITIONER_ADS)
						 << std::setw(w) << timings(i,SOLVER_AMGe) << std::setw(w) << timings(i,SOLVER_ADS) << "\n";
		std::cout << "}\n";

		std::cout << "\n{\n";
		std::cout << "% || uh - uH ||\n";
		errors_L2.PrintMatlab(std::cout);
		std::cout << "% || uh ||\n";
		norm_L2.Print(std::cout, nLevels);
		std::cout << "% || curl ( uh - uH ) ||\n";
		errors_div.PrintMatlab(std::cout);
		std::cout << "% || curl uh ||\n";
		norm_div.Print(std::cout, nLevels);
		std::cout << "}\n";
	}


	for(int i(0); i < targets.Size(); ++i)
		delete targets[i];

	for(int i(0); i < help.Size(); ++i)
	{
		delete help[i];
		delete sol_AMGe[i];
		delete sol_ADS[i];
	}

	for(int i(0); i < Ml.Size(); ++i)
	{
		delete Ml[i];
		delete Wl[i];
	}

	for(int i(1); i < nLevels; ++i)
	{
		delete rhs[i];
		delete ess_data[i];
	}


	for(int i(0); i < sequence.Size(); ++i)
		delete sequence[i];

	//----------------------------------------------------//
	for(int i(0); i < topology.Size(); ++i)
		delete topology[i];

	delete pmesh;

	MPI_Finalize();

	return 0;
}
