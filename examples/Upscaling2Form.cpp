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

#include <fstream>
#include <sstream>

#include <mpi.h>

#include "../src/elag.hpp"

enum {TOPOLOGY=0, SPACES, ASSEMBLY, PRECONDITIONER, SOLVER, NSTAGES};

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

	elag_trace_init(comm);

	GetPot cmm_line(argc, argv);

	//Overload options from command line
	int feorder = cmm_line("--feorder", 0);
	int upscalingOrder = cmm_line("--upscalingorder", feorder);
	std::string meshfile = cmm_line("--meshfile", "../meshes/boxcyl.mesh3d");
	int ser_ref_levels = cmm_line("--ser_ref_levels", 0);
	int par_ref_levels = cmm_line("--par_ref_levels", 4);

	//DEFAULTED LINEAR SOLVER OPTIONS
	int print_iter = 0;
	int max_num_iter = 500;
	double rtol = 1e-6;
	double atol = 1e-12;

	Mesh *mesh;

	if(myid == 0)
	{
		std::cout << "Read mesh " << meshfile   << "\n";
		std::cout << "Finite Element Order " << feorder << "\n";
		std::cout << "Upscaling Order " << upscalingOrder << "\n";
		std::cout << "Refine mesh in serial "   <<   ser_ref_levels << " times. \n";
		std::cout << "Refine mesh in parallel "   <<   par_ref_levels << " times. \n";
	}

	// 2. Read the (serial) mesh from the given mesh file and uniformely refine it.
	std::ifstream imesh(meshfile.c_str());
	if (!imesh)
	{
		if(myid == 0)
			std::cerr << "\nCan not open mesh file: " << meshfile << '\n' << std::endl;
		return 2;
	}
	mesh = new Mesh(imesh, 1, 1);
	imesh.close();
	Array<int> ess_attr(mesh->bdr_attributes.Max());
	ess_attr = 1;

	for (int l = 0; l < ser_ref_levels; l++)
		mesh->UniformRefinement();

	int nDimensions = mesh->Dimension();

	ParMesh * pmesh = new ParMesh(comm, *mesh);
	delete mesh;

	int nLevels = par_ref_levels+1;
	Array<int> level_nElements(nLevels);
	for (int l = 0; l < par_ref_levels; l++)
	{
		level_nElements[par_ref_levels-l] = pmesh->GetNE();
		pmesh->UniformRefinement();
	}
	level_nElements[0] = pmesh->GetNE();

	if(nDimensions == 3)
		pmesh->ReorientTetMesh();

	ConstantCoefficient coeffL2(1.);
	ConstantCoefficient coeffHdiv(1.);
	Vector ubdr_vect(pmesh->Dimension());
	ubdr_vect = 0.0;
	VectorConstantCoefficient ubdr(ubdr_vect);
	Vector f_vect(pmesh->Dimension());
	f_vect = 0.0;
	f_vect[nDimensions-1] = 1.0;
	VectorConstantCoefficient f(f_vect);

	DenseMatrix timings(nLevels, NSTAGES);
	timings = 0.0;

	StopWatch chrono;


	MFEMRefinedMeshPartitioner partitioner(nDimensions);
	Array<AgglomeratedTopology *> topology(nLevels);
	topology = static_cast<AgglomeratedTopology *>(NULL);

	StopWatch chronoInterior;
	chrono.Clear();
	chrono.Start();
	chronoInterior.Clear();
	chronoInterior.Start();
	elag_trace("Generate Fine Grid Topology");
	topology[0] = new AgglomeratedTopology( pmesh, nDimensions );
	elag_trace("Generate Fine Grid Topology Finished");
	chronoInterior.Stop();
	timings(0,TOPOLOGY) = chronoInterior.RealTime();
	for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
	{
		Array<int> partitioning(topology[ilevel]->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT));
		chronoInterior.Clear();
		chronoInterior.Start();
		partitioner.Partition(topology[ilevel]->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT), level_nElements[ilevel+1], partitioning);
		topology[ilevel+1] = topology[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);
		chronoInterior.Stop();
		timings(ilevel+1,TOPOLOGY) = chronoInterior.RealTime();
	}
	chrono.Stop();
	if(myid == 0)
		std::cout<<"Timing ELEM_AGG: Mesh Agglomeration done in " << chrono.RealTime() << " seconds \n";

	//-----------------------------------------------------//

	chronoInterior.Clear();
	chronoInterior.Start();
	double tolSVD = 1e-9;
	Array<DeRhamSequence *> sequence(topology.Size() );
	sequence[0] = new DeRhamSequence3D_FE(topology[0], pmesh, feorder);

	Array< MultiVector *> targets( sequence[0]->GetNumberOfForms() );

	int jFormStart = 0;
	sequence[0]->SetjformStart( jFormStart );

	dynamic_cast<DeRhamSequenceFE *>(sequence[0])->ReplaceMassIntegrator(AgglomeratedTopology::ELEMENT, 3, new MassIntegrator(coeffL2), false);
	dynamic_cast<DeRhamSequenceFE *>(sequence[0])->ReplaceMassIntegrator(AgglomeratedTopology::ELEMENT, 2, new VectorFEMassIntegrator(coeffHdiv), true);


	Array<Coefficient *> L2coeff;
	Array<VectorCoefficient *> Hdivcoeff;
	fillVectorCoefficientArray(nDimensions, upscalingOrder, Hdivcoeff);
	fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);


	int jform(0);

	targets[jform] = static_cast<MultiVector *>(NULL);
	++jform;

	targets[jform]= static_cast<MultiVector *>(NULL);
	++jform;

	targets[jform] = dynamic_cast<DeRhamSequenceFE *>(sequence[0])->InterpolateVectorTargets(jform, Hdivcoeff);
	++jform;

	targets[jform] = dynamic_cast<DeRhamSequenceFE *>(sequence[0])->InterpolateScalarTargets(jform, L2coeff);
	++jform;


	freeCoeffArray(L2coeff);
	freeCoeffArray(Hdivcoeff);


	sequence[0]->SetTargets( targets );
	chronoInterior.Stop();
	timings(0,SPACES) = chronoInterior.RealTime();

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
		timings(i+1, SPACES) = chronoInterior.RealTime();
		if(myid == 0)
			std::cout<<"Timing ELEM_AGG_LEVEL"<<i<<": Coarsering done in " << chronoInterior.RealTime() << " seconds \n";
	}
	chrono.Stop();

	if(myid == 0)
		std::cout<<"Timing ELEM_AGG: Coarsering done in " << chrono.RealTime() << " seconds \n";

	int form = 2;

	//	testUpscalingHdiv(sequence);
	FiniteElementSpace * fespace = sequence[0]->FemSequence()->GetFeSpace(form);
	LinearForm b(fespace);
	b.AddDomainIntegrator( new VectorFEDomainLFIntegrator(f));
	b.Assemble();

	GridFunction lift(fespace);
	lift.ProjectBdrCoefficientNormal(ubdr, ess_attr);

	DenseMatrix errors_L2_2(nLevels, nLevels);
	errors_L2_2 = 0.0;
	Vector norm_L2_2(nLevels);
	norm_L2_2 = 0.;

	DenseMatrix errors_div_2(nLevels, nLevels);
	errors_div_2 = 0.0;
	Vector norm_div_2(nLevels);
	norm_div_2 = 0.;

	Array<int> iter(nLevels);
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

	Array<Vector *> sol(nLevels);
	Array<Vector *> help(nLevels);

	sol  = static_cast<Vector *>(NULL);
	help     = static_cast<Vector *>(NULL);

	for(int k(0); k < nLevels; ++k)
	{
		sol[k]   = new Vector( sequence[k]->GetNumberOfDofs(form) );
		*(sol[k]) = 0.;
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

		const SharingMap & hdiv_dofTrueDof( sequence[k]->GetDofHandler(nDimensions-1)->GetDofTrueDof() );

		Vector prhs( hdiv_dofTrueDof.GetTrueLocalSize() );
		hdiv_dofTrueDof.Assemble(*(rhs[k]), prhs);
		HypreParMatrix * pA = Assemble(hdiv_dofTrueDof, *A, hdiv_dofTrueDof);

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

		HypreExtension::HypreADSData data;
//		data.cycle_type = 1;
		data.dataAMG.theta = 0.6;
//		data.dataAMG.agg_levels = 0;
//		data.dataAMS.cycle_type = 1;
		data.dataAMS.dataAlpha.theta = 0.6;
//		data.dataAMS.dataAlpha.agg_levels = 0;

		chrono.Clear();
		chrono.Start();
//		HypreSmoother prec(*pA, HypreSmoother::Jacobi);
		HypreExtension::HypreADS prec(*pA, sequence[k], data);
		Vector tmp1(pA->Height()), tmp2(pA->Height() );
		tmp1 = 1.; tmp2 = 2.;
		prec.Mult(tmp1, tmp2);
		chrono.Stop();
		tdiff = chrono.RealTime();
		if(myid == 0)
			std::cout << "Timing LEVEL " << k << ": Preconditioner done in " << tdiff << "s. \n";
		timings(k, PRECONDITIONER) = tdiff;

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
			std::cout << "Timing LEVEL " << k << ": Solver done in " << tdiff << "s. \n";
		timings(k,SOLVER) = tdiff;

		if(myid == 0)
		{
			if(pcg.GetConverged())
				std::cout << "PCG converged in " << pcg.GetNumIterations() << " with a final residual norm " << pcg.GetFinalNorm() << "\n";
			else
				std::cout << "PCG did not converge in " << pcg.GetNumIterations() << ". Final residual norm is " << pcg.GetFinalNorm() << "\n";
		}

		hdiv_dofTrueDof.Distribute(psol, *(sol[k]));
		iter[k] = pcg.GetNumIterations();
		
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
					  << "nit" << std::setw(w) << "Topology" << std::setw(w) << "TSpaces"
					 << std::setw(w) << "Assembly" << std::setw(w)  << "Preconditioner"
					 << std::setw(w) << "Solver\n";
		for(int i(0); i < nLevels; ++i)
			std::cout << i << std::setw(w) << ndofs[i] << std::setw(w) << nnz[i] << std::setw(w) 
						 << iter[i]
						 << std::setw(w) << timings(i,TOPOLOGY)
						 << std::setw(w) << timings(i,SPACES)
						 << std::setw(w) << timings(i,ASSEMBLY) << std::setw(w) << timings(i,PRECONDITIONER)
						 << std::setw(w) << timings(i,SOLVER) << "\n";
		std::cout << "}\n";

		std::cout << "\n{\n";
		std::cout << "% || uh - uH ||\n";
		errors_L2.PrintMatlab(std::cout);
		std::cout << "% || uh ||\n";
		norm_L2.Print(std::cout, nLevels);
		std::cout << "% || div ( uh - uH ) ||\n";
		errors_div.PrintMatlab(std::cout);
		std::cout << "% || div uh ||\n";
		norm_div.Print(std::cout, nLevels);
		std::cout << "}\n";
	}


	for(int i(0); i < targets.Size(); ++i)
		delete targets[i];

	for(int i(0); i < help.Size(); ++i)
	{
		delete help[i];
		delete sol[i];
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
