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

enum {TOPOLOGY=0, SEQUENCE, ASSEMBLY, PRECONDITIONER, SOLVER, NSTAGES};


// Analytical Solution: http://www.phys.uri.edu/~gerhard/PHY204/tsl94.pdf. Q = 1., k = 1, R = 1
void electricField(const Vector & x, Vector & y)
{
	y = x;
	double r = y.Norml2();

	if( r > 1.)
	{
		double r3 = r*r*r;
		y *= 1./(3.*r3);
	}
	else
		y *= 1./3.;
}

double electricField_r(Vector & x)
{

	double r = x.Norml2();

	if( r > 1.)
		return 1./(3.*r*r);
	else
		return r/3.;
}

double electricPotential(Vector & x)
{
	double r = x.Norml2();
	if(r > 1.)
		return 1./(3.*r);
	else
		return .5*( 1. - (r*r)/3. );
}

/*
double source(Vector & x)
{
	double r = x.Norml2();
	if( r >= 1.)
		return 0.;
	else
		return 1.;
}
*/

Mesh* cartesian_bfly_mesh(int n1, int n2)
{
   Mesh *mesh;

   int i, nf, nr, vi[8];
   double v[3];

   nr = n1+n2;
   mesh = new Mesh(3,8+nr*7, 1+3*nr,6+6*nr );

   // the cubic element at the origin
   v[0]=0; v[1]=0; v[2]=0; mesh->AddVertex(v);
   v[0]=1; v[1]=0; v[2]=0; mesh->AddVertex(v);
   v[0]=1; v[1]=1; v[2]=0; mesh->AddVertex(v);
   v[0]=0; v[1]=1; v[2]=0; mesh->AddVertex(v);
   v[0]=0; v[1]=0; v[2]=1; mesh->AddVertex(v);
   v[0]=1; v[1]=0; v[2]=1; mesh->AddVertex(v);
   v[0]=1; v[1]=1; v[2]=1; mesh->AddVertex(v);
   v[0]=0; v[1]=1; v[2]=1; mesh->AddVertex(v);

   for (i = 0; i < 8; i++) vi[i]=i; mesh->AddHex(vi,1);

   vi[0]=0; vi[1]=1; vi[2]=5; vi[3]=4; mesh->AddBdrQuad(vi,1);
   vi[0]=0; vi[1]=3; vi[2]=7; vi[3]=4; mesh->AddBdrQuad(vi,2);
   vi[0]=0; vi[1]=1; vi[2]=2; vi[3]=3; mesh->AddBdrQuad(vi,3);

   // interface location
   nf = n1;  // at the transition between the radial and non-radial part
   // nf = n1+n2/2; // in the middle of the radial part

   // the x/y/z-extruded outer shells
   for (i = 1; i <= nr; i++)
   {
      // vertices
      v[0]=i+1; v[1]=0;   v[2]=0;   mesh->AddVertex(v);
      v[0]=i+1; v[1]=i+1; v[2]=0;   mesh->AddVertex(v);
      v[0]=0;   v[1]=i+1; v[2]=0;   mesh->AddVertex(v);
      v[0]=0;   v[1]=0;   v[2]=i+1; mesh->AddVertex(v);
      v[0]=i+1; v[1]=0;   v[2]=i+1; mesh->AddVertex(v);
      v[0]=i+1; v[1]=i+1; v[2]=i+1; mesh->AddVertex(v);
      v[0]=0;   v[1]=i+1; v[2]=i+1; mesh->AddVertex(v);

      // elements
      vi[0]=-6+7*i;  vi[1]=vi[0]+7; vi[5]=vi[0]+11; vi[4]=vi[0]+4;
      vi[3]=vi[0]+1; vi[2]=vi[0]+8; vi[6]=vi[0]+12; vi[7]=vi[0]+5;
      if (i <= nf) mesh->AddHex(vi,1); else mesh->AddHex(vi,2);
      vi[0]=-5+7*i;  vi[1]=vi[0]+7; vi[5]=vi[0]+11; vi[4]=vi[0]+4;
      vi[3]=vi[0]+1; vi[2]=vi[0]+8; vi[6]=vi[0]+12; vi[7]=vi[0]+5;
      if (i <= nf) mesh->AddHex(vi,1); else mesh->AddHex(vi,2);
      vi[0]=-3+7*i;  vi[1]=vi[0]+1; vi[5]=vi[0]+8; vi[4]=vi[0]+7;
      vi[3]=vi[0]+3; vi[2]=vi[0]+2; vi[6]=vi[0]+9; vi[7]=vi[0]+10;
      if (i <= nf) mesh->AddHex(vi,1); else mesh->AddHex(vi,2);

      // boundary faces
      vi[0]=-6+7*i; vi[1]=vi[0]+7; vi[2]=vi[0]+11; vi[3]=vi[0]+4; mesh->AddBdrQuad(vi,1);
      vi[0]=-3+7*i; vi[1]=vi[0]+1; vi[2]=vi[0]+8;  vi[3]=vi[0]+7; mesh->AddBdrQuad(vi,1);
      vi[0]=-4+7*i; vi[1]=vi[0]+7; vi[2]=vi[0]+11; vi[3]=vi[0]+4; mesh->AddBdrQuad(vi,2);
      vi[0]=-3+7*i; vi[1]=vi[0]+3; vi[2]=vi[0]+10; vi[3]=vi[0]+7; mesh->AddBdrQuad(vi,2);
      vi[0]=-6+7*i; vi[1]=vi[0]+7; vi[2]=vi[0]+8;  vi[3]=vi[0]+1; mesh->AddBdrQuad(vi,3);
      vi[0]=-4+7*i; vi[1]=vi[0]-1; vi[2]=vi[0]+6;  vi[3]=vi[0]+7; mesh->AddBdrQuad(vi,3);
   }

   i--;
   // the exterior boundary faces with ||v||_\infty=nr+1
   vi[0]=3+7*i; vi[1]=vi[0]-1; vi[2]=vi[0]+3; vi[3]=vi[0]+4; mesh->AddBdrQuad(vi,4);
   vi[0]=1+7*i; vi[1]=vi[0]+1; vi[2]=vi[0]+5; vi[3]=vi[0]+4; mesh->AddBdrQuad(vi,4);
   vi[0]=4+7*i; vi[1]=vi[0]+1; vi[2]=vi[0]+2; vi[3]=vi[0]+3; mesh->AddBdrQuad(vi,4);

/*
   i = nf;
   // the interface betweeen material 2 and 3 ||v||_\infty=nf+1
   vi[0]=3+7*i; vi[1]=vi[0]-1; vi[2]=vi[0]+3; vi[3]=vi[0]+4;
   mesh->AddBdrQuad(vi,5);
   vi[0]=1+7*i; vi[1]=vi[0]+1; vi[2]=vi[0]+5; vi[3]=vi[0]+4;
   mesh->AddBdrQuad(vi,5);
   vi[0]=4+7*i; vi[1]=vi[0]+1; vi[2]=vi[0]+2; vi[3]=vi[0]+3;
   mesh->AddBdrQuad(vi,5);
*/
   mesh->FinalizeHexMesh(1);

   return mesh;
}

int n1 = 4;
int n2 = 4;
double R1 = 1.0;     // max radius of the mapped intermediate shell
double R2 = 1.2;     // max radius of the mapped outer shell
double R0 = R2/6.0;  // scaling for the cube at the origin

// This function maps the point p with L_inf norm between r_in and r_out to the
// point v with L_2 norm between R_in and R_out.
//
// The mapping is more involved (and nicer) than simple radial projection,
// because the surface elements on the outer shell are "uniform", in the sense
// that they are based on vertices obtained by uniform splittings of the 3
// interface arks which are then connected with great circles (the geodesics on
// the sphere).
//
// Specifically, in each of the the extruded shells we introduce a spherical
// coordinate system based on two angles that can be used to easily specify
// uniform subdivisions of the needed great circles. For example, in the
// z-extruded outer shell, we introduce thetax=acos(x/r) and thetay=acos(y/r)
// and then replace them with appropriate uniform expressions of y/z and x/z.
// In the simplest case, y=0, we get thetax = pi/2-pi/4*x/z, but the formulas
// get more involved for x>y>0. The final mapping is obtained by substituting
// these formulas into x=r*cos(thetax), y=r*cos(thetay), z=r*sqrt(1-x^2-y^2).
void uniform_shell_transformation(const Vector &p, Vector &v,
                                  double r_in, double r_out,
                                  double R_in, double R_out)
{
   v = p;

   if (v.Norml2() == 0) // this probably shouldn't happen
   {
      v = 0.0;
      return;
   }

   double lambda, r;
   lambda = (v.Normlinf()-r_in)/(r_out-r_in);

   double x = p(0), y = p(1), z = p(2);
   double thetax, thetay, thetaz; // angles to the x-, y- and z-axis
   const double M_ACOS_3 = M_PI_2 - acos(1/sqrt(3));

   if (z >= x && z >= y) // z-extruded outer shell
   {
      thetax = M_PI_2-(M_PI_2-acos(M_SQRT1_2*sin(M_PI_2-M_ACOS_3*y/z)))*x/z;
      thetay = M_PI_2-(M_PI_2-acos(M_SQRT1_2*sin(M_PI_2-M_ACOS_3*x/z)))*y/z;
      v(0) = cos(thetax);
      v(1) = cos(thetay);
      v(2) = sqrt(1-v(0)*v(0)-v(1)*v(1));
   }
   else if (x >= y && x >= z) // x-extruded outer shell
   {
      thetay = M_PI_2-(M_PI_2-acos(M_SQRT1_2*sin(M_PI_2-M_ACOS_3*z/x)))*y/x;
      thetaz = M_PI_2-(M_PI_2-acos(M_SQRT1_2*sin(M_PI_2-M_ACOS_3*y/x)))*z/x;
      v(1) = cos(thetay);
      v(2) = cos(thetaz);
      v(0) = sqrt(1-v(1)*v(1)-v(2)*v(2));
   }
   else if (y >= z && y >= x) // y-extruded outer shell
   {
      thetaz = M_PI_2-(M_PI_2-acos(M_SQRT1_2*sin(M_PI_2-M_ACOS_3*x/y)))*z/y;
      thetax = M_PI_2-(M_PI_2-acos(M_SQRT1_2*sin(M_PI_2-M_ACOS_3*z/y)))*x/y;
      v(2) = cos(thetaz);
      v(0) = cos(thetax);
      v(1) = sqrt(1-v(2)*v(2)-v(0)*v(0));
   }

   r = (1-lambda)*R_in + lambda*R_out;
   v *= r;
}

void bfly_transformation(const Vector &p, Vector &v)
{
   v = p;

   if (v.Normlinf() <= 1)
      v *= R0;
   else if (v.Normlinf() > 1 && v.Normlinf() <= n1+1)
   {
      double lambda = (v.Normlinf()-1.0)/n1;
      double lambdap = pow(lambda,0.5);

      // interpolate between the two mappings
      uniform_shell_transformation(p,v,1,n1+1,R0,R1);
      add((1-lambdap)*R0,p,lambdap,v,v);

      // make the r-spacing uniform
      v *= (1-lambda)*R0/v.Normlinf() + lambda*R1/v.Norml2();
   }
   else
      uniform_shell_transformation(p,v,n1+1,n1+n2+1,R1,R2);
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

	StopWatch chrono;

	//PROGRAM OPTIONS
	int ser_ref_levels = cmm_line("--nref_serial", 0);
	int par_ref_levels = cmm_line("--nref_parallel", 2);
	int coarseringFactor = cmm_line("--coarseringFactor", 8);
	int feorder = cmm_line("--feorder", 0);
	int upscalingOrder = cmm_line("--upscalingorder", feorder);

	//DEFAULTED LINEAR SOLVER OPTIONS
	int print_iter = (myid==0);
	int max_num_iter = 1000;
	double rtol = 1e-6;
	double atol = 1e-20;

	Mesh *mesh;

	{
		std::stringstream msg;
		msg << "Refine mesh in serial "   <<   ser_ref_levels << " times. \n";
		msg << "Refine mesh in parallel "   <<   par_ref_levels << " times. \n";
		msg << "Unstructured coarsening factor " << coarseringFactor << "\n";
		msg << "FE order " << feorder << "\n";
		msg << "Upscaling order " << upscalingOrder << "\n";

		msg << "MINRES: Max Number Iterations " << max_num_iter << "\n";
		msg << "MINRES: rtol " << rtol << "\n";
		msg << "MINRES: atol " << atol << "\n";

		RootOutput(comm, 0, std::cout, msg.str());
	}

	// 2. Read the (serial) mesh from the given mesh file and uniformely refine it.
	mesh = cartesian_bfly_mesh(n1, n2);

	for (int l = 0; l < ser_ref_levels; l++)
		mesh->UniformRefinement();

	int nDimensions = mesh->Dimension();
	elag_assert(nDimensions == 3);

	{
	FiniteElementCollection *fec = new H1_FECollection(feorder+1, 3);
	FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, 3);
	mesh->SetNodalFESpace(fespace);
	}

	{
		std::ofstream m("bfly_mesh.mesh");
		m << *mesh;
	}

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

	pmesh->Transform(bfly_transformation);

	if(nDimensions == 3)
		pmesh->ReorientTetMesh();

	Vector source(2);
	source(0) = 1.; source(1) = 0.;
	PWConstCoefficient source_coeff(source);
	FunctionCoefficient potential_coeff(electricPotential);
	VectorFunctionCoefficient efield_coeff(nDimensions, electricField);
	FunctionCoefficient efield_r_coeff(electricField_r);
	FunctionCoefficient fbdr(electricPotential);


	ConstantCoefficient coeffL2(1.);
	ConstantCoefficient coeffHdiv(1.);

	MFEMRefinedMeshPartitioner partitioner(nDimensions);
	Array<AgglomeratedTopology *> topology(nLevels);
	topology = static_cast<AgglomeratedTopology *>(NULL);

	DenseMatrix timings(nLevels, NSTAGES);
	timings = 0.0;

	chrono.Clear();
	chrono.Start();
	topology[0] = new AgglomeratedTopology( pmesh, 2 );
	chrono.Stop();
	timings(0, TOPOLOGY) = chrono.RealTime();

	for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
	{
		chrono.Clear();
		chrono.Start();
		Array<int> partitioning(topology[ilevel]->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT));
		partitioner.Partition(topology[ilevel]->GetNumberLocalEntities(AgglomeratedTopology::ELEMENT), level_nElements[ilevel+1], partitioning);
		topology[ilevel+1] = topology[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);
		chrono.Stop();
		timings(ilevel+1, TOPOLOGY) = chrono.RealTime();
		if(myid == 0)
			std::cout<<"Timing ELEM_AGG_Level " << ilevel <<": Mesh Agglomeration done in " << chrono.RealTime() << " seconds \n";
	}

	//-----------------------------------------------------//

	double tolSVD = 1e-9;
	Array<DeRhamSequence *> sequence(topology.Size() );

	chrono.Clear();
	chrono.Start();
	sequence[0] = new DeRhamSequence3D_FE(topology[0], pmesh, feorder);

	Array< MultiVector *> targets( sequence[0]->GetNumberOfForms() );

//	int jFormStart = nDimensions-1;
//	sequence[0]->SetjformStart( jFormStart );

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


//	sequence[0]->SetjformStart(2);
	sequence[0]->SetTargets( targets );
	chrono.Stop();
	timings(0, SEQUENCE) = chrono.RealTime();

	for(int i(0); i < nLevels-1; ++i)
	{
		sequence[i]->SetSVDTol( tolSVD );
		chrono.Clear();
		chrono.Start();
		sequence[i+1] = sequence[i]->Coarsen();
		chrono.Stop();
		timings(i+1, SEQUENCE) = chrono.RealTime();
		if(myid == 0)
			std::cout<<"Timing ELEM_AGG_LEVEL"<<i<<": Coarsening done in " << chrono.RealTime() << " seconds \n";
	}


	int uform = pmesh->Dimension() - 1;
	int pform = pmesh->Dimension();

	//	testUpscalingHdiv(sequence);
	FiniteElementSpace * ufespace = sequence[0]->FemSequence()->GetFeSpace(uform);
	FiniteElementSpace * pfespace = sequence[0]->FemSequence()->GetFeSpace(pform);

	LinearForm b(ufespace);
	b.AddBoundaryIntegrator( new VectorFEBoundaryFluxLFIntegrator(fbdr));
	b.Assemble();
	b *= -1.;

	LinearForm q(pfespace);
	q.AddDomainIntegrator( new DomainLFIntegrator(source_coeff) );
	q.Assemble();
	q *= -1.;

	DenseMatrix u_errors_L2_2(nLevels, nLevels);
	u_errors_L2_2 = 0.0;
	DenseMatrix p_errors_L2_2(nLevels, nLevels);
	p_errors_L2_2 = 0.0;
	DenseMatrix errors_div_2(nLevels, nLevels);
	errors_div_2 = 0.0;

	DenseMatrix analytical_errors_L2_2(nLevels,3);
	analytical_errors_L2_2 = 0.0;

	Vector solutionNorm_2(3), solutionNorm(3);

	int quadrule_order = 2*feorder+2;
	Array<const IntegrationRule *> irs(Geometry::NumGeom);
	irs = NULL;
	irs[Geometry::TETRAHEDRON] = &(IntRules.Get(Geometry::TETRAHEDRON, quadrule_order));
	irs[Geometry::CUBE] = &(IntRules.Get(Geometry::CUBE, quadrule_order));

	double tmp = ComputeLpNorm(2., efield_r_coeff, *pmesh, irs);
	solutionNorm_2(0) = tmp*tmp;
	tmp = ComputeLpNorm(2., source_coeff, *pmesh, irs);
	solutionNorm_2(1) = tmp*tmp;
	tmp = ComputeLpNorm(2., potential_coeff, *pmesh, irs);
	solutionNorm_2(2) = tmp*tmp;

	MPI_Reduce(solutionNorm_2.GetData(), solutionNorm.GetData(), solutionNorm.Size(), MPI_DOUBLE,MPI_SUM, 0, comm);
	if(myid == 0)
		std::transform(solutionNorm.GetData(), solutionNorm.GetData()+solutionNorm.Size(), solutionNorm.GetData(), (double(*)(double)) sqrt);

	Array<int> iter(nLevels);
	iter = 0;
	Array<int> ndofs(nLevels);
	ndofs = 0;

	double tdiff;

	Array<SparseMatrix *> allPu(nLevels-1);
	Array<SparseMatrix *> allPp(nLevels-1);
	Array<SparseMatrix *> allD(nLevels);

	for(int i = 0; i < nLevels - 1; ++i)
	{
		allPu[i] = sequence[i]->GetP(uform);
		allPp[i] = sequence[i]->GetP(pform);
	}

	for(int i = 0; i < nLevels; ++i)
		allD[i] = sequence[i]->GetDerivativeOperator(uform);

	Array<SparseMatrix *> Ml(nLevels);
	Array<SparseMatrix *> Wl(nLevels);

	for(int k(0); k < nLevels; ++k)
	{
		chrono.Clear();
		chrono.Start();
		Ml[k] = sequence[k]->ComputeMassOperator(uform);
		Wl[k] = sequence[k]->ComputeMassOperator(pform);
		chrono.Stop();
		tdiff = chrono.RealTime();
		if(myid == 0)
			std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in " << tdiff << "s. \n";
		timings(k, ASSEMBLY) += tdiff;
	}

	Array< Array<int>* > blockOffsets(nLevels);
	for(int k(0); k < nLevels; ++k)
	{
		blockOffsets[k] = new Array<int>(3);
		int * p = blockOffsets[k]->GetData();
		p[0] = 0;
		p[1] = sequence[k]->GetNumberOfDofs(uform);
		p[2] = p[1] + sequence[k]->GetNumberOfDofs(pform);
	}

	Array<BlockVector *> rhs(nLevels);
	rhs[0] = new BlockVector( *(blockOffsets[0]) );
	rhs[0]->GetBlock(0) = b;
	rhs[0]->GetBlock(1) = q;

	for(int i = 0; i < nLevels-1; ++i)
	{
		rhs[i+1] = new BlockVector( *(blockOffsets[i+1]) );
		allPu[i]->MultTranspose(rhs[i]->GetBlock(0), rhs[i+1]->GetBlock(0) );
		allPp[i]->MultTranspose(rhs[i]->GetBlock(1), rhs[i+1]->GetBlock(1) );
	}

	Array<BlockVector *> sol(nLevels);
	Array<BlockVector *> help(nLevels);

	sol = static_cast<BlockVector *>(NULL);
	help     = static_cast<BlockVector *>(NULL);

	for(int k(0); k < nLevels; ++k)
	{
		sol[k]   = new BlockVector( *(blockOffsets[k]) );
		help[k]  = new BlockVector( *(blockOffsets[k]) );
		*(help[k]) = 0.;
	}

	for(int k(0); k < nLevels; ++k)
	{
		chrono.Clear();
		chrono.Start();
		SparseMatrix * M = Ml[k];
		SparseMatrix * W = Wl[k];
		SparseMatrix * D = allD[k];
		SparseMatrix * B = Mult(*W, *D);
		*B *= -1.;
		SparseMatrix * Bt = Transpose(*B);

		{
			std::stringstream fname;
			fname << "B"<<k<<".mtx\n";
			std::ofstream fid(fname.str().c_str());
			B->PrintMatlab(fid);
		}

		const SharingMap & l2_dofTrueDof( sequence[k]->GetDofHandler(pform)->GetDofTrueDof() );
		const SharingMap & hdiv_dofTrueDof( sequence[k]->GetDofHandler(uform)->GetDofTrueDof() );

		Array<int> trueBlockOffsets(3);
		trueBlockOffsets[0] = 0;
		trueBlockOffsets[1] = hdiv_dofTrueDof.GetTrueLocalSize();
		trueBlockOffsets[2] = trueBlockOffsets[1] + l2_dofTrueDof.GetTrueLocalSize();
		ndofs[k] = trueBlockOffsets.Last();


		BlockVector prhs( trueBlockOffsets );
		hdiv_dofTrueDof.Assemble(rhs[k]->GetBlock(0), prhs.GetBlock(0) );
		l2_dofTrueDof.Assemble(rhs[k]->GetBlock(1), prhs.GetBlock(1) );

		HypreParMatrix * pM = Assemble(hdiv_dofTrueDof, *M, hdiv_dofTrueDof);
		HypreParMatrix * pB = Assemble(l2_dofTrueDof, *B, hdiv_dofTrueDof);
		HypreParMatrix * pBt = Assemble(hdiv_dofTrueDof, *Bt, l2_dofTrueDof);

		BlockOperator op(trueBlockOffsets);
		op.owns_blocks = 1;
		op.SetBlock(0,0,pM);
		op.SetBlock(0,1, pBt);
		op.SetBlock(1,0, pB);

		chrono.Stop();
		tdiff = chrono.RealTime();
		if(myid == 0)
			std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in " << tdiff << "s. \n";
		timings(k, ASSEMBLY) += tdiff;

#if 0
		HypreParMatrix * tmp = Assemble(hdiv_dofTrueDof, *Bt, l2_dofTrueDof);
		Vector diag( pM->Height() );
		pM->GetDiag(diag);

		for(int i =0; i < diag.Size(); ++i)
			diag(i) = 1./diag(i);

		tmp->ScaleRows(diag);
		HypreParMatrix * S = ParMult(pB, tmp);
		delete tmp;

		HypreDiagScale *Mprec = new HypreDiagScale(*pM);
		HypreExtension::HypreBoomerAMG *Sprec = new HypreExtension::HypreBoomerAMG(*S);

		BlockDiagonalPreconditioner prec(trueBlockOffsets);
		prec.owns_blocks = 1;
		prec.SetDiagonalBlock(0,Mprec);
		prec.SetDiagonalBlock(1,Sprec);

		//Solver
		{
			chrono.Clear();
			chrono.Start();

			{
				BlockVector x( trueBlockOffsets );
				BlockVector y( trueBlockOffsets );
				x = 1.0; y = 0.;
				prec.Mult(x,y);
			}

			chrono.Stop();
			tdiff = chrono.RealTime();

			if(myid == 0)
				std::cout << "Timing PRECONDITIONER_LEVEL " << k << ": Preconditioner Computed " << tdiff << "s. \n";
			timings(k,PRECONDITIONER) = tdiff;


			BlockVector psol( trueBlockOffsets );
			psol = 0.;
			MINRESSolver minres(comm);
			minres.SetPrintLevel(print_iter);
			minres.SetMaxIter(max_num_iter);
			minres.SetRelTol(rtol);
			minres.SetAbsTol(atol);
			minres.SetOperator(op );
			minres.SetPreconditioner( prec );
			chrono.Clear();
			chrono.Start();
			minres.Mult(prhs, psol );
			chrono.Stop();
			tdiff = chrono.RealTime();
			if(myid == 0)
				std::cout << "Timing MINRES_LEVEL " << k << ": Solver done in " << tdiff << "s. \n";
			timings(k,SOLVER) = tdiff;

			if(myid == 0)
			{
				if(minres.GetConverged())
					std::cout << "PCG converged in " << minres.GetNumIterations() << " with a final residual norm " << minres.GetFinalNorm() << "\n";
				else
					std::cout << "PCG did not converge in " << minres.GetNumIterations() << ". Final residual norm is " << minres.GetFinalNorm() << "\n";
			}

			hdiv_dofTrueDof.Distribute(psol.GetBlock(0), sol[k]->GetBlock(0));
			l2_dofTrueDof.Distribute(psol.GetBlock(1), sol[k]->GetBlock(1));
			iter[k] = minres.GetNumIterations();
		}

		delete S;
#else
		BlockMatrix mat( *(blockOffsets[k]) );
		mat.owns_blocks = 0;
		mat.SetBlock(0,0,M);
		mat.SetBlock(0,1, Bt);
		mat.SetBlock(1,0, B);

		Array<int> essbdr(pmesh->bdr_attributes.Max());
		essbdr= 0;
		Array<DeRhamSequence *> actseq(sequence.GetData()+k, sequence.Size() - k);

		chrono.Clear();
		chrono.Start();

		MLDivFree prec(&mat, actseq, essbdr);
		BlockVector psol( trueBlockOffsets );
		psol = 0.;

		elag_assert( prhs.CheckFinite() == 0);
		prec.Mult(prhs, psol);

		chrono.Stop();
		tdiff = chrono.RealTime();

		if(myid == 0)
			std::cout << "Timing PRECONDITIONER_LEVEL " << k << ": Preconditioner Computed " << tdiff << "s. \n";
		timings(k,PRECONDITIONER) = tdiff;

		FGMRESSolver cg(comm);
		cg.iterative_mode = true;
		cg.SetPrintLevel(print_iter);
		cg.SetMaxIter(max_num_iter);
		cg.SetRelTol(rtol);
		cg.SetAbsTol(atol);
		cg.SetOperator(op );
		cg.SetPreconditioner( prec );
		chrono.Clear();
		chrono.Start();
		cg.Mult(prhs, psol );
		chrono.Stop();
		tdiff = chrono.RealTime();
		if(myid == 0)
			std::cout << "Timing MINRES_LEVEL " << k << ": Solver done in " << tdiff << "s. \n";
		timings(k,SOLVER) = tdiff;

		if(myid == 0)
		{
			if(cg.GetConverged())
				std::cout << "PCG converged in " << cg.GetNumIterations() << " with a final residual norm " << cg.GetFinalNorm() << "\n";
			else
				std::cout << "PCG did not converge in " << cg.GetNumIterations() << ". Final residual norm is " << cg.GetFinalNorm() << "\n";
		}
		iter[k] = cg.GetNumIterations();


		hdiv_dofTrueDof.Distribute(psol.GetBlock(0), sol[k]->GetBlock(0));
		l2_dofTrueDof.Distribute(psol.GetBlock(1), sol[k]->GetBlock(1));


#endif

		//ERROR NORMS
		{
			*(help[k]) = *(sol[k]);
			for(int j = k; j > 0; --j)
			{
				allPu[j-1]->Mult( help[j]->GetBlock(0), help[j-1]->GetBlock(0) );
				allPp[j-1]->Mult( help[j]->GetBlock(1), help[j-1]->GetBlock(1) );
			}

			Vector dsol( allD[0]->Size() );
			allD[0]->Mult(help[0]->GetBlock(0), dsol );

			GridFunction gf;
			gf.Update(ufespace,help[0]->GetBlock(0), 0);
			double err = gf.ComputeL2Error(efield_coeff);
			analytical_errors_L2_2(k,0) = err*err;
			gf.Update(pfespace,dsol, 0);
			err = gf.ComputeL2Error(source_coeff);
			analytical_errors_L2_2(k,1) = err*err;
			gf.Update(pfespace,help[0]->GetBlock(1), 0);
			err = gf.ComputeL2Error(potential_coeff, irs);
			analytical_errors_L2_2(k,2) = err*err;

			for(int j(0); j < k; ++j)
			{
				if(help[j]->Size() != sol[j]->Size() || sol[j]->GetBlock(0).Size() != allD[j]->Width() )
					mfem_error("size don't match \n");

				int usize  = sol[j]->GetBlock(0).Size();
				int psize  = sol[j]->GetBlock(1).Size();
				int dsize = allD[j]->Size();
				Vector u_H( help[j]->GetBlock(0).GetData(), usize );
				Vector u_h( sol[j]->GetBlock(0).GetData(), usize  );
				Vector p_H( help[j]->GetBlock(1).GetData(), psize );
				Vector p_h( sol[j]->GetBlock(1).GetData(), psize  );
				Vector u_diff( usize ), du_diff( dsize ), p_diff( psize );
				u_diff = 0.; du_diff = 0.; p_diff = 0.;

				subtract(u_H, u_h, u_diff);
				allD[j]->Mult(u_diff, du_diff);
				subtract(p_H, p_h, p_diff);

				u_errors_L2_2(k,j) =  Ml[j]->InnerProduct(u_diff, u_diff);
				errors_div_2(k,j) =  Wl[j]->InnerProduct(du_diff, du_diff);
				p_errors_L2_2(k,j) =  Wl[j]->InnerProduct(p_diff, p_diff);
			}
		}

		//VISUALIZE SOLUTION
		if(1)
		{
			MultiVector u(sol[k]->GetBlock(0).GetData(), 1, sol[k]->GetBlock(0).Size() );
			sequence[k]->show(uform, u);
			MPI_Barrier(comm);
			MultiVector p(sol[k]->GetBlock(1).GetData(), 1, sol[k]->GetBlock(1).Size() );
			sequence[k]->show(pform, p);
			MPI_Barrier(comm);
			MultiVector div_u(1, sol[k]->GetBlock(1).Size());
			allD[k]->Mult(sol[k]->GetBlock(0), div_u);
			sequence[k]->show(pform, div_u);
		}

		delete B;
		delete Bt;
	}

	DenseMatrix u_errors_L2(nLevels, nLevels);
	u_errors_L2 = 0.;
	DenseMatrix p_errors_L2(nLevels, nLevels);
	p_errors_L2 = 0.;
	DenseMatrix errors_div(nLevels, nLevels);
	errors_div = 0.;
	DenseMatrix analytical_errors(nLevels, 3);
	analytical_errors = 0.;
//	DenseMatrix timings(nLevels-1, NSTAGES);

	MPI_Reduce(u_errors_L2_2.Data(), u_errors_L2.Data(),u_errors_L2.Height()*u_errors_L2.Width(), MPI_DOUBLE,MPI_SUM, 0, comm);
	MPI_Reduce(p_errors_L2_2.Data(), p_errors_L2.Data(),p_errors_L2.Height()*p_errors_L2.Width(), MPI_DOUBLE,MPI_SUM, 0, comm);
	MPI_Reduce(errors_div_2.Data(), errors_div.Data(),errors_div.Height()*errors_div.Width(), MPI_DOUBLE,MPI_SUM, 0, comm);
	MPI_Reduce(analytical_errors_L2_2.Data(), analytical_errors.Data(),analytical_errors_L2_2.Height()*analytical_errors_L2_2.Width(), MPI_DOUBLE,MPI_SUM, 0, comm);

	if(myid == 0)
	{
		std::transform(u_errors_L2.Data(), u_errors_L2.Data()+u_errors_L2.Height()*u_errors_L2.Width(), u_errors_L2.Data(), (double(*)(double)) sqrt);
		std::transform(p_errors_L2.Data(), p_errors_L2.Data()+p_errors_L2.Height()*p_errors_L2.Width(), p_errors_L2.Data(), (double(*)(double)) sqrt);
		std::transform(errors_div.Data(), errors_div.Data()+errors_div.Height()*errors_div.Width(), errors_div.Data(), (double(*)(double)) sqrt);
		std::transform(analytical_errors.Data(), analytical_errors.Data()+analytical_errors.Height()*analytical_errors.Width(), analytical_errors.Data(), (double(*)(double)) sqrt);
	}

	if(myid == 0)
	{
		std::cout << "\n{\n";
		int w = 14;
		std::cout << "%level" << std::setw(w) << "Topology" << std::setw(w) << "Sequence\n";
		for(int i(0); i < nLevels; ++i)
			std::cout<< i << std::setw(w) << timings(i,TOPOLOGY) << std::setw(w) << timings(i,SEQUENCE) << "\n";
		std::cout << "}\n";

		std::cout << "\n{\n";
		std::cout << "%level" << std::setw(w) << "size" << std::setw(w) << "nit"
			<< std::setw(w) << "Assembly" << std::setw(w) << "Prec "
			<< std::setw(w) << "Solver\n";
		for(int i(0); i < nLevels; ++i)
			std::cout<< i << std::setw(w) << ndofs[i] << std::setw(w) << iter[i]
		         << std::setw(w) << timings(i,ASSEMBLY) << std::setw(w) << timings(i,PRECONDITIONER)
		         << std::setw(w) << timings(i,SOLVER) << "\n";
		std::cout << "}\n";

		std::cout << "\n{\n";
		std::cout << "|| u_ex ||, ||div u_ex ||. || p_ex ||\n";
		solutionNorm.Print(std::cout, solutionNorm.Size());
		std::cout << "}\n";

		std::cout << "\n{\n";
		std::cout << "|| u_ex - u_H ||, ||div u_ex - u_H ||. || p_ex - pH||\n";
		analytical_errors.PrintMatlab(std::cout);
		std::cout << "}\n";

		std::cout << "\n{\n";
		std::cout << "% || uh - uH ||\n";
		u_errors_L2.PrintMatlab(std::cout);
		std::cout << "% || ph - pH ||\n";
		p_errors_L2.PrintMatlab(std::cout);
		std::cout << "% || div ( uh - uH ) ||\n";
		errors_div.PrintMatlab(std::cout);

		std::cout << "}\n";
	}


	for(int i(0); i < nLevels; ++i)
		delete blockOffsets[i];

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

	for(int i(0); i < nLevels; ++i)
	{
		delete rhs[i];
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
