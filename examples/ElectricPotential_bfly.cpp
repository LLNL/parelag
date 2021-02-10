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

/**
   This particular example is interesting because it is the only
   example using the div-free solver.
*/

#include <fstream>
#include <memory>
#include <sstream>

#include <mpi.h>

#include "elag.hpp"

using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

using namespace mfem;

enum {TOPOLOGY=0, SEQUENCE, ASSEMBLY, PRECONDITIONER, SOLVER};

const int NSTAGES = 5;
const char * stage_names[] = {"TOPOLOGY", "SEQUENCE", "ASSEMBLY",
                              "PRECONDITIONER", "SOLVER"};

// Analytical Solution: http://www.phys.uri.edu/~gerhard/PHY204/tsl94.pdf.
// Q = 1., k = 1, R = 1
void electricField(const Vector & x, Vector & y)
{
    y = x;
    double r = y.Norml2();

    if (r > 1.)
    {
        double r3 = r*r*r;
        y *= 1./(3.*r3);
    }
    else
        y *= 1./3.;
}

double electricField_r(const Vector & x)
{

    double r = x.Norml2();

    if ( r > 1.)
        return 1./(3.*r*r);
    else
        return r/3.;
}

double electricPotential(const Vector & x)
{
    double r = x.Norml2();
    if (r > 1.)
        return 1./(3.*r);
    else
        return .5*( 1. - (r*r)/3. );
}

unique_ptr<Mesh> cartesian_bfly_mesh(int n1, int n2)
{
    int i, nf, nr, vi[8];
    double v[3];

    nr = n1+n2;
    auto mesh = make_unique<mfem::Mesh>(3, 8+nr*7, 1+3*nr, 6+6*nr);

    // the cubic element at the origin
    v[0]=0; v[1]=0; v[2]=0; mesh->AddVertex(v);
    v[0]=1; v[1]=0; v[2]=0; mesh->AddVertex(v);
    v[0]=1; v[1]=1; v[2]=0; mesh->AddVertex(v);
    v[0]=0; v[1]=1; v[2]=0; mesh->AddVertex(v);
    v[0]=0; v[1]=0; v[2]=1; mesh->AddVertex(v);
    v[0]=1; v[1]=0; v[2]=1; mesh->AddVertex(v);
    v[0]=1; v[1]=1; v[2]=1; mesh->AddVertex(v);
    v[0]=0; v[1]=1; v[2]=1; mesh->AddVertex(v);

    for (i = 0; i < 8; i++) vi[i]=i;
    mesh->AddHex(vi,1);

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
        vi[0]=-6+7*i; vi[1]=vi[0]+7; vi[2]=vi[0]+11; vi[3]=vi[0]+4;
        mesh->AddBdrQuad(vi,1);
        vi[0]=-3+7*i; vi[1]=vi[0]+1; vi[2]=vi[0]+8;  vi[3]=vi[0]+7;
        mesh->AddBdrQuad(vi,1);
        vi[0]=-4+7*i; vi[1]=vi[0]+7; vi[2]=vi[0]+11; vi[3]=vi[0]+4;
        mesh->AddBdrQuad(vi,2);
        vi[0]=-3+7*i; vi[1]=vi[0]+3; vi[2]=vi[0]+10; vi[3]=vi[0]+7;
        mesh->AddBdrQuad(vi,2);
        vi[0]=-6+7*i; vi[1]=vi[0]+7; vi[2]=vi[0]+8;  vi[3]=vi[0]+1;
        mesh->AddBdrQuad(vi,3);
        vi[0]=-4+7*i; vi[1]=vi[0]-1; vi[2]=vi[0]+6;  vi[3]=vi[0]+7;
        mesh->AddBdrQuad(vi,3);
    }

    i--;
    // the exterior boundary faces with ||v||_\infty=nr+1
    vi[0]=3+7*i; vi[1]=vi[0]-1; vi[2]=vi[0]+3; vi[3]=vi[0]+4;
    mesh->AddBdrQuad(vi,4);
    vi[0]=1+7*i; vi[1]=vi[0]+1; vi[2]=vi[0]+5; vi[3]=vi[0]+4;
    mesh->AddBdrQuad(vi,4);
    vi[0]=4+7*i; vi[1]=vi[0]+1; vi[2]=vi[0]+2; vi[3]=vi[0]+3;
    mesh->AddBdrQuad(vi,4);

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
    // 1. Initialize MPI
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    StopWatch chrono;

    // program options
    const char* meshfile_c = "../meshes/sphere_in_sphere_25K.mesh3D";
    int ser_ref_levels = 0;
    int par_ref_levels = 2;
    int coarseningFactor = 8;
    int feorder = 0;
    int upscalingOrder = 0;
    bool do_visualize = true;
    OptionsParser args(argc, argv);
    args.AddOption(&meshfile_c, "-m", "--mesh",
                   "MFEM mesh file to load.");
    args.AddOption(&ser_ref_levels, "-sr", "--nref_serial",
                   "Number of times to refine serial mesh.");
    args.AddOption(&par_ref_levels, "-pr", "--nref_parallel",
                   "Number of times to refine parallel mesh.");
    args.AddOption(&coarseningFactor, "-cf", "--coarseningFactor",
                   "Geometric coarsening ratio.");
    args.AddOption(&feorder, "-feo", "--feorder",
                   "Polynomial order of fine finite element space.");
    args.AddOption(&upscalingOrder, "-uo", "--upscalingorder",
                   "Polynomial order of coarse space.");
    args.AddOption(&do_visualize, "-v", "--do-visualize", "-nv", "--no-visualize",
                   "Do interactive GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(std::cout);
        MPI_Finalize();
        return 1;
    }
    PARELAG_ASSERT(args.Good());
    std::string meshfile(meshfile_c);

    // default linear solver options
    const int print_iter = (myid==0);
    constexpr int max_num_iter = 1000;
    constexpr double rtol = 1e-6;
    constexpr double atol = 1e-20;

    {
        std::stringstream msg;
        msg << "Refine mesh in serial "   <<   ser_ref_levels << " times.\n";
        msg << "Refine mesh in parallel "   <<   par_ref_levels << " times.\n";
        msg << "Unstructured coarsening factor " << coarseningFactor << "\n";
        msg << "FE order " << feorder << "\n";
        msg << "Upscaling order " << upscalingOrder << "\n";

        msg << "MINRES: Max Number Iterations " << max_num_iter << "\n";
        msg << "MINRES: rtol " << rtol << "\n";
        msg << "MINRES: atol " << atol << "\n";

        RootOutput(comm, 0, std::cout, msg.str());
    }

    // build a Cartesian butterfly mesh
    shared_ptr<ParMesh> pmesh;
    {
        auto mesh = cartesian_bfly_mesh(n1, n2);

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        elag_assert(mesh->Dimension() == 3);

        FiniteElementCollection *fec = new H1_FECollection(feorder+1, 3);
        FiniteElementSpace *fespace = new FiniteElementSpace(mesh.get(), fec, 3);
        mesh->SetNodalFESpace(fespace);

        if (false)
        {
            std::ofstream meshstream("bfly_mesh.mesh");
            meshstream << *mesh;
        }

        pmesh = make_shared<mfem::ParMesh>(comm,*mesh);
    }

    const int nDimensions = pmesh->Dimension();
    const int nLevels = par_ref_levels+1;
    Array<int> level_nElements(nLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    level_nElements[0] = pmesh->GetNE();

    pmesh->Transform(bfly_transformation);

    if (nDimensions == 3)
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
    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    DenseMatrix timings(nLevels, NSTAGES);
    timings = 0.0;
    chrono.Clear();
    chrono.Start();
    const int codim = 2;
    topology[0] = make_shared<AgglomeratedTopology>(pmesh, codim);
    chrono.Stop();
    timings(0, TOPOLOGY) = chrono.RealTime();

    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    for (int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        chrono.Clear();
        chrono.Start();
        Array<int> partitioning(topology[ilevel]->GetNumberLocalEntities(at_elem));
        partitioner.Partition(topology[ilevel]->GetNumberLocalEntities(at_elem),
                              level_nElements[ilevel+1], partitioning);
        topology[ilevel+1] =
            topology[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);
        chrono.Stop();
        timings(ilevel+1, TOPOLOGY) = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_Level " << ilevel
                      << ": Mesh Agglomeration done in " << chrono.RealTime()
                      << " seconds.\n";
    }

    //-----------------------------------------------------//

    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());
    chrono.Clear();
    chrono.Start();
    sequence[0] = make_shared<DeRhamSequence3D_FE>(
        topology[0], pmesh.get(), feorder);

    DeRhamSequenceFE* DRSequence_FE = sequence[0]->FemSequence();

    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 3, make_unique<MassIntegrator>(coeffL2), false);
    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 2, make_unique<VectorFEMassIntegrator>(coeffHdiv), true);

    // set up coefficients / targets
    // DRSequence_FE->SetjformStart(2);
    DRSequence_FE->SetUpscalingTargets(nDimensions, upscalingOrder, 2);
    chrono.Stop();
    timings(0, SEQUENCE) = chrono.RealTime();

    constexpr double tolSVD = 1e-9;
    for (int i(0); i < nLevels-1; ++i)
    {
        sequence[i]->SetSVDTol(tolSVD);
        chrono.Clear();
        chrono.Start();
        sequence[i+1] = sequence[i]->Coarsen();
        chrono.Stop();
        timings(i+1, SEQUENCE) = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL" << i << ": Coarsening done in "
                      << chrono.RealTime() << " seconds \n";
    }

    const int uform = pmesh->Dimension() - 1;
    const int pform = pmesh->Dimension();

    // testUpscalingHdiv(sequence);
    FiniteElementSpace * ufespace = DRSequence_FE->GetFeSpace(uform);
    FiniteElementSpace * pfespace = DRSequence_FE->GetFeSpace(pform);

    LinearForm b(ufespace);
    b.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fbdr));
    b.Assemble();
    b *= -1.;

    LinearForm q(pfespace);
    q.AddDomainIntegrator(new DomainLFIntegrator(source_coeff) );
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
    irs[Geometry::TETRAHEDRON] =
        &(IntRules.Get(Geometry::TETRAHEDRON, quadrule_order));
    irs[Geometry::CUBE] = &(IntRules.Get(Geometry::CUBE, quadrule_order));

    double tmp = ComputeLpNorm(2., efield_r_coeff, *pmesh, irs);
    solutionNorm_2(0) = tmp*tmp;
    tmp = ComputeLpNorm(2., source_coeff, *pmesh, irs);
    solutionNorm_2(1) = tmp*tmp;
    tmp = ComputeLpNorm(2., potential_coeff, *pmesh, irs);
    solutionNorm_2(2) = tmp*tmp;

    MPI_Reduce(solutionNorm_2.GetData(), solutionNorm.GetData(),
               solutionNorm.Size(), MPI_DOUBLE,MPI_SUM, 0, comm);
    if (myid == 0)
        std::transform(solutionNorm.GetData(),
                       solutionNorm.GetData()+solutionNorm.Size(),
                       solutionNorm.GetData(), (double(*)(double)) sqrt);

    Array<int> iter(nLevels);
    iter = 0;
    Array<int> ndofs(nLevels);
    ndofs = 0;
    Array<int> nnzs(nLevels);
    nnzs = 0;

    double tdiff;

    Array<SparseMatrix *> allPu(nLevels-1);
    Array<SparseMatrix *> allPp(nLevels-1);
    Array<SparseMatrix *> allD(nLevels);

    for (int i=0; i<nLevels-1; ++i)
    {
        allPu[i] = sequence[i]->GetP(uform);
        allPp[i] = sequence[i]->GetP(pform);
    }

    for (int i = 0; i < nLevels; ++i)
        allD[i] = sequence[i]->GetDerivativeOperator(uform);

    std::vector<unique_ptr<mfem::SparseMatrix> > Ml(nLevels);
    std::vector<unique_ptr<mfem::SparseMatrix> > Wl(nLevels);
    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        Ml[k] = sequence[k]->ComputeMassOperator(uform);
        Wl[k] = sequence[k]->ComputeMassOperator(pform);
        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in "
                      << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;
    }

    std::vector<Array<int>> blockOffsets(nLevels);
    for (int k(0); k < nLevels; ++k)
    {
        blockOffsets[k].SetSize(3);
        int * p = blockOffsets[k].GetData();
        p[0] = 0;
        p[1] = sequence[k]->GetNumberOfDofs(uform);
        p[2] = p[1] + sequence[k]->GetNumberOfDofs(pform);
    }

    std::vector<unique_ptr<BlockVector>> rhs(nLevels);
    rhs[0] = make_unique<BlockVector>(blockOffsets[0]);
    rhs[0]->GetBlock(0) = b;
    rhs[0]->GetBlock(1) = q;

    for (int i=0; i < nLevels-1; ++i)
    {
        rhs[i+1] = make_unique<BlockVector>(blockOffsets[i+1]);
        allPu[i]->MultTranspose(rhs[i]->GetBlock(0), rhs[i+1]->GetBlock(0) );
        allPp[i]->MultTranspose(rhs[i]->GetBlock(1), rhs[i+1]->GetBlock(1) );
    }

    std::vector<unique_ptr<BlockVector>> sol(nLevels);
    std::vector<unique_ptr<BlockVector>> help(nLevels);

    for (int k(0); k < nLevels; ++k)
    {
        sol[k] = make_unique<BlockVector>(blockOffsets[k]);
        help[k] = make_unique<BlockVector>(blockOffsets[k]);
        *(help[k]) = 0.;
    }

    for (int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();

        SparseMatrix * M = Ml[k].get();
        SparseMatrix * W = Wl[k].get();
        SparseMatrix * D = allD[k];
        auto B = ToUnique(Mult(*W, *D));
        *B *= -1.0;
        auto Bt = ToUnique(Transpose(*B));

        {
            std::ostringstream fname;
            fname << "B" << k << ".mtx";
            std::ofstream fid(fname.str().c_str());
            B->PrintMatlab(fid);
        }

        const SharingMap & l2_dofTrueDof(
            sequence[k]->GetDofHandler(pform)->GetDofTrueDof());
        const SharingMap & hdiv_dofTrueDof(
            sequence[k]->GetDofHandler(uform)->GetDofTrueDof());

        Array<int> trueBlockOffsets(3);
        trueBlockOffsets[0] = 0;
        trueBlockOffsets[1] = hdiv_dofTrueDof.GetTrueLocalSize();
        trueBlockOffsets[2] =
            trueBlockOffsets[1] + l2_dofTrueDof.GetTrueLocalSize();
        ndofs[k] = trueBlockOffsets.Last();

        BlockVector prhs(trueBlockOffsets);
        hdiv_dofTrueDof.Assemble(rhs[k]->GetBlock(0), prhs.GetBlock(0));
        l2_dofTrueDof.Assemble(rhs[k]->GetBlock(1), prhs.GetBlock(1));

        auto pM = Assemble(hdiv_dofTrueDof, *M, hdiv_dofTrueDof);
        auto pB = Assemble(l2_dofTrueDof, *B, hdiv_dofTrueDof);
        auto pBt = Assemble(hdiv_dofTrueDof, *Bt, l2_dofTrueDof);

        // could multiply B by 2, but don't think that's actually the
        // appropriate measure
        nnzs[k] = pM->NNZ() + pB->NNZ();

        BlockOperator op(trueBlockOffsets);
        op.owns_blocks = 0;
        op.SetBlock(0,0, pM.get());
        op.SetBlock(0,1, pBt.get());
        op.SetBlock(1,0, pB.get());

        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in "
                      << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;

        auto mat = make_shared<BlockMatrix>(blockOffsets[k]);
        mat->owns_blocks = 0;
        mat->SetBlock(0, 0, M);
        mat->SetBlock(0, 1, Bt.get());
        mat->SetBlock(1, 0, B.get());

        Array<int> essbdr(pmesh->bdr_attributes.Max());
        essbdr = 0;

        int actsize = sequence.size() - k;
        std::vector<shared_ptr<DeRhamSequence> > actseq(actsize);
        for (int jj=0; jj<actsize; ++jj)
            actseq[jj] = sequence[jj+k];

        chrono.Clear();
        chrono.Start();

        MLDivFree prec(mat, actseq, essbdr);
        BlockVector psol(trueBlockOffsets);
        psol = 0.;

        elag_assert(prhs.CheckFinite() == 0);
        prec.Mult(prhs, psol);

        chrono.Stop();
        tdiff = chrono.RealTime();

        if (myid == 0)
            std::cout << "Timing PRECONDITIONER_LEVEL " << k
                      << ": Preconditioner Computed " << tdiff << "s.\n";
        timings(k,PRECONDITIONER) = tdiff;

        FGMRESSolver fgmres(comm);
        fgmres.iterative_mode = true;
        fgmres.SetPrintLevel(print_iter);
        fgmres.SetMaxIter(max_num_iter);
        fgmres.SetRelTol(rtol);
        fgmres.SetAbsTol(atol);
        fgmres.SetOperator(op );
        fgmres.SetPreconditioner( prec );
        chrono.Clear();
        chrono.Start();
        fgmres.Mult(prhs, psol );
        chrono.Stop();
        tdiff = chrono.RealTime();
        if (myid == 0)
            std::cout << "Timing MINRES_LEVEL " << k << ": Solver done in "
                      << tdiff << "s.\n";
        timings(k,SOLVER) = tdiff;

        if (myid == 0)
        {
            if (fgmres.GetConverged())
                std::cout << "FGMRES converged in " << fgmres.GetNumIterations()
                          << " with a final residual norm "
                          << fgmres.GetFinalNorm() << "\n";
            else
                std::cout << "FGMRES did not converge in "
                          << fgmres.GetNumIterations()
                          << ". Final residual norm is "
                          << fgmres.GetFinalNorm() << "\n";
        }
        iter[k] = fgmres.GetNumIterations();

        hdiv_dofTrueDof.Distribute(psol.GetBlock(0), sol[k]->GetBlock(0));
        l2_dofTrueDof.Distribute(psol.GetBlock(1), sol[k]->GetBlock(1));

        // error norms
        {
            *(help[k]) = *(sol[k]);
            for (int j = k; j > 0; --j)
            {
                allPu[j-1]->Mult( help[j]->GetBlock(0), help[j-1]->GetBlock(0) );
                allPp[j-1]->Mult( help[j]->GetBlock(1), help[j-1]->GetBlock(1) );
            }

            Vector dsol( allD[0]->Size() );
            allD[0]->Mult(help[0]->GetBlock(0), dsol );

            GridFunction gf;
            gf.MakeRef(ufespace,help[0]->GetBlock(0), 0);
            double err = gf.ComputeL2Error(efield_coeff);
            analytical_errors_L2_2(k,0) = err*err;
            gf.MakeRef(pfespace,dsol, 0);
            err = gf.ComputeL2Error(source_coeff);
            analytical_errors_L2_2(k,1) = err*err;
            gf.MakeRef(pfespace,help[0]->GetBlock(1), 0);
            err = gf.ComputeL2Error(potential_coeff, irs);
            analytical_errors_L2_2(k,2) = err*err;

            for (int j(0); j < k; ++j)
            {
                if (help[j]->Size() != sol[j]->Size() ||
                    sol[j]->GetBlock(0).Size() != allD[j]->Width() )
                    mfem_error("size don't match \n");

                const int usize  = sol[j]->GetBlock(0).Size();
                const int psize  = sol[j]->GetBlock(1).Size();
                const int dsize = allD[j]->Size();
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

        // visualize solution
        if (do_visualize)
        {
            MultiVector u(sol[k]->GetBlock(0).GetData(), 1,
                          sol[k]->GetBlock(0).Size());
            sequence[k]->show(uform, u);
            MPI_Barrier(comm);
            MultiVector p(sol[k]->GetBlock(1).GetData(), 1,
                          sol[k]->GetBlock(1).Size());
            sequence[k]->show(pform, p);
            MPI_Barrier(comm);
            MultiVector div_u(1, sol[k]->GetBlock(1).Size());
            allD[k]->Mult(sol[k]->GetBlock(0), div_u);
            sequence[k]->show(pform, div_u);
        }
    }

    DenseMatrix u_errors_L2(nLevels, nLevels);
    u_errors_L2 = 0.;
    DenseMatrix p_errors_L2(nLevels, nLevels);
    p_errors_L2 = 0.;
    DenseMatrix errors_div(nLevels, nLevels);
    errors_div = 0.;
    DenseMatrix analytical_errors(nLevels, 3);
    analytical_errors = 0.;
    // DenseMatrix timings(nLevels-1, NSTAGES);

    MPI_Reduce(u_errors_L2_2.Data(), u_errors_L2.Data(),
               u_errors_L2.Height()*u_errors_L2.Width(),
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(p_errors_L2_2.Data(), p_errors_L2.Data(),
               p_errors_L2.Height()*p_errors_L2.Width(),
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(errors_div_2.Data(), errors_div.Data(),
               errors_div.Height()*errors_div.Width(),
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(analytical_errors_L2_2.Data(), analytical_errors.Data(),
               analytical_errors_L2_2.Height()*analytical_errors_L2_2.Width(),
               MPI_DOUBLE, MPI_SUM, 0, comm);

    if (myid == 0)
    {
        std::transform(
            u_errors_L2.Data(),
            u_errors_L2.Data()+u_errors_L2.Height()*u_errors_L2.Width(),
            u_errors_L2.Data(), (double(*)(double)) sqrt);
        std::transform(
            p_errors_L2.Data(),
            p_errors_L2.Data()+p_errors_L2.Height()*p_errors_L2.Width(),
            p_errors_L2.Data(), (double(*)(double)) sqrt);
        std::transform(
            errors_div.Data(),
            errors_div.Data()+errors_div.Height()*errors_div.Width(),
            errors_div.Data(), (double(*)(double)) sqrt);
        std::transform(
            analytical_errors.Data(),
            analytical_errors.Data() +
            analytical_errors.Height()*analytical_errors.Width(),
            analytical_errors.Data(), (double(*)(double)) sqrt);
    }

    if (myid == 0)
    {
        std::cout << "\n{\n";
        int w = 14;
        std::cout << "%tlevel" << std::setw(w) << "Topology" << std::setw(w)
                  << "Sequence\n";
        for (int i(0); i < nLevels; ++i)
            std::cout << i << std::setw(w) << timings(i,TOPOLOGY) << std::setw(w)
                      << timings(i,SEQUENCE) << "\n";
        std::cout << "}\n";
    }

    /*
      std::cout << "%ulevel" << std::setw(w) << "size" << std::setw(w) << "nnz" << std::setw(w) << "nit"
      << std::setw(w) << "Assembly" << std::setw(w) << "Prec "
      << std::setw(w) << "Solver\n";
      for (int i(0); i < nLevels; ++i)
      std::cout<< i << std::setw(w) << ndofs[i] << std::setw(w) << nnzs[i] << std::setw(w) << iter[i]
      << std::setw(w) << timings(i,ASSEMBLY) << std::setw(w) << timings(i,PRECONDITIONER)
      << std::setw(w) << timings(i,SOLVER) << "\n";
      std::cout << "}\n";
    */
    OutputUpscalingTimings(ndofs, nnzs, iter, timings,
                           stage_names);

    if (myid == 0)
    {
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

    if (myid == 0)
    {
        std::cout << "u l2-like errors: ";
        int h = u_errors_L2.Height();
        for (int i=0; i<h-1; ++i)
            std::cout << std::scientific << std::setprecision(4)
                      << u_errors_L2.Elem(h-1-i,0) << " ";
        std::cout << std::endl;
        h = p_errors_L2.Height();
        if (h > 0)
            std::cout << "p l2-like errors: ";
        for (int i=0; i<h-1; ++i)
            std::cout << std::scientific << std::setprecision(4)
                      << p_errors_L2.Elem(h-1-i,0) << " ";
        if (h > 0)
            std::cout << std::endl;
        h = errors_div.Height();
        std::cout << "u energy-like errors: ";
        for (int i=0; i<h-1; ++i)
            std::cout << std::scientific << std::setprecision(4)
                      << errors_div.Elem(h-1-i,0) << " ";
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
