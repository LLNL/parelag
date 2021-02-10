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

//                       Elag Upscaling - Parallel Version
//
//              NUMERICAL UPSCALING OF QUASI STATIC MAXWELL'S EQUATION
//
//                (muInv * curl E, curl W) + (sigma * E, W) = (RHS, W)
//
// Compile with: make upsme.cpp
//
// Sample runs:
//
// mpirun -np 4 ./upsme.exe --feorder 0 --nref_serial 3 --nref_parallel 2
//
// ./upsme.exe --mesh ../meshes/cube456.mesh3d --nref_serial 1 --feorder --no-visualize
//
// mpirun -np 4 ./upsme.exe --mesh ../meshes/cube456.mesh3d
//    --nref_serial 1 --nref_parallel 1 --feorder 0 --upscalingorder 0
//
// mpirun -np 4 ./examples/UpscalingMaxwell.exe --use-petsc --unassembled
//    --ser_ref_levels 1 --petsc-opts ../examples/.petsc_rc_maxwell_bddc
// NOTE:  Runs stated require the directory meshes located in an upper directory
//
// Description: This example code solves a 3D electromagnetic
//              diffusion problem corresponding to the bilinear form
//              of the quasi-static Maxwell equation (muInv * curl E,
//              curl W) + (sigma * E, W) = (RHS, W) where E is the
//              electric field, W is a test function, mu is the
//              magnetic permeability (constant), sigma is the
//              electrical conductivity (function) and RHS is the
//              r.h.s. We close the system with Dirichlet boundary
//              conditions E x n = <given tangential field>.
//
//              Here, we use a given exact solution E and compute the
//              corresponding RHS. We discretize with Nedelec finite
//              elements.
//
//              The example demonstrates the use of H(curl) finite
//              element spaces with the curl-curl and the (vector
//              finite element) mass bilinear form, as well as the
//              computation of discretization and elements of energy
//              error when the exact solution is known.
//
//              Visualization of the electric field and the electrical
//              conductivity can be done using glvis (when glvis is
//              running) or with Visit (by saving solution to file).
//
//
// Luz Angelica Caudillo Mata, Umberto Villa, Panayot Vassilevski
// lacmajedrez@gmail.com
// LLNL - Computation - CASC Intern
// Creation:      July 21, 2015
// Modification:  August 13, 2015

#include <fstream>
#include <sstream>
#include <mpi.h>
#include "elag.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

enum {EMPTY = 0, AMS};
enum {ASSEMBLY = 0, PRECONDITIONER_EMPTY, PRECONDITIONER_AMS, SOLVER_EMPTY, SOLVER_AMS};

const int NSOLVERS = 2;
const char * solver_names[] = {"EMPTY","AMS"};

const int NSTAGES = 5;
const char * stage_names[] = {"ASSEMBLY", "prec EMPTY", "prec AMS",
                              "SOLVER_EMPTY", "SOLVER_AMS"};

// Expression for the magnetic permeability
const   double MU    = 4.0*M_PI*1e-2;

const   double kappa = M_PI;

void  E_exact(const Vector &x, Vector &E)
{ // Exact expression for electric field
    E(0) = sin(kappa * x(1));
    E(1) = sin(kappa * x(2));
    E(2) = sin(kappa * x(0));
}

void  CurlE_exact(const Vector &x, Vector &CE)
{ // Exact expression for curl of E_exact
    CE(0) = -kappa * cos(kappa * x(2));
    CE(1) = -kappa * cos(kappa * x(0));
    CE(2) = -kappa * cos(kappa * x(1));
}

double fh(double v)
{   // Auxiliar function used in function sig_spc (which defines expression for
    // the electrical conductivity)
    // This function can be continuous or discontinuous depending on the values
    // of the parameter a:
    // e.g. a = 1.0,  fh is continuous,
    //      a = 1e-9, fh is discontinuous.  For this case to work optimally, the
    //                computational domain must be in [0,1]x[0,1]x[0,1]
    //
    // NOTE:  The root of this function happens at x = -(b/a) + x0

    double fhv,a,b,x0,p;

    // Configuration for continuous sigma
    // // Initialize variables
    // a = 1.0;
    // b = M_PI/2.0 + 0.01;
    // x0 = 0.5;

    // // Define function
    // fhv = exp(v) / (atan(a*(v-x0)) + b);

    // Configuration for discontinuous sigma
    // Cherry picked values to give a function varying in the range [1e-3,1e3]
    p   = 9.0;
    a   = 1e-9;
    b   = M_PI/8.0;
    x0  = 0.4;

    // Define function
    fhv =  exp( p * sin( exp(v) / (atan(a*(v-x0) + b)) ) ) ;

    // Return value
    return fhv;
}

double sig_spc(const Vector &x)
{  //Expression for the electrical conductivity

   // Expression for continuous conductivity
   //return ( 0.5 + exp( sin( fh(x(0)) + fh(x(1)) + fh(x(2)) ) ) );

   // Expression for discontinuous conductivity
    return ( fh(x(0)) + fh(x(1)) + fh(x(2)) );
}

void RHS_exact(const Vector &x, Vector &rhs)
{  // Expression for the R.H.S = curl(1/mu * curl E) + sigma E

    Vector y = x;
    double sigma = sig_spc(y);

    rhs(0) = (sin(kappa * x(1)) * (MU * sigma + kappa * kappa)) / MU;
    rhs(1) = (sin(kappa * x(2)) * (MU * sigma + kappa * kappa)) / MU;
    rhs(2) = (sin(kappa * x(0)) * (MU * sigma + kappa * kappa)) / MU;
}

int main (int argc, char *argv[])
{
    // 1. Initialize MPI, read command line data
    mpi_session sess(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    StopWatch chrono;

    // 2. Read command-line options and set defaulted values Reading
    //    setup from command line makes easier to run different
    //    simulations in parallel in one of the LC servers at LLNL.
    const char* meshfile_c = "../../mfem/data/fichera.mesh";
    int ser_ref_levels = 0;
    int par_ref_levels = 2;
    int feorder = 0;
    int upscalingOrder = 0;
    bool do_visualize = true;
    bool reportTiming = true;
    bool petsc = false;
    bool unassembled = false;
    const char* petscopts_c = "";
    OptionsParser args(argc, argv);
    args.AddOption(&meshfile_c, "-m", "--mesh",
                   "MFEM mesh file to load.");
    args.AddOption(&ser_ref_levels, "-sr", "--nref_serial",
                   "Number of times to refine serial mesh.");
    args.AddOption(&par_ref_levels, "-pr", "--nref_parallel",
                   "Number of times to refine parallel mesh.");
    args.AddOption(&feorder, "-feo", "--feorder",
                   "Polynomial order of fine finite element space.");
    args.AddOption(&upscalingOrder, "-uo", "--upscalingorder",
                   "Target polynomial order of coarse space.");
    args.AddOption(&do_visualize, "-v", "--do-visualize", "-nv", "--no-visualize",
                   "Do interactive GLVis visualization.");
    args.AddOption(&reportTiming, "--report-timing", "--report-timing",
                   "--no-report-timing", "--no-report-timing",
                   "Whether to print out timings as we proceed.");
    args.AddOption(&petsc, "--use-petsc", "--use-petsc",
                   "--no-petsc", "--no-petsc",
                   "Enable Petsc solvers etc.");
    args.AddOption(&petscopts_c, "--petsc-opts", "--petsc-opts",
                   "Options file for Petsc.");
    args.AddOption(&unassembled, "--unassembled", "--unassembled",
                   "--assembled", "--assembled",
                   "Whether to use Petsc assembled or unassembled operator.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(std::cout);
        MPI_Finalize();
        return 1;
    }
    PARELAG_ASSERT(args.Good());
    std::string meshfile(meshfile_c);
    std::string petscopts(petscopts_c);

#ifdef ParELAG_ENABLE_PETSC
    PetscErrorCode ierr;
    Operator::Type petsc_tid;
    if (unassembled)
        petsc_tid = Operator::PETSC_MATIS;
    else
        petsc_tid = Operator::PETSC_MATAIJ;
    if (!petscopts.size())
    {
        ierr = PetscInitialize(NULL, NULL, NULL, NULL);
    }
    else
    {
        ierr = PetscInitialize(NULL, NULL, petscopts.c_str(), NULL);
    }
    PARELAG_ASSERT(!ierr);
#else
    if (petsc)
    {
        if (myid == 0)
        {
            std::cerr << "ParElag was not configured to be used with PETSc" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
#endif

    const int form = 1; // 0 = GRADIENT, 1 = CURL, 2 = DIVERGENCE

    // Defaulted linear solver options
    constexpr int printIts = 0;// 0 = do not print, 1 = print
    constexpr int maxIts = 500;// Num max iterations
    constexpr double rTol = 1e-6;// Relative tolerance
    constexpr double aTol = 1e-12;// Absolute tolerance

    // Computed variables from input
    const int nLevels = par_ref_levels+1; // Number of levels in method

    // 3. Read mesh info from the given mesh file and form vector of
    //    essential boundary conditions (Dirichlet) using information.
    //    NOTE: This script assumes that given mesh is the coarse
    //          mesh. Thus we will refine it inside to generate the
    //          fine mesh.

    shared_ptr<ParMesh> pmesh;
    Array<int> ess_attr;
    {
        // 3.1. Read the (serial, coarse) mesh from the given mesh file
        // and check dimensions
        std::ifstream imesh(meshfile.c_str());
        if (!imesh)
        {
            std::cerr << "\nCan not open mesh file: " << meshfile << "\n\n";
            return EXIT_FAILURE;
        }
        auto mesh = make_unique<Mesh>(imesh, 1, 1);
        imesh.close();

        // Make sure it's a 3D mesh :)
        const int nDimensions = mesh->Dimension();
        if (nDimensions != 3)
        {
            std::cerr << "\nThis example requires a 3D mesh\n" << std::endl;
            return EXIT_FAILURE;
        }

        // 3.2. Setup vector of esssential boundary conditions using
        // info of coarse mesh
        const int true_nbdr = mesh->bdr_attributes.Size();
        ess_attr.SetSize(true_nbdr);
        ess_attr = 1;

        // 3.3 Refine given serial mesh to increase resolution.
        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        // Print information of the fine mesh
        if (myid == 0)
        {
            std::cout << "Number of elements in serial mesh: "
                      << mesh->GetNE() << std::endl;
            std::cout << "Number of edges in serial mesh   : "
                      << mesh->GetNEdges() << std::endl;
            std::cout << "Number of faces in serial mesh   : "
                      << mesh->GetNFaces() << std::endl;
        }

        // 3.4 Create files to plot conductivity by defining an auxiliar L2 space
        //     as the electrical conductivity is a piece-wise constant function
        L2_FECollection fec_l2(0, nDimensions);
        FiniteElementSpace fespace_l2(mesh.get(), &fec_l2);

        // Get values of sigma grid function at the cell centers
        FunctionCoefficient sigmaA(sig_spc);
        GridFunction sigAux(&fespace_l2);
        sigAux.ProjectCoefficient(sigmaA);

        std::ofstream mesh_ofs("refined.mesh");
        mesh_ofs.precision(8);
        mesh->Print(mesh_ofs);

        // Save conductivity
        std::ofstream sig_ofs("sigma.gf");
        sig_ofs.precision(8);
        sigAux.Save(sig_ofs);

        if (!do_visualize)
        {
            // That is, user is asking for visit files rather than
            // plotting in glvis socket

            VisItDataCollection visit_sig("sigma",mesh.get());

            visit_sig.RegisterField("sigma", &sigAux);
            visit_sig.SetCycle(1);
            visit_sig.Save();
        }

        // 3.5 Initialize a parallel mesh with refined mesh
        //     NOTE:  This refined mesh will be our fine mesh in the 2-level method
        pmesh = make_shared<ParMesh>(comm, *mesh);
    }

    constexpr int nDimensions = 3;

    // 3.6 Setup array with number of elements for each level in the method.
    //      Refine distributed mesh
    Array<int> level_nElements(nLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    level_nElements[0] = pmesh->GetNE();

    // 3.7 Make sure elements has proper orientation after refinement. We do it
    //      after refinement as in 3D this process is quite expensive
    pmesh->ReorientTetMesh();

    // Stop timer
    chrono.Clear();

    // 4. Define agglomeration (i.e. partition of the mesh in coarser entities)

    // Start timer
    chrono.Start();

    //Setup array of nLevels of aglomerates
    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    // Set finest mesh at the first (i.e. 0) level
    topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);

    // Initialize partitioner and form agglomerates at each level
    MFEMRefinedMeshPartitioner partitioner(nDimensions);
    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
    {
        Array<int> partitioning(
            topology[ilevel]->GetNumberLocalEntities(at_elem));
        partitioner.Partition(topology[ilevel]->GetNumberLocalEntities(at_elem),
                              level_nElements[ilevel+1], partitioning);
        topology[ilevel+1] =
            topology[ilevel]->CoarsenLocalPartitioning(partitioning, 0, 0);
    }

    // Stop timer
    chrono.Stop();

    // Print timming
    if (myid == 0 && reportTiming)
        std::cout << "\nTiming ELEM_AGG: Mesh Agglomeration done in "
                  << chrono.RealTime() << " seconds.\n";

    // 5. Form De-Rham Sequence of FE spaces at the finest level and setup mass
    //    matrices using information of PDE coefficients
    //    NOTE:  It uses MFEM to generate the spaces


    // Initialize De-Rham sequence of subspaces
    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());
    sequence[0] = make_shared<DeRhamSequence3D_FE>(
        topology[0], pmesh.get(), feorder);

    DeRhamSequenceFE* DRSequence_FE = sequence[0]->FemSequence();

    // Starting space from which the De-Rham sequence will be build.
    // jFormStart = 0: starts from H1      - only in 3D
    // jFormStart = 1: starts from H(curl) - only in 3D
    // jFormStart = 2: starts from H(div)  - only in 3D
    // jFormStart = 3: starts from L2      - only in 3D
    sequence[0]->SetjformStart(0);

    // Setup the coefficients of bilinear form
    FunctionCoefficient sigma(sig_spc);
    ConstantCoefficient muInv(1.0/MU);

    // Form edge - mass matrix i.e. (sigma * E, W)
    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, form, make_unique<VectorFEMassIntegrator>(sigma), false);

    // Form face - mass matrix i.e. something like (muInv * B, F)
    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, form+1, make_unique<VectorFEMassIntegrator>(muInv), true);

    // 6. Form basis functions and compute P
    //      Setup for Vassilevski's method after all this process we get the P!

    // First we setup targets
    Array<Coefficient *> L2coeff;
    Array<VectorCoefficient *> Hdivcoeff;
    Array<VectorCoefficient *> Hcurlcoeff; // only needed for form < 2
    Array<Coefficient *> H1coeff; // only needed for form < 1
    fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);
    fillVectorCoefficientArray(nDimensions, upscalingOrder, Hdivcoeff);
    if (form < 2)
        fillVectorCoefficientArray(nDimensions, upscalingOrder, Hcurlcoeff);
    if (form < 1)
        fillCoefficientArray(nDimensions, upscalingOrder+1, H1coeff);

    // Setup targets, ie, global functions that live in the coarse spaces
    std::vector<unique_ptr<MultiVector>>
        targets(sequence[0]->GetNumberOfForms());

    int jform = 0;
    if (form < 1)
        targets[jform] = DRSequence_FE->InterpolateScalarTargets(jform, H1coeff);

    ++jform;
    if (form < 2)
    {
        if (nDimensions == 3)
        {
            targets[jform] =
                DRSequence_FE->InterpolateVectorTargets(jform, Hcurlcoeff);
            ++jform;
        }
    }
    else
    {
        ++jform;
    }
    targets[jform] = DRSequence_FE->InterpolateVectorTargets(jform, Hdivcoeff);
    ++jform;
    targets[jform] = DRSequence_FE->InterpolateScalarTargets(jform, L2coeff);
    ++jform;

    freeCoeffArray(L2coeff);
    freeCoeffArray(Hdivcoeff);
    freeCoeffArray(Hcurlcoeff);
    freeCoeffArray(H1coeff);

    Array<MultiVector *> targets_in(targets.size());
    for (int ii = 0; ii < targets_in.Size(); ++ii)
        targets_in[ii] = targets[ii].get();
    sequence[0]->SetTargets(targets_in);

    // Get the FE spaces from the DeRham sequences
    //Nedelec space
    FiniteElementSpace *e_fespace = DRSequence_FE->GetFeSpace(form);
    // Raviart-Thomas space
    FiniteElementSpace *ce_fespace = DRSequence_FE->GetFeSpace(form+1);

    chrono.Clear();

    // Setup SVD tolerance
    constexpr double tolSVD = 1e-9;

    // Corsening - builds the global P and local A_H = P_k^T
    // diag(A_h_agglomerate) P_k for each level
    chrono.Start();
    for(int i(0); i < nLevels-1; ++i)
    {
        sequence[i]->SetSVDTol(tolSVD);
        StopWatch chronoInterior;
        chronoInterior.Clear();
        chronoInterior.Start();
        sequence[i+1] = sequence[i]->Coarsen();
        chronoInterior.Stop();
        if (myid == 0 && reportTiming)
            std::cout << "\nTiming ELEM_AGG_LEVEL" << i
                      << ": Coarsening done in " << chronoInterior.RealTime()
                      << " seconds.\n";
    }
    chrono.Stop();

    if (myid == 0 && reportTiming)
        std::cout << "Timing ELEM_AGG: Coarsening done in "
                  << chrono.RealTime() << " seconds.\n";

    // 7. Form system (Ax = b) to be solved in all levels
    FiniteElementSpace * fespace = sequence[0]->FemSequence()->GetFeSpace(form);

    // 7.1 Set up the linear form b(.) which corresponds to the
    //  right-hand side of the FEM linear system, which in this case
    //  is (RHS,phi_i) where RHS is given by the function RHS_exact
    //  and phi_i are the basis functions in the finite element
    //  fespace.
    VectorFunctionCoefficient RHS(3, RHS_exact);
    auto b = make_unique<LinearForm>(fespace);
    b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(RHS));
    b->Assemble();

    // 7.2 Define the vector bdry as a finite element grid function
    //  corresponding to fespace. Initialize bdry by projecting the
    //  exact solution. Note that only values from the boundary edges
    //  will be used when eliminating the non-homogeneous boundary
    //  condition to modify the r.h.s. vector b.
    VectorFunctionCoefficient E(3, E_exact);
    auto bdry = make_unique<GridFunction>(fespace);
    bdry->ProjectCoefficient(E);

    // 7.3 Initialize error and norm matrices for testing purposes.
    //     Errors meassured:
    //     - l2 error - ||u - v||_l2 and ||u||_l2
    //     - energy   - ||u - v||_l2 + ||curl u - curl v||_l2
    //

    DenseMatrix errors_L2_2(nLevels, nLevels);
    DenseMatrix errors_curl_2(nLevels, nLevels);

    Vector analytical_errors_L2_2(nLevels);
    Vector analytical_curl_errors_2(nLevels);

    DenseMatrix timings(nLevels, NSTAGES);

    Vector      norm_L2_2(nLevels);//Save norms of e_fine
    Vector      norm_curl_2(nLevels);


    // Variables to visualize with VisIt
    std::ostringstream oss;
    oss << "upsmep-feo" << feorder << "_sr" << ser_ref_levels << "_pr"
        << par_ref_levels;
    VisItDataCollection visit_sol(oss.str().c_str(), pmesh.get());

    Array2D<int> iter(nLevels, NSOLVERS);
    Array<int> ndofs(nLevels);
    Array<int> nnz(nLevels);

    analytical_errors_L2_2 = 0.0;
    analytical_curl_errors_2 = 0.0;

    errors_L2_2 = 0.0;
    norm_L2_2 = 0.0;
    errors_curl_2 = 0.0;
    norm_curl_2 = 0.0;
    timings = 0.0;

    iter = 0;
    ndofs = 0;
    nnz = 0;

    // Auxiliar grid function to messure curl error
    VectorFunctionCoefficient CurlE(3, CurlE_exact);

    // Auxiliar variable to messure time
    double timer;

    // Arrays to hold all CURL's and P's operators at the different levels
    Array<SparseMatrix *> allP(nLevels-1);
    Array<SparseMatrix *> allCURL(nLevels);

    // here I get the P for level i
    for(int i = 0; i < nLevels - 1; ++i)
        allP[i] = sequence[i]->GetP(form);

    // here I get the curl for level i
    for(int i = 0; i < nLevels; ++i)
        allCURL[i] = sequence[i]->GetDerivativeOperator(form);

    // Assembling mass matrices for all levles
    std::vector<unique_ptr<SparseMatrix>> Msigma(nLevels);
    std::vector<unique_ptr<SparseMatrix>> MmuInv(nLevels);

    for(int k = 0; k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        //edge mass matrix
        Msigma[k] = sequence[k]->ComputeMassOperator(form);
        // face mass matrix
        MmuInv[k] = sequence[k]->ComputeMassOperator(form+1);
        chrono.Stop();
        timer = chrono.RealTime();

        // Inform user
        if (myid == 0 && reportTiming)
            std::cout << "Timing ELEM_AGG_LEVEL " << k
                      << ": Assembly done in " << timer << "s. \n";
        timings(k, ASSEMBLY) += timer;
    }

    // Assembly RHS and vector of essential boundary conditions for
    // all levels. Propagation of boundary conditions through all the
    // levels is done here as well
    std::vector<unique_ptr<Vector>> rhs(nLevels);
    std::vector<unique_ptr<Vector>> ess_data(nLevels);

    rhs[0]      = std::move(b);
    ess_data[0] = std::move(bdry);
    for(int i = 0; i < nLevels-1; ++i)
    {
        rhs[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(form));
        ess_data[i+1] = make_unique<Vector>(sequence[i+1]->GetNumberOfDofs(form));
        // coarse rhs = P^T * (fine rhs)
        sequence[i]->GetP(form)->MultTranspose(*(rhs[i]), *(rhs[i+1]));
        sequence[i]->GetPi(form)->ComputeProjector();
        sequence[i]->GetPi(form)->GetProjectorMatrix().Mult(
            *(ess_data[i]), *(ess_data[i+1]) );
    }

    // Initialize vector of solutions at all levels
    std::vector<unique_ptr<Vector>> sol(nLevels);
    std::vector<unique_ptr<Vector>> help(nLevels);

    // Prepare vector solution at all levels
    for(int k = 0; k < nLevels; ++k)
    {
        sol[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
        *(sol[k]) = 0.0;
        help[k] = make_unique<Vector>(sequence[k]->GetNumberOfDofs(form));
        *(help[k]) = 0.0;
    }

    // After all that work, finally assembly and solve system of equations at
    // all levels! :)
    for(int k = 0; k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();

        std::unique_ptr<HypreParMatrix> pA;
#ifdef ParELAG_ENABLE_PETSC
        std::unique_ptr<PetscParMatrix> ppA;
#endif
        Array<int> emarker;
        const SharingMap & form_dofTrueDof(
            sequence[k]->GetDofHandler(form)->GetDofTrueDof() );
        {
            // Form matrix of bilinear form (sigma E, W) + (muInv CURL E, CURL W)
            // i.e. A = Msig + C^T*Mf*C
            SparseMatrix * Me = Msigma[k].get();// edge mass matrix
            SparseMatrix * Mf = MmuInv[k].get();// face mass matrix
            SparseMatrix * C = allCURL[k];// curl matrix
            auto A = ToUnique(Add(*Me, *ExampleRAP(*C,*Mf,*C)));

            // The boundary conditions are implemented by marking all
            // the boundary attributes from the mesh as essential
            // (Dirichlet). After assembly and finalizing we extract
            // the corresponding sparse matrix A.
            const int nlocdofs = A->Height();

            emarker.SetSize(nlocdofs);
            emarker = 0;

            sequence[k]->GetDofHandler(form)->MarkDofsOnSelectedBndr(
                ess_attr, emarker);

            for(int mm = 0; mm < nlocdofs; ++mm)
                if(emarker[mm])
                    A->EliminateRowCol(
                        mm, ess_data[k]->operator ()(mm), *(rhs[k]) );

            if (!petsc)
            {
                pA = Assemble(form_dofTrueDof, *A, form_dofTrueDof);
            }
#ifdef ParELAG_ENABLE_PETSC
            else
            {
                ppA = AssemblePetsc(form_dofTrueDof, *A, form_dofTrueDof,
                                    petsc_tid);
            }
#endif
        }

        // Prepare elements for AMS solver in parallel
        // Serial system into parallel system
        Vector prhs( form_dofTrueDof.GetTrueLocalSize() );
        form_dofTrueDof.Assemble(*(rhs[k]), prhs);

        // Error check - does size of matrix A and rhs match?
        // Get info about num of degrees of freedom and num of non zeros
#ifdef ParELAG_ENABLE_PETSC
        if (petsc)
        {
            elag_assert(prhs.Size() == ppA->Height() );
            ndofs[k] = ppA->M();
            nnz[k]   = ppA->NNZ();
        }
        else
#endif
        {
            elag_assert(prhs.Size() == pA->Height() );
            ndofs[k] = pA->GetGlobalNumRows();
            nnz[k]   = pA->NNZ();
        }
        chrono.Stop();
        timer = chrono.RealTime();

        // Inform the user we finished with the assebly
        if (myid == 0 && reportTiming)
            std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in "
                      << timer << "s. \n";
        timings(k, ASSEMBLY) += timer;


        {
            timings(k, PRECONDITIONER_EMPTY) = 0.;
            timings(k,SOLVER_EMPTY) = 0.;
            iter(k,EMPTY) = 0;
        }

#ifdef ParELAG_ENABLE_PETSC
        if (petsc)
        {
            iter(k,AMS) = UpscalingPetscSolver(
                form, ppA.get(), prhs,sequence[k].get(),k,
                &emarker,NULL,
                PRECONDITIONER_AMS, SOLVER_AMS,
                printIts, maxIts, rTol, aTol,
                timings, form_dofTrueDof, *(sol[k]), reportTiming);
        }
        else
#endif
        {
            iter(k,AMS) = UpscalingHypreSolver(
                form, pA.get(), prhs,sequence[k].get(),k,
                PRECONDITIONER_AMS, SOLVER_AMS,
                printIts, maxIts, rTol, aTol,
                timings, form_dofTrueDof, *(sol[k]), reportTiming);
        }
        //ERROR NORMS
        {
            *(help[k]) = *(sol[k]);
            for(int j = k; j > 0; --j)
                allP[j-1]->Mult( *(help[j]), *(help[j-1]) );

            norm_L2_2(k) = Msigma[k]->InnerProduct(*(sol[k]), *(sol[k]) );
            Vector dsol( allCURL[k]->Height() );

            allCURL[k]->Mult(*(sol[k]), dsol );
            norm_curl_2(k) = MmuInv[k]->InnerProduct(dsol, dsol );

            Vector dsol0( allCURL[0]->Height() );
            allCURL[0]->Mult( *(help[0]), dsol0 );

            // Compute errors with exact solution
            GridFunction gf;
            // Remember: help[0] will temporarily store the upscaled
            // solution projected at the finest level
            gf.MakeRef(e_fespace,*(help[0]), 0);

            double err = gf.ComputeL2Error(E);

            analytical_errors_L2_2(k) = err*err;
            gf.MakeRef(ce_fespace,dsol0, 0);

            err = gf.ComputeL2Error(CurlE);
            analytical_curl_errors_2(k) = err*err;

            for(int j = 0; j < k; ++j)
            {
                if(help[j]->Size() != sol[j]->Size() || sol[j]->Size() != allCURL[j]->Width() )
                    mfem_error("Size don't match \n");

                const int size  = sol[j]->Size();
                const int dsize = allCURL[j]->Height();

                Vector e_H(help[j]->GetData(), size);
                Vector e_h(sol[j]->GetData(), size);
                Vector e_diff(size), de_diff(dsize);
                e_diff  = 0.0;
                de_diff = 0.0;

                subtract(e_H, e_h, e_diff);         //  e_diff = e_h - e_H
                allCURL[j]->Mult(e_diff, de_diff);  // de_diff = CURL * e_diff

                // NOTE: I think this norms should be computed with a
                // unitary face and edge matrix (i.e. sig = 1, mu = 1)
                errors_L2_2(k,j) =  Msigma[j]->InnerProduct(e_diff, e_diff);
                // e_diff^T * M_sig * e_diff
                errors_curl_2(k,j)=  MmuInv[j]->InnerProduct(de_diff,de_diff);
                //de_diff^T * M_muInv * de_diff
            }
        }

        //VISUALIZE SOLUTION FOR EACH LEVEL
        if (true)
        {
            if (do_visualize)
            {
                // GlVis visualization - send solution by socket to a
                // glvis active server
                MultiVector tmp(sol[k]->GetData(), 1, sol[k]->Size() );
                sequence[k]->show(form, tmp);
            }

            // VisIt visualization - write files for solution
            if (!do_visualize)
            {
                GridFunction aux;
                // Remember: help[0] will temporarily store the
                // upscaled solution projected at the finest level
                aux.MakeRef(e_fespace,*(help[0]), 0);
                visit_sol.RegisterField("solution", &aux);
                visit_sol.SetCycle(k);
                visit_sol.Save();
            }
        }
    }

    // 8. Compute errors of the finest mesh and different upscaled
    //    solutions at the different levels

    DenseMatrix errors_L2(nLevels, nLevels);
    errors_L2 = 0.;
    Vector norm_L2(nLevels);
    norm_L2 = 0.;
    DenseMatrix errors_curl(nLevels, nLevels);
    errors_curl = 0.;
    Vector norm_curl(nLevels);
    norm_curl = 0.;

    Vector analytical_errors_L2(nLevels);
    analytical_errors_L2 = 0.0;

    Vector analytical_errors_curl(nLevels);
    analytical_errors_curl = 0.0;



    // Reduce (or gather) error and norm information from all
    // processors, so we get a global vector
    MPI_Reduce(errors_L2_2.Data(), errors_L2.Data(),
               errors_L2.Height()*errors_L2.Width(),
               MPI_DOUBLE,MPI_SUM, 0, comm);
    MPI_Reduce(norm_L2_2.GetData(), norm_L2.GetData(), norm_L2.Size(),
               MPI_DOUBLE,MPI_SUM, 0, comm);
    MPI_Reduce(errors_curl_2.Data(), errors_curl.Data(),
               errors_curl.Height()*errors_curl.Width(),
               MPI_DOUBLE,MPI_SUM, 0, comm);
    MPI_Reduce(norm_curl_2.GetData(), norm_curl.GetData(), norm_curl.Size(),
               MPI_DOUBLE,MPI_SUM, 0, comm);

    //  what I have in each processor -> processor[0] (sum together),
    //  what we sent, action, num processor to be send, communicator
    //  manager
    MPI_Reduce(analytical_errors_L2_2.GetData(),
               analytical_errors_L2.GetData(), analytical_errors_L2.Size(),
               MPI_DOUBLE,MPI_SUM, 0, comm);
    MPI_Reduce(analytical_curl_errors_2.GetData(),
               analytical_errors_curl.GetData(), analytical_errors_curl.Size(),
               MPI_DOUBLE,MPI_SUM, 0, comm);

    // sqrt
    std::transform(errors_L2.Data(),
                   errors_L2.Data()+errors_L2.Height()*errors_L2.Width(),
                   errors_L2.Data(), (double(*)(double)) sqrt);
    std::transform(norm_L2.GetData(), norm_L2.GetData()+norm_L2.Size(),
                   norm_L2.GetData(), (double(*)(double)) sqrt);
    std::transform(errors_curl.Data(),
                   errors_curl.Data()+errors_curl.Height()*errors_curl.Width(),
                   errors_curl.Data(), (double(*)(double)) sqrt);
    std::transform(norm_curl.GetData(), norm_curl.GetData()+norm_curl.Size(),
                   norm_curl.GetData(), (double(*)(double)) sqrt);

    std::transform(analytical_errors_L2.GetData(),
                   analytical_errors_L2.GetData()+analytical_errors_L2.Size(),
                   analytical_errors_L2.GetData(), (double(*)(double)) sqrt);
    std::transform(analytical_errors_curl.GetData(),
                   analytical_errors_curl.GetData()+analytical_errors_curl.Size(),
                   analytical_errors_curl.GetData(), (double(*)(double)) sqrt);

    // Print table with info at the different levels as well as errors and norms info
    if (myid == 0)
    {
        if (reportTiming)
        {
            std::cout << std::string(50,'=') << '\n';
            constexpr int w = 14;

            // Print labels table
            std::cout << "%level" << std::setw(w)
                      << "size" << std::setw(w)
                      << "nnz" << std::setw(w)
                      << "nit EMPTY" << std::setw(w)
                      << "nit AMS" << std::setw(w)
                      << "Assembly" << std::setw(w)
                      << "Prec EMPTY" << std::setw(w)
                      << "Prec AMS" << std::setw(w)
                      << "Solver EMPTY" << std::setw(w)
                      << "Solver AMS\n";

            // Print table with levels info
            for(int i = 0; i < nLevels; ++i)
                std::cout << i << std::setw(w)
                          << ndofs[i] << std::setw(w)
                          << nnz[i] << std::setw(w)
                          << iter(i,EMPTY) << std::setw(w)
                          << iter(i,AMS) << std::setw(w)
                          << timings(i,ASSEMBLY) << std::setw(w)
                          << timings(i,PRECONDITIONER_EMPTY)  << std::setw(w)
                          << timings(i,PRECONDITIONER_AMS) << std::setw(w)
                          << timings(i,SOLVER_EMPTY) << std::setw(w)
                          << timings(i,SOLVER_AMS) << "\n";
            std::cout << std::string(50,'=') << '\n';
        }


        // Print errors and sol norms at different levels
        std::cout << std::string(50,'=') << '\n';
        std::cout << "% || eh - eH ||_Msig\n";
        errors_L2.PrintMatlab(std::cout);
        std::cout << "\n";

        std::cout << "% || eh ||_Msig\n";
        norm_L2.Print(std::cout, nLevels);
        std::cout << "\n";

        std::cout << "% || der ( eh - eH ) ||_MmuInv\n";
        errors_curl.PrintMatlab(std::cout);
        std::cout << "\n";

        std::cout << "% || der eh ||_MmuInv\n";
        norm_curl.Print(std::cout, nLevels);
        std::cout << "\n";

        std::cout << "% || e_exact - e_appr ||_L2 \n";
        analytical_errors_L2.Print(std::cout, nLevels);
        std::cout << "\n";

        std::cout << "% || der (e_exact - e_appr) ||_L2 \n";
        analytical_errors_curl.Print(std::cout, nLevels);
        std::cout << std::string(50,'=') << '\n';
    }

#ifdef ParELAG_ENABLE_PETSC
    PetscFinalize();
#endif

    return EXIT_SUCCESS;
}
