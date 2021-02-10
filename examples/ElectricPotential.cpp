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

#include <fstream>
#include <sstream>

#include <mpi.h>

#include "elag.hpp"

using namespace mfem;
using namespace parelag;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

enum {TOPOLOGY=0, SEQUENCE, ASSEMBLY, PRECONDITIONER, SOLVER, NSTAGES};


// Analytical Solution: http://www.phys.uri.edu/~gerhard/PHY204/tsl94.pdf.
// Q = 1., k = 1, R = 1
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

double electricField_r(const Vector & x)
{

    double r = x.Norml2();

    if( r > 1.)
        return 1./(3.*r*r);
    else
        return r/3.;
}

double electricPotential(const Vector & x)
{
    double r = x.Norml2();
    if(r > 1.)
        return 1./(3.*r);
    else
        return .5*( 1. - (r*r)/3. );
}

int main (int argc, char *argv[])
{
    // 1. Initialize MPI
    mpi_session sess(argc,argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    StopWatch chrono;

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
                   "Target polynomial order of coarse space.");
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
    constexpr int print_iter = 0;
    constexpr int max_num_iter = 1000;
    constexpr double rtol = 1e-9;
    constexpr double atol = 1e-12;

    {
        std::stringstream msg;
        msg << "Read mesh " << meshfile   << "\n";
        msg << "Refine mesh in serial "   <<   ser_ref_levels << " times. \n";
        msg << "Refine mesh in parallel "   <<   par_ref_levels << " times. \n";
        msg << "Unstructured coarsening factor " << coarseningFactor << "\n";
        msg << "FE order " << feorder << "\n";
        msg << "Upscaling order " << upscalingOrder << "\n";

        msg << "MINRES: Max Number Iterations " << max_num_iter << "\n";
        msg << "MINRES: rtol " << rtol << "\n";
        msg << "MINRES: atol " << atol << "\n";

        RootOutput(comm, 0, std::cout, msg.str());
    }

    // 2. Read the (serial) mesh from the given mesh file and uniformely refine it.
    std::ifstream imesh(meshfile.c_str());
    if (!imesh)
    {
        std::cerr << "\nCan not open mesh file: " << meshfile.c_str() << "\n\n";
        return EXIT_FAILURE;
    }

    auto mesh = make_unique<Mesh>(imesh, 1, 1);
    imesh.close();

    for (int l = 0; l < ser_ref_levels; l++)
        mesh->UniformRefinement();

    const int nDimensions = mesh->Dimension();
    elag_assert(nDimensions == 3);

    auto pmesh = make_shared<ParMesh>(comm, *mesh);
    mesh.reset();

    const int nLevels = par_ref_levels+1;
    Array<int> level_nElements(nLevels);
    for (int l = 0; l < par_ref_levels; l++)
    {
        level_nElements[par_ref_levels-l] = pmesh->GetNE();
        pmesh->UniformRefinement();
    }
    level_nElements[0] = pmesh->GetNE();

    if(nDimensions == 3)
        pmesh->ReorientTetMesh();

    Vector source(2);
    source(0) = 1.; source(1) = 0.;
    PWConstCoefficient source_coeff(source);
    FunctionCoefficient potential_coeff(electricPotential);
    VectorFunctionCoefficient efield_coeff(nDimensions, electricField);
    FunctionCoefficient efield_r_coeff(electricField_r);
    Array<int> nat_bdr_attributes(2);
    nat_bdr_attributes[0] = 0; nat_bdr_attributes[1] = 1;
    RestrictedCoefficient fbdr(potential_coeff, nat_bdr_attributes);


    ConstantCoefficient coeffL2(1.);
    ConstantCoefficient coeffHdiv(1.);

    MFEMRefinedMeshPartitioner partitioner(nDimensions);
    std::vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    DenseMatrix timings(nLevels, NSTAGES);
    timings = 0.0;

    chrono.Clear();
    chrono.Start();
    topology[0] = make_shared<AgglomeratedTopology>(pmesh, 1);
    chrono.Stop();
    timings(0, TOPOLOGY) = chrono.RealTime();

    constexpr auto at_elem = AgglomeratedTopology::ELEMENT;
    for(int ilevel = 0; ilevel < nLevels-1; ++ilevel)
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
        if(myid == 0)
            std::cout << "Timing ELEM_AGG_Level " << ilevel
                      << ": Mesh Agglomeration done in " << chrono.RealTime()
                      << " seconds.\n";
    }

    //-----------------------------------------------------//

    std::vector<shared_ptr<DeRhamSequence>> sequence(topology.size());

    chrono.Clear();
    chrono.Start();
    sequence[0] =
        make_shared<DeRhamSequence3D_FE>(topology[0],pmesh.get(),feorder);

    DeRhamSequenceFE * DRSequence_FE = sequence[0]->FemSequence();

    int jFormStart = nDimensions-1;
    sequence[0]->SetjformStart( jFormStart );

    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 3, make_unique<MassIntegrator>(coeffL2), false);
    DRSequence_FE->ReplaceMassIntegrator(
        at_elem, 2, make_unique<VectorFEMassIntegrator>(coeffHdiv), true);


    Array<Coefficient *> L2coeff;
    Array<VectorCoefficient *> Hdivcoeff;
    fillVectorCoefficientArray(nDimensions, upscalingOrder, Hdivcoeff);
    fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);


    // We own targets, but the interface requires mfem::Array. So we
    // hold a vector of unique_ptr and create an mfem::Array to view
    // the raw ptrs.
    std::vector<unique_ptr<MultiVector>>
        targets(sequence[0]->GetNumberOfForms());
    int jform(0);

    targets[jform] = nullptr;
    ++jform;

    targets[jform] = nullptr;
    ++jform;

    targets[jform] = DRSequence_FE->InterpolateVectorTargets(jform, Hdivcoeff);
    ++jform;

    targets[jform] = DRSequence_FE->InterpolateScalarTargets(jform, L2coeff);
    ++jform;

    freeCoeffArray(L2coeff);
    freeCoeffArray(Hdivcoeff);

    // Here's the viewing mfem::Array
    Array<MultiVector *> targets_in(targets.size());
    for (int ii = 0; ii < targets_in.Size(); ++ii)
        targets_in[ii] = targets[ii].get();

    sequence[0]->SetjformStart(2);
    sequence[0]->SetTargets(targets_in);
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
        if(myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << i << ": Coarsening done in "
                      << chrono.RealTime() << " seconds.\n";
    }

    const int uform = pmesh->Dimension() - 1;
    const int pform = pmesh->Dimension();

    // testUpscalingHdiv(sequence);
    FiniteElementSpace * ufespace = sequence[0]->FemSequence()->GetFeSpace(uform);
    FiniteElementSpace * pfespace = sequence[0]->FemSequence()->GetFeSpace(pform);

    LinearForm b(ufespace);
    b.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fbdr));
    b.Assemble();
    b *= -1.;

    LinearForm q(pfespace);
    q.AddDomainIntegrator(new DomainLFIntegrator(source_coeff));
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

    const int quadrule_order = 2*feorder+2;
    Array<const IntegrationRule *> irs(Geometry::NumGeom);
    irs = nullptr;
    irs[Geometry::TETRAHEDRON] = &(IntRules.Get(Geometry::TETRAHEDRON, quadrule_order));
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
    {
        std::transform(solutionNorm.GetData(),
                       solutionNorm.GetData()+solutionNorm.Size(),
                       solutionNorm.GetData(), (double(*)(double)) sqrt);
    }

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

    std::vector<unique_ptr<SparseMatrix>> Ml(nLevels);
    std::vector<unique_ptr<SparseMatrix>> Wl(nLevels);

    for(int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        Ml[k] = sequence[k]->ComputeMassOperator(uform);
        Wl[k] = sequence[k]->ComputeMassOperator(pform);
        chrono.Stop();
        tdiff = chrono.RealTime();
        if(myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in "
                      << tdiff << "s.\n";
        timings(k, ASSEMBLY) += tdiff;
    }

    std::vector<Array<int>> blockOffsets(nLevels);
    for(int k(0); k < nLevels; ++k)
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

    for(int i = 0; i < nLevels-1; ++i)
    {
        rhs[i+1] = make_unique<BlockVector>(blockOffsets[i+1]);
        allPu[i]->MultTranspose(rhs[i]->GetBlock(0), rhs[i+1]->GetBlock(0) );
        allPp[i]->MultTranspose(rhs[i]->GetBlock(1), rhs[i+1]->GetBlock(1) );
    }

    std::vector<unique_ptr<BlockVector>> sol(nLevels);
    std::vector<unique_ptr<BlockVector>> help(nLevels);

    for(int k(0); k < nLevels; ++k)
    {
        sol[k] = make_unique<BlockVector>(blockOffsets[k]);
        help[k] = make_unique<BlockVector>(blockOffsets[k]);
        *(help[k]) = 0.;
    }

    for(int k(0); k < nLevels; ++k)
    {
        chrono.Clear();
        chrono.Start();
        auto M = Ml[k].get();
        auto W = Wl[k].get();
        auto D = allD[k];
        unique_ptr<SparseMatrix> B{Mult(*W, *D)};
        *B *= -1.;
        unique_ptr<SparseMatrix> Bt{Transpose(*B)};

        const SharingMap & l2_dofTrueDof(
            sequence[k]->GetDofHandler(pform)->GetDofTrueDof());
        const SharingMap & hdiv_dofTrueDof(
            sequence[k]->GetDofHandler(uform)->GetDofTrueDof());

        Array<int> trueBlockOffsets(3);
        trueBlockOffsets[0] = 0;
        trueBlockOffsets[1] = hdiv_dofTrueDof.GetTrueLocalSize();
        trueBlockOffsets[2] =
            trueBlockOffsets[1] + l2_dofTrueDof.GetTrueLocalSize();

        BlockVector prhs(trueBlockOffsets);
        hdiv_dofTrueDof.Assemble(rhs[k]->GetBlock(0), prhs.GetBlock(0) );
        l2_dofTrueDof.Assemble(rhs[k]->GetBlock(1), prhs.GetBlock(1) );

        auto pM = Assemble(hdiv_dofTrueDof, *M, hdiv_dofTrueDof);
        auto pB = Assemble(l2_dofTrueDof, *B, hdiv_dofTrueDof);
        auto pBt = Assemble(hdiv_dofTrueDof, *Bt, l2_dofTrueDof);

        BlockOperator op(trueBlockOffsets);
        op.owns_blocks = 0;
        op.SetBlock(0,0,pM.get());
        op.SetBlock(0,1, pBt.get());
        op.SetBlock(1,0, pB.get());

        auto tmp = Assemble(hdiv_dofTrueDof, *Bt, l2_dofTrueDof);
        Vector diag( pM->Height() );
        pM->GetDiag(diag);

        for(int i =0; i < diag.Size(); ++i)
            diag(i) = 1./diag(i);

        tmp->ScaleRows(diag);
        unique_ptr<HypreParMatrix> S{ParMult(pB.get(), tmp.get())};
        tmp.reset();

        HypreDiagScale *Mprec = new HypreDiagScale(*pM);
        HypreExtension::HypreBoomerAMG *Sprec = new HypreExtension::HypreBoomerAMG(*S);

        BlockDiagonalPreconditioner prec(trueBlockOffsets);
        prec.owns_blocks = 1;
        prec.SetDiagonalBlock(0,Mprec);
        prec.SetDiagonalBlock(1, Sprec);

        chrono.Stop();
        tdiff = chrono.RealTime();
        if(myid == 0)
            std::cout << "Timing ELEM_AGG_LEVEL " << k << ": Assembly done in "
                      << tdiff << "s. \n";
        timings(k, ASSEMBLY) += tdiff;

        ndofs[k] = pM->GetGlobalNumRows() + pB->GetGlobalNumRows();

        // Solver - H^1 - L_2
        {
            chrono.Clear();
            chrono.Start();

            {
                BlockVector x(trueBlockOffsets);
                BlockVector y(trueBlockOffsets);
                x = 1.0; y = 0.;
                prec.Mult(x,y);
            }

            chrono.Stop();
            tdiff = chrono.RealTime();

            if(myid == 0)
                std::cout << "Timing PRECONDITIONER_LEVEL " << k
                          << ": Preconditioner Computed " << tdiff << "s.\n";
            timings(k,PRECONDITIONER) = tdiff;

            BlockVector psol( trueBlockOffsets );
            psol = 0.;
            MINRESSolver minres(comm);
            minres.SetPrintLevel(print_iter);
            minres.SetMaxIter(max_num_iter);
            minres.SetRelTol(rtol);
            minres.SetAbsTol(atol);
            minres.SetOperator(op);
            minres.SetPreconditioner(prec);
            chrono.Clear();
            chrono.Start();
            minres.Mult(prhs, psol );
            chrono.Stop();
            tdiff = chrono.RealTime();
            if(myid == 0)
                std::cout << "Timing MINRES_LEVEL " << k << ": Solver done in "
                          << tdiff << "s.\n";
            timings(k,SOLVER) = tdiff;

            if(myid == 0)
            {
                if(minres.GetConverged())
                    std::cout << "Minres converged in "
                              << minres.GetNumIterations()
                              << " with a final residual norm "
                              << minres.GetFinalNorm() << ".\n";
                else
                    std::cout << "Minres did not converge in "
                              << minres.GetNumIterations()
                              << ". Final residual norm is "
                              << minres.GetFinalNorm() << "\n";
            }

            hdiv_dofTrueDof.Distribute(psol.GetBlock(0), sol[k]->GetBlock(0));
            l2_dofTrueDof.Distribute(psol.GetBlock(1), sol[k]->GetBlock(1));
            iter[k] = minres.GetNumIterations();
        }

        //ERROR NORMS
        {
            *(help[k]) = *(sol[k]);
            for(int j = k; j > 0; --j)
            {
                allPu[j-1]->Mult(help[j]->GetBlock(0), help[j-1]->GetBlock(0));
                allPp[j-1]->Mult(help[j]->GetBlock(1), help[j-1]->GetBlock(1));
            }

            Vector dsol( allD[0]->Size() );
            allD[0]->Mult(help[0]->GetBlock(0), dsol );

            GridFunction gf;
            gf.MakeRef(ufespace,help[0]->GetBlock(0), 0);
            double err = gf.ComputeL2Error(efield_coeff, irs);
            analytical_errors_L2_2(k,0) = err*err;
            gf.MakeRef(pfespace,dsol, 0);
            err = gf.ComputeL2Error(source_coeff, irs);
            analytical_errors_L2_2(k,1) = err*err;
            gf.MakeRef(pfespace,help[0]->GetBlock(1), 0);
            err = gf.ComputeL2Error(potential_coeff, irs);
            analytical_errors_L2_2(k,2) = err*err;



            for(int j(0); j < k; ++j)
            {
                if(help[j]->Size() != sol[j]->Size() || sol[j]->GetBlock(0).Size() != allD[j]->Width() )
                    mfem_error("size don't match \n");

                const int usize  = sol[j]->GetBlock(0).Size();
                const int psize  = sol[j]->GetBlock(1).Size();
                const int dsize = allD[j]->Size();
                Vector u_H( help[j]->GetData(), usize );
                Vector u_h( sol[j]->GetData(), usize  );
                Vector p_H( help[j]->GetData(), psize );
                Vector p_h( sol[j]->GetData(), psize  );
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
        if (do_visualize)
        {
            MultiVector u(sol[k]->GetData(), 1, sol[k]->GetBlock(0).Size() );
            sequence[k]->show(uform, u);
            MultiVector p(sol[k]->GetBlock(1).GetData(), 1, sol[k]->GetBlock(1).Size() );
            sequence[k]->show(pform, p);
        }
    }

    DenseMatrix u_errors_L2(nLevels, nLevels);
    DenseMatrix p_errors_L2(nLevels, nLevels);
    DenseMatrix errors_div(nLevels, nLevels);
    DenseMatrix analytical_errors(nLevels, 3);
    // DenseMatrix timings(nLevels-1, NSTAGES);

    MPI_Reduce(u_errors_L2_2.Data(), u_errors_L2.Data(),
               u_errors_L2.Height()*u_errors_L2.Width(),
               MPI_DOUBLE,MPI_SUM, 0, comm);
    MPI_Reduce(p_errors_L2_2.Data(), p_errors_L2.Data(),
               p_errors_L2.Height()*p_errors_L2.Width(),
               MPI_DOUBLE,MPI_SUM, 0, comm);
    MPI_Reduce(errors_div_2.Data(), errors_div.Data(),
               errors_div.Height()*errors_div.Width(),
               MPI_DOUBLE,MPI_SUM, 0, comm);
    MPI_Reduce(analytical_errors_L2_2.Data(), analytical_errors.Data(),
               analytical_errors_L2_2.Height()*analytical_errors_L2_2.Width(),
               MPI_DOUBLE,MPI_SUM, 0, comm);

    if(myid == 0)
    {
        std::transform(u_errors_L2.Data(),
                       u_errors_L2.Data()+u_errors_L2.Height()*u_errors_L2.Width(),
                       u_errors_L2.Data(), (double(*)(double)) sqrt);
        std::transform(p_errors_L2.Data(),
                       p_errors_L2.Data()+p_errors_L2.Height()*p_errors_L2.Width(),
                       p_errors_L2.Data(), (double(*)(double)) sqrt);
        std::transform(errors_div.Data(),
                       errors_div.Data()+errors_div.Height()*errors_div.Width(),
                       errors_div.Data(), (double(*)(double)) sqrt);
        std::transform(analytical_errors.Data(),
                       analytical_errors.Data()+analytical_errors.Height()*analytical_errors.Width(),
                       analytical_errors.Data(), (double(*)(double)) sqrt);
    }

    if(myid == 0)
    {
        std::cout << "\n{\n";
        int w = 14;
        std::cout << "%level" << std::setw(w) << "Topology"
                  << std::setw(w) << "Sequence\n";
        for(int i(0); i < nLevels; ++i)
            std::cout<< i << std::setw(w) << timings(i,TOPOLOGY)
                     << std::setw(w) << timings(i,SEQUENCE) << "\n";
        std::cout << "}\n";

        std::cout << "\n{\n";
        std::cout << "%level" << std::setw(w) << "size" << std::setw(w) << "nit"
                  << std::setw(w) << "Assembly" << std::setw(w) << "Prec "
                  << std::setw(w) << "Solver\n";
        for(int i(0); i < nLevels; ++i)
            std::cout << i << std::setw(w) << ndofs[i] << std::setw(w) << iter[i]
                      << std::setw(w) << timings(i,ASSEMBLY)
                      << std::setw(w) << timings(i,PRECONDITIONER)
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

    return EXIT_SUCCESS;
}
