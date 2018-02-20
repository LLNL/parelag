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
   collect some shared code from UpscalingForm and related
   examples, so not everything is copy/pasted all the time

   Andrew T. Barker
   atb@llnl.gov
   2 September 2015
*/
#include <memory>

#include "UpscalingPieces.hpp"
#include "utilities/MemoryUtils.hpp"

using namespace mfem;
using std::unique_ptr;

namespace parelag
{
unique_ptr<SparseMatrix> ExampleRAP(const SparseMatrix & Rt,
                                    const SparseMatrix & A,
                                    const SparseMatrix & P)
{
    unique_ptr<SparseMatrix> AP{Mult(A,P)};
    unique_ptr<SparseMatrix> R{Transpose(Rt)};
    return unique_ptr<SparseMatrix>{Mult(*R,*AP)};
}
// In a world where Transpose and Mult returned
// unique_ptr<SparseMatrix>, this function could be written in 1 line:
//
// return Mult(*Mult(*Transpose(Rt),A),P);
//
// or
//
// return Mult(*Transpose(Rt),*Mult(A,P));
//
// Wouldn't that be fun?


void OutputUpscalingTimings(const Array<int>& ndofs,
                            const Array<int>& nnz,
                            const Array2D<int>& iter,
                            const DenseMatrix& timings,
                            const char ** solver_names,
                            const char ** stage_names)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);

    const int nLevels = timings.Height();
    const int NSTAGES = timings.Width();
    const int NSOLVERS = iter.NumCols();

    if (myid == 0)
    {
        std::cout << std::endl << "{" << std::endl;
        constexpr int w = 14;
        std::cout << "%level" << std::setw(w) << "size"
                  << std::setw(w) << "nnz";
        for (int i_solver=0; i_solver<NSOLVERS; ++i_solver)
            std::cout << std::setw(w) << "nit " << solver_names[i_solver];
        for (int i_stage=0; i_stage<NSTAGES; ++i_stage)
            std::cout << std::setw(w) << stage_names[i_stage];
        std::cout << std::endl;
        for (int i_level=0; i_level<nLevels; ++i_level)
        {
            std::cout << i_level << std::setw(w)
                      << ndofs[i_level] << std::setw(w)
                      << nnz[i_level];
            for (int i_solver=0; i_solver<NSOLVERS; ++i_solver)
                std::cout << std::setw(w) << iter(i_level,i_solver);
            for (int i_stage=0; i_stage<NSTAGES; ++i_stage)
                std::cout << std::setw(w) << timings(i_level,i_stage);
            std::cout << std::endl;
        }
        std::cout << "}" << std::endl;
    }
}

void OutputUpscalingTimings(const Array<int>& ndofs,
                            const Array<int>& nnz,
                            const Array<int>& iter,
                            const DenseMatrix& timings,
                            const char ** stage_names)
{
    const char * solver_names[] = {""};
    Array2D<int> iter2d(iter.Size(),1);
    for (int i=0; i<iter.Size(); ++i)
        iter2d(i,0) = iter[i];
    OutputUpscalingTimings(ndofs, nnz, iter2d, timings, solver_names, stage_names);
}


void OutputUpscalingErrors(const DenseMatrix& u_errors_L2,
                           const Vector& u_norms_L2,
                           const DenseMatrix& p_errors_L2,
                           const Vector& p_norms_L2,
                           const DenseMatrix& errors_der,
                           const Vector& norms_der)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);
    int nLevels = u_errors_L2.Height();

    // the block below is legacy output that we used to be able
    // to parse with some python scripts, we should maybe move
    // to something more generic, flexible and "nicer" (JSON? XML?)
    if (myid == 0)
    {
        std::cout << std::endl << "{" << std::endl;
        std::cout << "% || uh - uH ||" << std::endl;
        u_errors_L2.PrintMatlab(std::cout);
        std::cout << "% || uh ||" << std::endl;
        u_norms_L2.Print(std::cout, nLevels);
        if (p_errors_L2.Height() > 0)
        {
            std::cout << "% || ph - pH ||" << std::endl;
            p_errors_L2.PrintMatlab(std::cout);
            std::cout << "% || ph ||" << std::endl;
            p_norms_L2.Print(std::cout, nLevels);
        }
        std::cout << "% || der ( uh - uH ) ||" << std::endl;
        errors_der.PrintMatlab(std::cout);
        std::cout << "% || der uh ||" << std::endl;
        norms_der.Print(std::cout, nLevels);
        std::cout << "}" << std::endl;
    }

    // for now we are doing this to be GREP-able
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
        h = errors_der.Height();
        std::cout << "u energy-like errors: ";
        for (int i=0; i<h-1; ++i)
            std::cout << std::scientific << std::setprecision(4)
                      << errors_der.Elem(h-1-i,0) << " ";
        std::cout << std::endl;
    }
}

void OutputUpscalingErrors(const DenseMatrix& u_errors_L2,
                           const Vector& u_norms_L2,
                           const DenseMatrix& errors_der,
                           const Vector& norms_der)
{
    DenseMatrix dummy_mat(0,0);
    Vector dummy_vec(0);

    OutputUpscalingErrors(u_errors_L2, u_norms_L2,
                          dummy_mat, dummy_vec,
                          errors_der, norms_der);
}

void ReduceAndOutputUpscalingErrors(const DenseMatrix& u_errors_L2_2,
                                    const Vector& u_norm_L2_2,
                                    const DenseMatrix& p_errors_L2_2,
                                    const Vector& p_norm_L2_2,
                                    const DenseMatrix& errors_der_2,
                                    const Vector& norm_der_2)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);
    int nLevels = u_errors_L2_2.Height();

    DenseMatrix u_errors_L2(nLevels, nLevels);
    Vector u_norm_L2(nLevels);
    DenseMatrix p_errors_L2(p_errors_L2_2.Height(), p_errors_L2_2.Width());
    Vector p_norm_L2(p_norm_L2_2.Size());
    DenseMatrix errors_der(nLevels, nLevels);
    Vector norm_der(nLevels);

    MPI_Reduce(u_errors_L2_2.Data(), u_errors_L2.Data(),
               u_errors_L2.Height()*u_errors_L2.Width(),
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(u_norm_L2_2.GetData(), u_norm_L2.GetData(),
               u_norm_L2.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    if (p_errors_L2_2.Height() > 0)
    {
        MPI_Reduce(p_errors_L2_2.Data(), p_errors_L2.Data(),
                   p_errors_L2.Height()*p_errors_L2.Width(),
                   MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(p_norm_L2_2.GetData(), p_norm_L2.GetData(),
                   p_norm_L2.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    }
    MPI_Reduce(errors_der_2.Data(), errors_der.Data(),
               errors_der.Height()*errors_der.Width(),
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(norm_der_2.GetData(), norm_der.GetData(),
               norm_der.Size(), MPI_DOUBLE, MPI_SUM, 0, comm);

    if (myid == 0)
    {
        auto self_sqrt = [](double& a){a = std::sqrt(a);};
        std::for_each(
            u_errors_L2.Data(),
            u_errors_L2.Data()+u_errors_L2.Height()*u_errors_L2.Width(),
            self_sqrt);
        std::for_each(
            u_norm_L2.GetData(), u_norm_L2.GetData()+u_norm_L2.Size(),
            self_sqrt);
            
        if (p_errors_L2_2.Height() > 0)
        {
            std::for_each(
                p_errors_L2.Data(),
                p_errors_L2.Data()+p_errors_L2.Height()*p_errors_L2.Width(),
                self_sqrt);
            std::for_each(
                p_norm_L2.GetData(), p_norm_L2.GetData()+p_norm_L2.Size(),
                self_sqrt);
        }
        std::for_each(
            errors_der.Data(),
            errors_der.Data()+errors_der.Height()*errors_der.Width(),
            self_sqrt);
        std::for_each(
            norm_der.GetData(), norm_der.GetData()+norm_der.Size(),
            self_sqrt);
    }

    OutputUpscalingErrors(u_errors_L2, u_norm_L2,
                          p_errors_L2, p_norm_L2,
                          errors_der, norm_der);
}

void ReduceAndOutputUpscalingErrors(const DenseMatrix& u_errors_L2_2,
                                    const Vector& u_norm_L2_2,
                                    const DenseMatrix& errors_der_2,
                                    const Vector& norm_der_2)
{
    DenseMatrix dummy_mat(0,0);
    Vector dummy_vec(0);

    ReduceAndOutputUpscalingErrors(u_errors_L2_2, u_norm_L2_2,
                                   dummy_mat, dummy_vec,
                                   errors_der_2, norm_der_2);
}

/**
   Solves with AMG, AMS, ADS depending on form

   this is a lot of arguments and I'm not sure it's worth it...
*/
int UpscalingHypreSolver(int form,
                         HypreParMatrix *pA,
                         const Vector &prhs,
                         DeRhamSequence * sequence,
                         int k, int prec_timing_index,
                         int solver_timing_index,
                         int print_iter,
                         int max_num_iter,
                         double rtol, double atol,
                         DenseMatrix& timings,
                         const SharingMap& form_dofTrueDof,
                         Vector &solution_out,
                         bool report_timing)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int myid;
    MPI_Comm_rank(comm, &myid);

    StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    std::unique_ptr<HypreSolver> prec;
    HypreExtension::HypreBoomerAMGData amg_data;
    HypreExtension::HypreAMSData ams_data;
    HypreExtension::HypreADSData ads_data;
    if (form == 0)
        prec = make_unique<HypreExtension::HypreBoomerAMG>(*pA, amg_data);
    else if (form == 1)
        prec = make_unique<HypreExtension::HypreAMS>(*pA,sequence,ams_data);
    else if (form == 2)
    {
        ads_data.dataAMG.theta = 0.6;
        ads_data.dataAMS.dataAlpha.theta = 0.6;
        prec = make_unique<HypreExtension::HypreADS>(*pA,sequence,ads_data);
    }
    Vector tmp1(pA->Height()), tmp2(pA->Height() );
    tmp1 = 1.; tmp2 = 2.;
    prec->Mult(tmp1, tmp2);
    chrono.Stop();
    double tdiff = chrono.RealTime();
    if (myid == 0 && report_timing)
        std::cout << "Timing HYPRE_LEVEL " << k
                  << ": Preconditioner done in " << tdiff << "s."
                  << std::endl;
    timings(k, prec_timing_index) = tdiff;

    Vector psol( pA->Height() );
    psol = 0.;
    CGSolver pcg(comm);
    pcg.SetPrintLevel(print_iter);
    pcg.SetMaxIter(max_num_iter);
    pcg.SetRelTol(rtol);
    pcg.SetAbsTol(atol);
    pcg.SetOperator(*pA);
    pcg.SetPreconditioner(*prec);
    chrono.Clear();
    chrono.Start();
    pcg.Mult(prhs, psol);
    chrono.Stop();
    tdiff = chrono.RealTime();
    if (myid == 0 && report_timing)
        std::cout << "Timing HYPRE_LEVEL " << k
                  << ": Solver done in " << tdiff << "s." << std::endl;
    timings(k, solver_timing_index) = tdiff;

    if(myid == 0)
    {
        if(pcg.GetConverged())
            std::cout << "PCG converged in " << pcg.GetNumIterations()
                      << " with a final residual norm " << pcg.GetFinalNorm()
                      << "." << std::endl;
        else
            std::cout << "PCG did not converge in " << pcg.GetNumIterations()
                      << ". Final residual norm is " << pcg.GetFinalNorm()
                      << "." << std::endl;
    }
    form_dofTrueDof.Distribute(psol, solution_out);
    return pcg.GetNumIterations();
}

#ifdef ParELAG_ENABLE_PETSC
int UpscalingPetscSolver(int form,
                         PetscParMatrix *pA,
                         const Vector &prhs,
                         DeRhamSequence * sequence,
                         int k,
                         Array<int> *ebdr,
                         Array<int> *nbdr,
                         int prec_timing_index,
                         int solver_timing_index,
                         int print_iter,
                         int max_num_iter,
                         double rtol, double atol,
                         DenseMatrix& timings,
                         const SharingMap& form_dofTrueDof,
                         Vector &solution_out,
                         bool report_timing)
{
    MPI_Comm comm = pA->GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);

    PetscErrorCode ierr;

    // options prefix for command line customization
    // i.e. -parelag_level_ksp_type gmres -parelag_level_pc_type ilu
    std::string prefix = "parelag_level" + std::to_string(k) + "_";

    StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    std::unique_ptr<PetscPreconditioner> prec;

    // depending on the matrix type, we select the preconditioner
    PetscBool ismatis;
    Mat A = *pA;
    ierr = PetscObjectTypeCompare((PetscObject)A,MATIS,&ismatis);
    CHKERRQ(ierr);
    if (ismatis)
    {
        PetscBDDCSolverParams params;
        params.SetEssBdrDofs(ebdr,true);
        params.SetNatBdrDofs(nbdr,true);
        prec = make_unique<PetscBDDCSolver>(*pA,params,prefix);
        if (form == 1) // H(curl)
        {
          // We need the discrete gradient
          PC pc = *prec;
          PetscParMatrix G(sequence->ComputeTrueD(0).release(),
                           Operator::PETSC_MATAIJ);
          ierr = PCBDDCSetDiscreteGradient(pc, G, 0, PETSC_DECIDE, PETSC_TRUE,
                                           PETSC_TRUE);
          CHKERRQ(ierr);
        }
        else if (form == 2) // H(div)
        {
          // Need to assemble on \int_{\partial\Omega_i} u \dot n
          // We assemble \int_{\Omega_i} div u p and the BDDC code will use
          // the divergence theorem
          int pform = form + 1;
          const SharingMap & l2(sequence->GetDofHandler(pform)->GetDofTrueDof());
          const SharingMap & hdiv(sequence->GetDofHandler(form)->GetDofTrueDof());
          SparseMatrix* D = sequence->GetDerivativeOperator(form);
          unique_ptr<SparseMatrix> W = sequence->ComputeMassOperator(form+1);
          auto B = ToUnique(Mult(*W, *D));
          auto pB = AssemblePetsc(l2, *B, hdiv, Operator::PETSC_MATIS);
          PC pc = *prec;
          ierr = PCBDDCSetDivergenceMat(pc,*pB,PETSC_FALSE,NULL);
          CHKERRQ(ierr);
        }
    }
    else  // matrix is in AIJ format, so use command line customization
    {
        prec = make_unique<PetscPreconditioner>(*pA,prefix);
    }

    Vector tmp1(pA->Height()), tmp2(pA->Height() );
    tmp1 = 1.; tmp2 = 2.;
    prec->Mult(tmp1, tmp2);
    chrono.Stop();
    double tdiff = chrono.RealTime();
    if (myid == 0 && report_timing)
        std::cout << "Timing PETSC_LEVEL " << k
                  << ": Preconditioner done in " << tdiff << "s."
                  << std::endl;
    timings(k, prec_timing_index) = tdiff;

    Vector psol( pA->Height() );
    PetscPCGSolver pcg(comm,prefix);
    pcg.SetPrintLevel(print_iter);
    pcg.SetMaxIter(max_num_iter);
    pcg.SetRelTol(rtol);
    pcg.SetAbsTol(atol);
    pcg.SetOperator(*pA);
    pcg.SetPreconditioner(*prec);
    chrono.Clear();
    chrono.Start();
    pcg.Mult(prhs, psol);
    chrono.Stop();
    tdiff = chrono.RealTime();
    if (myid == 0 && report_timing)
        std::cout << "Timing PETSC_LEVEL " << k
                  << ": Solver done in " << tdiff << "s." << std::endl;
    timings(k, solver_timing_index) = tdiff;

    if(myid == 0)
    {
        if(pcg.GetConverged())
            std::cout << "PCG converged in " << pcg.GetNumIterations()
                      << " with a final residual norm " << pcg.GetFinalNorm()
                      << "." << std::endl;
        else
            std::cout << "PCG did not converge in " << pcg.GetNumIterations()
                      << ". Final residual norm is " << pcg.GetFinalNorm()
                      << "." << std::endl;
    }
    form_dofTrueDof.Distribute(psol, solution_out);
    return pcg.GetNumIterations();
}
#endif
}//namespace parelag
