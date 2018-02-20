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

// A quick test of the block-2x2 LU preconditioner.

// The matrix is
//
// [ 2 1 0 0 ][ 1 0 ]
// [ 3 4 3 0 ][ 1 1 ]
// [ 0 3 4 3 ][ 1 1 ]
// [ 0 0 1 2 ][ 0 1 ]
// ------------------
// [ 2 1 2 0 ][ 3 0 ]
// [ 0 2 1 2 ][ 1 3 ]
//
// Initial guess is:
//
// [ 1 2 3 4 5 6 ]^T
//
// Right-hand side is:
//
// [ 2 1 3 1 4 1 ]^T
//
#include <iostream>
#include <memory>
#include <vector>

#include <mfem.hpp>

#include "utilities/elagError.hpp"
#include "utilities/mpiUtils.hpp"
#include "utilities/MemoryUtils.hpp"

#include "../ParELAG_ParameterList.hpp"
#include "../ParELAG_MfemBlockOperator.hpp"
#include "../ParELAG_SolverLibrary.hpp"

using namespace parelag;

std::unique_ptr<MfemBlockOperator> BuildMatrix()
{
    int mysize, myrank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&mysize);

    PARELAG_ASSERT(mysize == 1);

    auto size_a = 4u;
    auto size_b = 2u;

    int * row_starts_a = (int*) malloc(sizeof(int)*2);
    int * col_starts_a = (int*) malloc(sizeof(int)*2);

    row_starts_a[0] = col_starts_a[0] = 0;
    row_starts_a[1] = col_starts_a[1] = size_a;

    int * row_starts_b = (int*) malloc(sizeof(int)*2);
    int * col_starts_b = (int*) malloc(sizeof(int)*2);

    row_starts_b[0] = col_starts_b[0] = 0;
    row_starts_b[1] = col_starts_b[1] = size_b;

    auto* M = hypre_ParCSRMatrixCreate(
        comm,size_a,size_a,row_starts_a,col_starts_a,0,10,0);
    hypre_ParCSRMatrixOwnsRowStarts(M) = true;
    hypre_ParCSRMatrixOwnsColStarts(M) = true;

    auto* B = hypre_ParCSRMatrixCreate(
        comm,size_b,size_a,row_starts_b,col_starts_a,0, 6,0);
    hypre_ParCSRMatrixOwnsRowStarts(B) = false;
    hypre_ParCSRMatrixOwnsColStarts(B) = false;

    auto* Bt = hypre_ParCSRMatrixCreate(
        comm,size_a,size_b,row_starts_a,col_starts_b,0, 6,0);
    hypre_ParCSRMatrixOwnsRowStarts(Bt) = false;
    hypre_ParCSRMatrixOwnsColStarts(Bt) = false;

    auto* C = hypre_ParCSRMatrixCreate(
        comm,size_b,size_b,row_starts_b,col_starts_b,0, 3,0);
    hypre_ParCSRMatrixOwnsRowStarts(C) = true;
    hypre_ParCSRMatrixOwnsColStarts(C) = true;

    hypre_ParCSRMatrixInitialize(M);
    hypre_ParCSRMatrixInitialize(B);
    hypre_ParCSRMatrixInitialize(Bt);
    hypre_ParCSRMatrixInitialize(C);

    {
        std::vector<int> I_M = { 0, 2, 5, 8, 10 };
        std::vector<int> J_M = { 0, 1, 1, 0, 2, 2, 1, 3, 3, 2 };
        std::vector<double> D_M = { 2., 1., 4., 3., 3., 4., 3., 3., 2., 1. };

        std::copy(I_M.begin(),I_M.end(),M->diag->i);
        std::copy(J_M.begin(),J_M.end(),M->diag->j);
        std::copy(D_M.begin(),D_M.end(),M->diag->data);
    }
    {
        std::vector<int> I_B = { 0, 3, 6 };
        std::vector<int> J_B = { 0, 1, 2, 1, 2, 3 };
        std::vector<double> D_B = { 2., 1., 2., 2., 1., 2. };

        std::copy(I_B.begin(),I_B.end(),B->diag->i);
        std::copy(J_B.begin(),J_B.end(),B->diag->j);
        std::copy(D_B.begin(),D_B.end(),B->diag->data);
    }
    {
        std::vector<int> I_Bt = { 0, 1, 3, 5, 6 };
        std::vector<int> J_Bt = { 0, 0, 1, 0, 1, 1 };
        std::vector<double> D_Bt = { 1., 1., 1., 1., 1., 1. };

        std::copy(I_Bt.begin(),I_Bt.end(),Bt->diag->i);
        std::copy(J_Bt.begin(),J_Bt.end(),Bt->diag->j);
        std::copy(D_Bt.begin(),D_Bt.end(),Bt->diag->data);
    }
    {
        std::vector<int> I_C = { 0, 1, 3 };
        std::vector<int> J_C = { 0, 1, 0 };
        std::vector<double> D_C = { 3., 3., 1. };

        std::copy(I_C.begin(),I_C.end(),C->diag->i);
        std::copy(J_C.begin(),J_C.end(),C->diag->j);
        std::copy(D_C.begin(),D_C.end(),C->diag->data);
    }


    auto out = make_unique<MfemBlockOperator>(std::vector<int>({0,4,6}),
                                              std::vector<int>({0,4,6}));
    out->SetBlock(0,0,std::make_shared<mfem::HypreParMatrix>(M));
    out->SetBlock(0,1,std::make_shared<mfem::HypreParMatrix>(Bt));
    out->SetBlock(1,0,std::make_shared<mfem::HypreParMatrix>(B));
    out->SetBlock(1,1,std::make_shared<mfem::HypreParMatrix>(C));

    return out;
}

int main(int argc, char** argv)
{
    mpi_session sess(argc,argv);

    auto matrix = std::shared_ptr<MfemBlockOperator>(BuildMatrix());

    mfem::Vector b(6), x0(6);
    {
        std::vector<double> tb = { 2., 1., 3., 1., 4., 1. };
        std::vector<double> tx0 = { 1., 2., 3., 4., 5., 6. };

        std::copy(tb.begin(), tb.end(), b.GetData());
        std::copy(tx0.begin(), tx0.end(), x0.GetData());
    }

    ParameterList solver_parameters("Preconditioner Library");

    auto& gs_list = solver_parameters.Sublist("Gauss-Seidel");
    gs_list.Get("Type","Hypre");
    auto gs_params = gs_list.Sublist("Solver Parameters");
    gs_params.Get("Type","Gauss-Seidel");

    auto& block_gs_list = solver_parameters.Sublist("BlkGS");
    block_gs_list.Get("Type","Block GS");

    auto& blkgs_params = block_gs_list.Sublist("Solver Parameters");
    blkgs_params.Get("A00 Inverse", "Gauss-Seidel");
    blkgs_params.Get("A11 Inverse", "Gauss-Seidel");
    blkgs_params.Get("S Type", "Diagonal");

    auto& block_jac_list = solver_parameters.Sublist("BlkJacobi");
    block_jac_list.Get("Type","Block Jacobi");

    auto& blkjac_params = block_jac_list.Sublist("Solver Parameters");
    blkjac_params.Get("A00 Inverse", "Gauss-Seidel");
    blkjac_params.Get("A11 Inverse", "Gauss-Seidel");
    blkjac_params.Get("S Type", "Diagonal");

    /****************************************************************/

    {
        mfem::Vector tmp(6);
        matrix->Mult(x0,tmp);
        tmp *= -1.;
        tmp += b;
        for (int ii = 0; ii < tmp.Size(); ++ii)
            std::cout << tmp[ii] << "\n";
    }

    auto lib = SolverLibrary::CreateLibrary(solver_parameters);
    auto prec_factory = lib->GetSolverFactory("BlkGS");
    auto solver_state = prec_factory->GetDefaultState();
    solver_state->SetForms({0,0});

    auto solver = prec_factory->BuildSolver(matrix,*solver_state);
    solver->iterative_mode = true;

    solver->Mult(b,x0);

    for (int ii = 0; ii < 6; ++ii)
        std::cout << std::setprecision(15) << x0[ii] << "\n";

    return EXIT_SUCCESS;
}
