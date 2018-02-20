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

#include <iostream>
#include <memory>
#include <vector>

#include <mfem.hpp>

#include "../ParELAG_SuperLUDistSolver.hpp"
#include "../ParELAG_StrumpackSolver.hpp"
#include "../ParELAG_SimpleXMLParameterListReader.hpp"
#include "utilities/elagError.hpp"
#include "utilities/MemoryUtils.hpp"

struct mpi_session
{
    mpi_session(int argc, char **argv) { MPI_Init(&argc,&argv); }
    ~mpi_session() { MPI_Finalize(); }
};

int main(int argc, char **argv)
{
    mpi_session mpi_sess(argc,argv);

    int mysize, myrank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&mysize);

    parelag::SimpleXMLParameterListReader reader;
    auto pl = reader.GetParameterList("test.xml");
    if (myrank == 0)
        pl->Print(std::cout);

    if (mysize > 8)
    {
        if (myrank == 0)
            std::cerr << "strum_test only runs for 8 or fewer procs. "
                      << "sorry. but not actually." << std::endl;
        return -1;
    }

    // Be lazy -- create the whole partitioning on each proc
    const int global_num_rows = 16;
    const int global_num_cols = global_num_rows;
    PARELAG_ASSERT(global_num_rows >= 6);
    //const int global_nnz = 22 + (global_num_rows-6)*5;

    std::vector<int> rc_starts(mysize+1);
    for (int ii = 0; ii < mysize; ++ii)
        rc_starts[ii] = ii*(global_num_rows/mysize);
    rc_starts.back() = global_num_rows;

    const int my_first_row = rc_starts[myrank];
    const int my_last_row = rc_starts[myrank+1];
    const int my_num_rows = my_last_row - my_first_row;

    // Create the matrix
    std::vector<int> row_ptr(my_num_rows+1),col_ind;
    std::vector<double> values,x_values(my_num_rows);

    // First row starts at 0
    row_ptr[0] = 0;

    for (int ii = 0; ii < my_num_rows; ++ii)
    {
        const int glob_r = my_first_row+ii;
        x_values[ii] = 0.25*(glob_r+1.0);
        for (int glob_c = glob_r - 3; glob_c < glob_r; glob_c += 2)
        {
            if (glob_c < 0)
                continue;

            col_ind.push_back(glob_c);
            values.push_back(-1.125);
        }

        col_ind.push_back(glob_r);
        values.push_back(6.25);

        for(int glob_c = glob_r+1; glob_c < glob_r+4; glob_c += 2)
        {
            if (glob_c < global_num_rows)
            {
                col_ind.push_back(glob_c);
                values.push_back(-1.125);
            }
        }
        row_ptr[ii+1] = static_cast<int>(col_ind.size());
    }

    auto par_A = std::make_shared<mfem::HypreParMatrix>(
        comm, my_num_rows, global_num_rows, global_num_cols,
        row_ptr.data(), col_ind.data(), values.data(),
        rc_starts.data()+myrank,rc_starts.data()+myrank);

    auto par_B = parelag::make_unique<mfem::HypreParVector>(*par_A,1);
    auto par_X = parelag::make_unique<mfem::HypreParVector>(*par_A,0);
    auto par_X_slu = parelag::make_unique<mfem::HypreParVector>(*par_A,0);

    for (int ii = 0; ii < my_num_rows; ++ii)
        (*par_X)(ii) = x_values[ii];

    par_A->Mult(*par_X,*par_B);
    *par_X = 0.0;

    // if (!myrank)
    // {
    //     for (int ii = 0; ii < par_X->Size(); ++ii)
    //         std::cout << (*par_X)(ii) << std::endl;
    //     std::cout << std::endl << std::endl;
    // }

    parelag::StrumpackSolver<double,double,int> strum(par_A);
    strum.Factor();
    strum.Mult(*par_B,*par_X);

    parelag::SuperLUDistSolver<double> slu(par_A);
    slu.Factor();
    slu.Mult(*par_B,*par_X_slu);

    *par_X_slu -= *par_X;

    if (!myrank)
        std::cout << "Test 1: ||x_slu - x_strum|| = " << par_X_slu->Norml2() << " --> "
                  << (par_X_slu->Norml2() < 1e-10 ? "Success!" : "Failure. :(")
                  << '\n';
    // {
    //     for (int ii = 0; ii < par_X->Size(); ++ii)
    //         std::cout << (*par_X)(ii) << std::endl;
    // }

    mfem::Vector myB(par_B->Size()),myX(par_X->Size()),myX_slu(par_X_slu->Size());
    for (int ii = 0; ii < my_num_rows; ++ii)
        myX(ii) = x_values[ii];

    par_A->Mult(myX,myB);
    myX = 0.0;
    myX_slu = 0.0;

    // if (!myrank)
    // {
    //     for (int ii = 0; ii < myX.Size(); ++ii)
    //         std::cout << myX(ii) << std::endl;
    //     std::cout << std::endl << std::endl;
    // }

    // for (int ii = 0; ii < 2; ++ii)
    //     slu.Factor();
    // for (int ii = 0; ii < 2; ++ii)
    strum.Mult(myB,myX);
    slu.Mult(myB,myX_slu);

    par_A->Mult(-1.0,myX,1.0,myB);

    if (!myrank)
        std::cout << "Test 2: ||b - A*x_strum|| = " << myB.Norml2() << " --> "
                  << (myB.Norml2() < 1e-10 ? "Success!" : "Failure. :(")
                  << '\n';

    // {
    //     for (int ii = 0; ii < myX.Size(); ++ii)
    //         std::cout << myX(ii) << std::endl;
    // }

    return 0;
}
