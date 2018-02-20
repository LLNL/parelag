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

#include "../ParELAG_TimeManager.hpp"
#include "utilities/MemoryUtils.hpp"

#include <array>

namespace tt
{
struct mpi_session
{
    mpi_session(int argc, char **argv)
    {
        MPI_Init(&argc,&argv);
    }
    ~mpi_session()
    {
        MPI_Finalize();
    }
};

struct widget
{
    widget(int blah)
        : blah_(blah)
    {}

    int blah_;
};

}

int main(int argc, char **argv)
{
    tt::mpi_session session(argc,argv);

    int myrank,mysize;
    MPI_Comm_size(MPI_COMM_WORLD,&mysize);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    auto time = parelag::TimeManager::AddTimer("total time");

    std::array<double,1000> vec;
    vec.fill(1.0);
    auto vsize = vec.size();

    {
        auto time = parelag::TimeManager::AddTimer("partial sum");

        for (int count = 0; count < 1000; ++count)
            for (auto i = decltype(vsize){1}; i < vsize; ++i)
                vec[i] += vec[i-1] / 4.0;
    }

    {
        auto time = parelag::TimeManager::AddTimer("weird thing");
        vec.fill(1.0);

        for (int count = 0; count < 1000; ++count)
            for (auto i = decltype(vsize){1}; i < vsize; ++i)
                vec[i] += vec[i-1] / 4.0;
    }

    parelag::TimeManager::Print(std::cout);
    parelag::TimeManager::PrintSerial(std::cout,-1,MPI_COMM_WORLD);

    if (mysize > 1)
        parelag::TimeManager::PrintSerial(std::cout,1,MPI_COMM_WORLD);


    parelag::TimeManager steve;
    {
        auto time = steve.AddTimer("Do a thing");
        for (int count = 0; count < 1000; ++count)
            for (auto i = decltype(vsize){1}; i < vsize; ++i)
                vec[i] += vec[i-1] / 4.0;
    }
    steve.Print();

    std::cout << "Unit test passed." << std::endl;
    return 0;
}
