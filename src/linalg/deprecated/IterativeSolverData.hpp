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

#ifndef ITERATIVESOLVERDATA_HPP_
#define ITERATIVESOLVERDATA_HPP_

#include <mfem.hpp>

namespace parelag
{
    class IterativeSolverData
    {
    public:
        IterativeSolverData():
            rel_tol(1e-6),
            abs_tol(1e-12),
            max_iter(1000),
            print_level(0)
        {};

        void SetOptions(mfem::IterativeSolver & solver) const
        {
            solver.SetAbsTol(abs_tol);
            solver.SetRelTol(rel_tol);
            solver.SetMaxIter(max_iter);
            solver.SetPrintLevel(print_level);
        }

        double rel_tol;
        double abs_tol;
        int max_iter;
        int print_level;
    };
}//namespace parelag
#endif /* ITERATIVESOLVERDATA_HPP_ */
