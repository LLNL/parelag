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

/*
 * MLHiptmairSolver.hpp
 *
 *  Created on: Apr 17, 2015
 *      Author: uvilla
 */

#ifndef SRC_LINALG_MLHIPTMAIRSOLVER_HPP_
#define SRC_LINALG_MLHIPTMAIRSOLVER_HPP_

#include <memory>

#include <mfem.hpp>

#include "elag_typedefs.hpp"
#include "amge/DeRhamSequence.hpp"
#include "linalg/legacy/ParELAG_HypreExtension.hpp"

namespace parelag
{
class HdivProblem3D
{
public:
    typedef HypreExtension::HypreADS CoarseSolver;
    typedef HypreExtension::HypreADSData CoarseSolverData;
    static int form;
};

/**
   This class is bytewise identical to HcurlProblem,
   could use a typedef I think...
*/
class HdivProblem2D
{
public:
    typedef HypreExtension::HypreAMS CoarseSolver;
    typedef HypreExtension::HypreAMSData CoarseSolverData;
    static int form;
};

class HcurlProblem
{
public:
    typedef HypreExtension::HypreAMS CoarseSolver;
    typedef HypreExtension::HypreAMSData CoarseSolverData;
    static int form;
};

class HypreSmootherData
{
public:
    HypreSmootherData();
    int type;
    int relax_times;
    double relax_weight;
    double omega;
    int poly_order;
    double poly_fraction;
};

/**
   This implements nearly Algorithm F1.1 on p. 482
   of PSV's book, see also Hiptmair and Xu.
*/
template<class PROBLEM>
class MLHiptmairSolver: public mfem::Solver
{
public:
    typedef PROBLEM Problem;  // ????
    typedef typename Problem::CoarseSolver CoarseSolver;
    typedef typename Problem::CoarseSolverData CoarseSolverData;

    MLHiptmairSolver(mfem::Array<DeRhamSequence *> & seqs, mfem::Array<int> & label_ess);
    void SetOperator(const mfem::Operator & A);
    void SetMatrix(mfem::HypreParMatrix * A);

    void SetHypreSmootherProperties(HypreSmootherData & data);
    void SetCoarseSolverProperties(CoarseSolverData & data);

    /// Operator application
    virtual void Mult (const mfem::Vector & x, mfem::Vector & y) const;

    /// Action of the transpose operator
    virtual void MultTranspose (mfem::Vector const&, mfem::Vector &) const
    {
        mfem::mfem_error("MultTranspose of MLPreconditioner not implemented");
    }

    virtual ~MLHiptmairSolver();

private:
    //! compute the hierarchy
    void compute();
    //! Recursive multigrid algorithm
    void MGCycle(int level) const;
    //! clean up data
    void cleanUp();

    //! ATB I believe Umberto is doing SA and Saux in opposite order from PSV Algorithm F1.1
    void presmoothing(int level,
                      const mfem::Vector & res,
                      mfem::Vector & sol) const;
    void postsmoothing(int level,
                       const mfem::Vector & res,
                       mfem::Vector & sol) const;

    mfem::Array<DeRhamSequence *> & seqs;
    mfem::Array<int> label_ess;

    //! number of levels
    int nLevels;

    //! A at each level (nLevels)
    //
    // FIXME (trb 12/10/15): It seems that A[0] is not guaranteed to
    // be destroyable. That's fine, but perhaps it gets special
    // treatment then? The destructor of this class will delete all of
    // the others...
    std::vector<ParallelCSRMatrix *> A;

    /**
       Auxiliary A at each level, but the coarsest (nLevels-1)
       this is D^T M D = D^T A D
    */
    std::vector<std::unique_ptr<ParallelCSRMatrix>> Aaux;

    //! Derivative operators at each levels, but the coarset (nLevels-1)
    std::vector<std::unique_ptr<ParallelCSRMatrix>> D;

    //! Prolongator operators between levels (nLevels-1)
    std::vector<std::unique_ptr<ParallelCSRMatrix>> P;

    //! A-Smoother at each level (nLevels-1)
    mfem::Array<mfem::HypreSmoother *> SA;
    //! Auxiliary-Smoother at each level (nLevels-1)
    mfem::Array<mfem::HypreSmoother *> Saux;
    HypreSmootherData Sdata;

    //! coarse solver
    CoarseSolver * coarseSolver;
    CoarseSolverData coarseSolverData;

    mutable mfem::Array<mfem::Vector*> v;   // Approximate solution on all coarse levels
    mutable mfem::Array<mfem::Vector*> d;   // Defects at all levels
    mutable mfem::Array<mfem::Vector*> t;

    mutable mfem::Vector res1;
    mutable mfem::Vector sol1;
    mutable mfem::Vector resaux;
    mutable mfem::Vector solaux;

    double arithmeticComplexity;
    double operatorComplexity;
};
}//namespace parelag
#endif /* SRC_LINALG_MLHIPTMAIRSOLVER_HPP_ */
