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


#ifndef PARELAG_STATIONARYSOLVER_HPP_
#define PARELAG_STATIONARYSOLVER_HPP_

#include <mfem.hpp>

#include "linalg/solver_core/ParELAG_Solver.hpp"

namespace parelag
{

/** \class StationarySolver
 *  \brief Run a solver as a stationary iteration.
 *
 *  \warning This class primarily exists for testing purposes and is
 *  not meant to be a high-performance implementation.
 */
class StationarySolver : public Solver
{
public:
    /** \brief Constructor taking all information.
     *
     *  \param op           The underlying operator.
     *  \param solver       The solver corresponding to op.
     *  \param rel_tol      The relative residual tolerance requirement.
     *  \param abs_tol      The absolute residual tolerance requirement.
     *  \param max_its      The maximum allowed iterations.
     *  \param print_iters  Output control flag.
     */
    StationarySolver(std::shared_ptr<mfem::Operator> op,
                     std::shared_ptr<mfem::Solver> solver,
                     double rel_tol, double abs_tol, size_t max_its,
                     bool print_iters);

    /** \brief Compute the action of the solver applied to a given
     *  vector.
     *
     *  \param rhs[in]     The given right-hand side.
     *  \param sol[inout]  The initial guess, overwritten by result.
     */
    void Mult(const mfem::Vector& rhs, mfem::Vector& sol) const override;

    /** \brief Compute the action of the transpose of the solver
     *  applied to a given vector.
     *
     *  \param rhs[in]     The given right-hand side.
     *  \param sol[inout]  The initial guess, overwritten by result.
     */
    void MultTranspose(
        const mfem::Vector& rhs, mfem::Vector& sol) const override;


private:

    /** \brief Implementation of SetOperator(). */
    void _do_set_operator(const std::shared_ptr<mfem::Operator>& op) override;

private:

    /** \brief Underlying operator */
    std::shared_ptr<mfem::Operator> Op_;

    /** \brief Solver object that gets called */
    std::shared_ptr<mfem::Solver> Solver_;

    /** \brief Temporary residual storage */
    mutable mfem::Vector Residual_;

    /** \brief Temporary correction storage */
    mutable mfem::Vector Correction_;

    /** \brief Another auxiliary vector */
    mutable mfem::Vector Tmp_;

    /** \brief The relative tolerance required for convergence */
    double RelativeTol_ = 0.0;

    /** \brief The absolute tolerance required for convergence */
    double AbsoluteTol_ = 0.0;

    /** \brief The maximum allowed number of iterations */
    size_t MaxIts_ = 1;

    /** \brief The relevant communicator */
    MPI_Comm Comm_ = MPI_COMM_WORLD;

    /** \brief Flag indicating whether the solver should print information */
    bool PrintIterations_ = false;

};// class StationarySolver
}// namespace parelag
#endif /* PARELAG_STATIONARYSOLVER_HPP_ */
