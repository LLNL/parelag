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


#ifndef PARELAG_HYPRESMOOTHERWRAPPER_HPP_
#define PARELAG_HYPRESMOOTHERWRAPPER_HPP_

#include "linalg/solver_core/ParELAG_Solver.hpp"

namespace parelag
{

/** \class HypreSmootherWrapper
 *  \brief Wrapper for HyperSmoother objects.
 *
 *  The map from string types to numeric types is given here:
 *
 *  String Type                  | Int Type
 *  ---------------------------- | --------
 *   "Jacobi"                    | 0
 *   "L1 Jacobi"                 | 1
 *   "L1 Gauss-Seidel"           | 2
 *   "Kaczmarz"                  | 3
 *   "L1 Gauss-Seidel Truncated" | 4
 *   "Lumped Jacobi"             | 5
 *   "Gauss-Seidel"              | 6
 *   "Chebyshev"                 | 16
 *   "Taubin"                    | 1001
 *   "FIR"                       | 1002
 *
 *  The meaningful parameters for this class are:
 *
 *  Name                   | Type   | Default value
 *  ---------------------- | ------ | -------------
 *   "Sweeps"              | int    | 1
 *   "Damping Factor"      | double | 1.0
 *   "Omega"               | double | 1.0
 *   "Cheby Poly Order"    | int    | 2
 *   "Cheby Poly Fraction" | double | 0.3
 */
class HypreSmootherWrapper : public Solver
{
public:

    /** \brief Construct from an operator, type, and ParameterList
     *
     *  \param A The underlying operator.
     *  \param type The numeric type of the selected smoother.
     *  \param params The parameters for the solver.
     */
    HypreSmootherWrapper(
        const std::shared_ptr<mfem::Operator>& A,
        int type, ParameterList& params);

    /** \brief Destructor. */
    ~HypreSmootherWrapper() = default;

    /** \brief Apply operator to a vector. */
    void Mult(const mfem::Vector& rhs, mfem::Vector& sol) const override
    {
        if (this->iterative_mode != smoo_->iterative_mode)
            smoo_->iterative_mode = this->iterative_mode;
        smoo_->Mult(rhs,sol);
    }

    /** \brief Apply operator transpose to a vector. */
    void MultTranspose(
        const mfem::Vector& rhs, mfem::Vector& sol) const override
    {
        if (this->iterative_mode != smoo_->iterative_mode)
            smoo_->iterative_mode = this->iterative_mode;
        smoo_->MultTranspose(rhs,sol);
    }

    /** \name Deleted special functions */
    ///@{
    /** \brief Default construction prohibited. */
    HypreSmootherWrapper() = delete;
    ///@}

private:

    /** \brief Implementation of SetOperator(). */
    void _do_set_operator(const std::shared_ptr<mfem::Operator>& op) override;

private:

    /** \brief The underlying Operator as a HypreParMatrix. */
    std::shared_ptr<mfem::HypreParMatrix> A_;

    /** \brief The mfem::HypreSmoother object. */
    std::shared_ptr<mfem::HypreSmoother> smoo_;

};// class HypreSmootherWrapper
}// namespace parelag
#endif /* PARELAG_HYPRESMOOTHERWRAPPER_HPP_ */
