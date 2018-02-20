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


#ifndef PARELAG_STRUMPACKSOLVER_HPP_
#define PARELAG_STRUMPACKSOLVER_HPP_

#include <memory>

#include <mfem.hpp>
#include <StrumpackSparseSolver.hpp>
#include <StrumpackSparseSolverMPIDist.hpp>

#include "ParELAG_Config.h"
#include "linalg/solver_core/ParELAG_DirectSolver.hpp"
#include "utilities/ParELAG_ParameterList.hpp"

namespace parelag
{

/** \class StrumpackSolver
 *  \brief This class manages a direct solve with Strumpack.
 *
 *  \warning This code may be **VERY** out of date. I do not often
 *           compile or use ParELAG with the Strumpack support
 *           enabled.
 */
template <typename Scalar=double, typename Real=double, typename Ordinal=int>
class StrumpackSolver : public DirectSolver
{
    using solver_type = typename
        strumpack::StrumpackSparseSolverMPIDist<Scalar,Real,Ordinal>;
    using scalar_type = Scalar;
    using real_type = Real;
    using ordinal_type = Ordinal;

public:
    /** \name Constructors and destructor */
    ///@{

    /** \brief Default constructor. */
    StrumpackSolver() = default;

    /** \brief Construct from an operator. */
    StrumpackSolver(const std::shared_ptr<mfem::Operator>& A);

    /** \brief Destructor. */
    ~StrumpackSolver() = default;

    ///@}
    /** \name The mfem::Solver interface */
    ///@{

    /** \brief Solves the system A*X = B.
     *
     *  \note Contents of X are overwritten.
     *
     *  \warning This will call Factor() if the matrix hasn't been factored.
     *
     *  \param[in]     B  The system right-hand side
     *  \param[in,out] X  The system initial guess and output solution.
     */
    void Mult(const mfem::Vector& B, mfem::Vector& X) const override;

    ///@}

private:

    /** \name Special things for this class */
    ///@{

    /** \brief Computes the LU decomposition of A.
     *
     *  \warning This method essentially forces a refactor. No
     *           checking is done as to whether the matrix is already
     *           factored. As factoring is quite expensive, this
     *           should only be called once.
     */
    void _do_factor() override;

    ///@}

    /** \name The parelag::Solver interface */
    ///@{

    /** \brief Handle the setting of the operator.
     *
     *  \note This resets the solver's state. If the previous operator was
     *        factored, a refactorization will be forced (this requires a
     *        separate call to Factor()).
     */
    void _do_set_operator(const std::shared_ptr<mfem::Operator>& A) override;

    void _do_set_solver_parameters(const ParameterList& pl) override;

    ///@}

private:

    /** \brief The MPI communicator for the matrix. */
    MPI_Comm Comm_ = MPI_COMM_NULL;

    /** \brief The underlying matrix operator. */
    std::shared_ptr<mfem::Operator> A_;

    /** \brief The underlying Strumpack object. */
    std::unique_ptr<solver_type> Solver_;

    /** \brief Whether the matrix factorization is complete. */
    mutable bool IsFactored_ = false;

};// class StrumpackSolver
}// namespace parelag
#endif /* PARELAG_STRUMPACKSOLVER_HPP_ */
