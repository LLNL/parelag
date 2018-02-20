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

#ifndef PARELAG_HYBRIDIZATIONSOLVER_HPP
#define PARELAG_HYBRIDIZATIONSOLVER_HPP

#include "linalg/solver_core/ParELAG_Solver.hpp"
#include "utilities/MemoryUtils.hpp"
#include "amge/HybridHdivL2.hpp"

namespace parelag
{

/// \class HybridizationSolver
/// Stores stores operators so that a given system may be hybridized
/// and solved.
///
/// The Mult operation will translate the problem into hybridized
/// form, solve the hybridized system, and transform back to the
/// "full" form
class HybridizationSolver : public Solver
{
public:

    /// \name Constructors and destructors
    ///@{

    /// Constructor taking Operators to move
    HybridizationSolver(
        std::shared_ptr<HybridHdivL2> Hybridization,
        std::shared_ptr<mfem::Solver> solver,
        mfem::Array<int>& Offsets,
        std::shared_ptr<mfem::SparseMatrix> D_Scale);

    /// Destructor
    ~HybridizationSolver() = default;

    ///@}
    /// \name Solver interface functions
    ///@{

    /// Set the operator for the solver object
    ///
    /// \warning This will not affect the to/from-hybridized-form
    /// operators. These may be set separately using other functions.
    void _do_set_operator(const std::shared_ptr<mfem::Operator>& op) override;

    /// Transform to the hybrid form, solve the system, and transform back
    ///
    /// \warning This does not support iterative mode!
    void Mult(const mfem::Vector& rhs, mfem::Vector& sol) const override;
    // FIXME: Because we don't store the non-hybridized matrix
    // (because, best I can tell, it's not necessary), we cannot use
    // this solver in iterative mode unless we can use ToHybridOp_ to
    // move the solution to the hybridized form too. Not storing the
    // non-hybridized matrix means we cannot compute the action of
    // this solver as a residual correction scheme.

    /// Transform to the hybrid form, solve the system, and transform back
    ///
    /// \warning This does not support iterative mode!
    void MultTranspose(const mfem::Vector& rhs, mfem::Vector& sol) const override;

    ///@}
    /// \name Extra methods
    ///@{

    /// Set the convert-to-hybridized-form operator
    void SetHybridization(std::shared_ptr<HybridHdivL2> Hybridization)
    {
        PARELAG_ASSERT(Hybridization);
        Hybridization_ = std::move(Hybridization);
    }

    ///@}

private:
    /// Hybridization object containing transformation between
    /// non-hybridized form and hybridized form
    std::shared_ptr<HybridHdivL2> Hybridization_;

    /// The solver for the hybridized system
    std::shared_ptr<mfem::Solver> Solver_;

    /// Auxiliary vectors for solving in the hybridized form
    mutable mfem::Vector pHybridRHS_, pHybridSol_;

    // Offsets of the original block system
    mfem::Array<int> Offsets_;

    std::shared_ptr<mfem::SparseMatrix> D_Scale_;
};

}// namespace parelag
#endif /* PARELAG_HYBRIDIZATIONSOLVER_HPP */
