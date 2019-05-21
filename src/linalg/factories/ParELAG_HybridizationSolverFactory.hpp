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


#ifndef PARELAG_HYBRIDIZATIONSOLVERFACTORY_HPP_
#define PARELAG_HYBRIDIZATIONSOLVERFACTORY_HPP_

#include "linalg/solver_core/ParELAG_BlockSolverFactory.hpp"
#include "utilities/MemoryUtils.hpp"
#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"
#include "amge/HybridHdivL2.hpp"
#include "linalg/dense/ParELAG_LDLCalculator.hpp"

namespace parelag
{

/// \class HybridizationSolverFactory
/// Creates the hybridization transformation and returns a solver
/// based on it.
class HybridizationSolverFactory : public BlockSolverFactory
{
public:
    /// \name Constructors and destructor
    //@{

    /// Default constructor with optional parameter list
    HybridizationSolverFactory(const ParameterList& params = ParameterList())
    {
        SetParameters(params);
    }

    /// Constructor taking a solver factory and a parameter list
    HybridizationSolverFactory(std::shared_ptr<SolverFactory> fact,
                         const ParameterList& params = ParameterList())
        : SolverFact_(std::move(fact))
    {
        SetParameters(params);
    }

    //@}

private:
    /// \name SolverFactory interface
    //@{

    /// Build the solver operator, blocked call
    std::unique_ptr<mfem::Solver> _do_build_block_solver(
        const std::shared_ptr<MfemBlockOperator>& op,
        SolverState& state) const override;

    /// Compute rescaling vector by smoothing
    mfem::Vector _get_scaling_by_smoothing(
        const ParallelCSRMatrix& op, int num_iter) const;

    /// Sets any required parameters that have not already been set.
    void _do_set_default_parameters() override;

    /// Create from a parameter list
    void _do_initialize(const ParameterList& Params) override;

    //@}

private:

    /// The solver to be used for solving the transformed system
    std::shared_ptr<SolverFactory> SolverFact_;
};

/// assuming symmetric problems
class AuxiliarySpacePreconditioner : public mfem::Solver
{
public:
    /// dofs are in true dofs numbering, coarse_map: coarse to fine
    AuxiliarySpacePreconditioner(ParallelCSRMatrix& op,
                                 const std::vector<mfem::Array<int> >& local_dofs,
                                 const SerialCSRMatrix& aux_map);

    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const;
    virtual void SetOperator(const mfem::Operator& op) {}

private:
    void Smoothing(const mfem::Vector& x, mfem::Vector& y) const;

    ParallelCSRMatrix& op_;
    mfem::HypreSmoother smoother;
    std::vector<mfem::Array<int> > local_dofs_;
    SerialCSRMatrix aux_map_;
    std::vector<mfem::DenseMatrix> local_ops_;
    std::vector<LDLCalculator> local_solvers_;
    std::unique_ptr<ParallelCSRMatrix> aux_op_;
    std::unique_ptr<mfem::HypreBoomerAMG> aux_solver_;
    mfem::CGSolver aux_cg_;
    std::vector<mfem::SparseMatrix> middle_map;
    std::vector<std::unique_ptr<ParallelCSRMatrix>> middle_op_;
    std::vector<std::unique_ptr<mfem::HypreSmoother>> middle_solver_;
};

class AuxSpaceCG : public mfem::Solver
{
public:
    AuxSpaceCG(std::unique_ptr<ParallelCSRMatrix> op,
               const std::vector<mfem::Array<int> >& local_dofs,
               const SerialCSRMatrix& aux_map);

    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const
    {
        cg_.Mult(x, y);
    }

    virtual void SetOperator(const mfem::Operator& op) {}

private:
    std::unique_ptr<ParallelCSRMatrix> op_;
    AuxiliarySpacePreconditioner prec_;
    mfem::CGSolver cg_;
};

}// namespace parelag
#endif /* PARELAG_HYBRIDIZATIONSOLVERFACTORY_HPP_ */
