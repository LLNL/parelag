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


#include "ParELAG_HybridizationSolverFactory.hpp"

#include "linalg/solver_ops/ParELAG_HybridizationSolver.hpp"
#include "linalg/solver_core/ParELAG_SolverLibrary.hpp"
#include "amge/DeRhamSequence.hpp"
#include "amge/HybridHdivL2.hpp"
#include "structures/SharingMap.hpp"

#include <numeric>

using std::make_shared;

namespace parelag
{

std::unique_ptr<mfem::Solver> HybridizationSolverFactory::_do_build_block_solver(
    const std::shared_ptr<MfemBlockOperator>& op, SolverState& state) const
{
    PARELAG_ASSERT(op);

    // Build
    auto& params = GetParameters();
    const auto hybrid_strategy =
        params.Get<std::string>("Hybridization strategy");

    if (hybrid_strategy == "Darcy")
    {
        auto& sequence = state.GetDeRhamSequence();
        auto sequence_ptr = state.GetDeRhamSequencePtr();
        auto forms = state.GetForms();

        // Whether the element H(div) dofs have same orientation on shared facet
        bool IsSameOrient = state.GetExtraParameter("IsSameOrient",false);

        // The constant weight in the system [M B^T; B -(W_weight*W)]
        double L2MassWeight = state.GetExtraParameter("L2MassWeight",0.0);

        bool act_on_trueDofs = state.GetExtraParameter("ActOnTrueDofs",false);

        // Scales of element H(div) mass matrices for the problem being solved
        // If not provided, the scale is treated as 1.0
        auto elemMatrixScaling = state.GetVector("elemMatrixScaling");

        auto label_ess = state.GetBoundaryLabels(0);
        mfem::Array<int> ess_HdivDofs;
        if (label_ess.size() > 0)
        {
            mfem::Array<int> ess_attr(label_ess.data(),label_ess.size());
            ess_HdivDofs.SetSize(sequence.GetNumberOfDofs(forms[0]));
            sequence.GetDofHandler(forms[0])->MarkDofsOnSelectedBndr(
                 ess_attr, ess_HdivDofs);
        }
        else
        {
            ess_HdivDofs.SetSize(sequence.GetNumberOfDofs(forms[0]));
            ess_HdivDofs = 0;
        }
        std::shared_ptr<HybridHdivL2> hybridization =
                std::make_shared<HybridHdivL2>(sequence_ptr, IsSameOrient,
                                               L2MassWeight, ess_HdivDofs,
                                               elemMatrixScaling);

        // Copy the hybridized matrix and eliminate the boundary condition
        // for the matrix. Note that at this stage no rhs is given, so the
        // original hybridized matrix (before elimination) is kept so that
        // later it can be used to finish the elimination process (for rhs)

        mfem::SparseMatrix HB_mat_copy(hybridization->GetHybridSystem());
        auto& dofhandler = const_cast<DofHandler&>(hybridization->GetDofMultiplier());
        const SharingMap& mult_dofTrueDof = dofhandler.GetDofTrueDof();

        // Eliminate essential multiplier dofs (1-1 map to natural Hdiv dofs)
        for (int i = 0; i < HB_mat_copy.Size(); i++)
            if (hybridization->GetEssentialMultiplierDofs()[i])
                HB_mat_copy.EliminateRowCol(i);
        auto pHB_mat = Assemble(mult_dofTrueDof, HB_mat_copy, mult_dofTrueDof);

        const int rescale_iter = state.GetExtraParameter("RescaleIteration", -20);
        const mfem::Vector& precomputed_scale = hybridization->GetRescaling();
        const bool use_precomputed_scaling = precomputed_scale.Size() && rescale_iter < 0;
        auto scaling_vector = use_precomputed_scaling ? hybridization->GetRescaling() :
                              _get_scaling_by_smoothing(*pHB_mat, std::abs(rescale_iter));

        // Smoothing renders scaling of essential dofs to be close to 0
        if (use_precomputed_scaling == false)
        {
            auto d_td = mult_dofTrueDof.get_entity_trueEntity();
            mfem::SparseMatrix diag;
            d_td->GetDiag(diag);

            for (int i = 0; i < HB_mat_copy.Size(); i++)
            {
                if (hybridization->GetEssentialMultiplierDofs()[i])
                {
                    if (diag.RowSize(i))
                    {
                        PARELAG_ASSERT(diag.RowSize(i) == 1);
                        scaling_vector(diag.GetRowColumns(i)[0]) = 1.0;
                    }
                }
            }
        }

        int* scale_i = new int[pHB_mat->Height()+1];
        int* scale_j = new int[pHB_mat->Height()];
        double* scale_data = new double[pHB_mat->Height()];
        std::iota(scale_i, scale_i+pHB_mat->Height()+1, 0);
        std::iota(scale_j, scale_j+pHB_mat->Height(), 0);
        std::copy_n(scaling_vector.GetData(), pHB_mat->Height(), scale_data);
        auto D_Scale = make_shared<mfem::SparseMatrix>(
                    scale_i, scale_j, scale_data,
                    pHB_mat->Height(), pHB_mat->Height());

        auto facet = AgglomeratedTopology::FACET;
        auto& facet_truefacet = sequence.GetTopology()->EntityTrueEntity(facet);
        int num_facets = facet_truefacet.GetLocalSize();

        std::shared_ptr<mfem::Solver> solver;

        if (!IsSameOrient || (facet_truefacet.GetGlobalSize() == pHB_mat->N()))
        {
            auto pD_Scale = make_unique<mfem::HypreParMatrix>(
                        pHB_mat->GetComm(), pHB_mat->N(), pHB_mat->RowPart(),
                        D_Scale.get());

            auto Scaled_pHB = ToUnique(RAP(pHB_mat.get(), pD_Scale.get()));

            auto hybrid_state = std::shared_ptr<SolverState>{
                    SolverFact_->GetDefaultState()};
            solver = SolverFact_->BuildSolver(std::move(Scaled_pHB),*hybrid_state);
        }
        else
        {
            auto& d_td = dofhandler.GetDofTrueDof();

            mfem::SparseMatrix loc_PV_map(d_td.GetLocalSize(), num_facets);
            mfem::Array<int> dof_help;
            for (int i = 0; i < num_facets; ++i)
            {
                dofhandler.GetDofs(facet, i, dof_help);
                loc_PV_map.Add(dof_help.Min(), i, 1.0);
            }
            loc_PV_map.Finalize();

            auto PV_map = IgnoreNonLocalRange(d_td, loc_PV_map, facet_truefacet);
            PV_map->ScaleRows(scaling_vector);

            auto& solver_params = SolverFact_->GetParameters();;
            auto prec = make_unique<AuxSpacePrec>(*pHB_mat, std::move(PV_map));
            solver = make_unique<PCG>(std::move(pHB_mat), std::move(prec), solver_params);

            *D_Scale = 1.0;
        }

        solver->iterative_mode = false;

        std::unique_ptr<mfem::Solver> hybrid_solve =
                make_unique<HybridizationSolver>(std::move(hybridization),
                                                 std::move(solver),
                                                 std::move(sequence_ptr),
                                                 std::move(D_Scale),
                                                 act_on_trueDofs);

        return hybrid_solve;
    }
    else
    {
        const bool invalid_hybridization_strategy = true;

        PARELAG_TEST_FOR_EXCEPTION(
            invalid_hybridization_strategy,
            std::runtime_error,
            "HybridizationSolverFactory::BuildBlockSolver(...):\n"
            "Hybridization strategy \"" << hybrid_strategy <<
            "\" is invalid. Currently, the only option is \"Darcy\"");
    }

}

mfem::Vector HybridizationSolverFactory::_get_scaling_by_smoothing(
     const ParallelCSRMatrix& op, int num_iter) const
{
    mfem::Vector scaling_vector(op.Height());
    scaling_vector = 1.0;

    if (num_iter > 0)
    {
        // Generate a diagonal scaling matrix by smoothing some random vector
        mfem::HypreSmoother smoother(const_cast<ParallelCSRMatrix&>(op));
        mfem::SLISolver sli(op.GetComm());
        sli.SetOperator(op);
        sli.SetPreconditioner(smoother);
        sli.SetMaxIter(num_iter);

        mfem::Vector zeros(op.Height());
        zeros = 0.0;
        sli.Mult(zeros, scaling_vector);
    }

    return scaling_vector;
}

void HybridizationSolverFactory::_do_set_default_parameters()
{
    auto& params = GetParameters();

    // Options: "Assemble then transform" or "transform then assemble"
    params.Get<std::string>("Hybridization strategy","Darcy");

    // May be any solver known to the library
    params.Get<std::string>("Solver","Invalid");
}

void HybridizationSolverFactory::_do_initialize(const ParameterList&)
{
    // Depends on a library
    PARELAG_ASSERT(HasValidSolverLibrary());

    const std::string solver_name =
        GetParameters().Get<std::string>("Solver");

    SolverFact_ = GetSolverLibrary().GetSolverFactory(solver_name);
}

AuxSpacePrec::AuxSpacePrec(
        ParallelCSRMatrix& op,
        std::unique_ptr<ParallelCSRMatrix> aux_map)
    : Solver(op.NumRows(), false),
      op_(op),
      aux_map_(std::move(aux_map))
{
    // Set up solver on the auxiliary space
    aux_op_.reset(RAP(&op, aux_map_.get()));
    aux_op_->CopyRowStarts();
    aux_op_->CopyColStarts();
    aux_solver_ = make_unique<mfem::HypreBoomerAMG>(*aux_op_);
    aux_solver_->SetPrintLevel(-1);

    smoother_ = make_unique<mfem::HypreSmoother>(op_);
}

void AuxSpacePrec::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    y = 0.0;
    smoother_->Mult(x, y);

    mfem::Vector residual(x), correction(y.Size());
    op_.Mult(-1.0, y, 1.0, residual);

    mfem::Vector x_aux(aux_map_->NumCols());
    mfem::Vector y_aux(aux_map_->NumCols());
    y_aux = 0.0;
    aux_map_->MultTranspose(residual, x_aux);
    aux_solver_->Mult(x_aux, y_aux);
    aux_map_->Mult(y_aux, correction);

    op_.Mult(-1.0, correction, 1.0, residual);
    y += correction;

    correction = 0.0;
    smoother_->Mult(residual, correction);
    y += correction;
}

pMultigrid::pMultigrid(
        ParallelCSRMatrix& op,
        const std::vector<mfem::Array<int> >& local_dofs,
        const mfem::Vector& scaling)
    : mfem::Solver(op.NumRows(), false),
      level(0),
      ops_(local_dofs[0].Size()),
      solvers_(local_dofs[0].Size())
{
    ops_[0].reset(mfem::Add(1.0, op, 0.0, op));
    ops_[0]->CopyRowStarts();
    ops_[0]->CopyColStarts();

    Ps_.reserve(local_dofs[0].Size()-1);

    int fine_size = ops_[0]->NumRows();

    mfem::Vector scaling_fine(scaling), scaling_coarse;
    for (int k = 1; k < local_dofs[0].Size(); ++k)
    {
        const int loc_fine_sz = local_dofs[0].Size() - k + 1;
        const int loc_coarse_sz = local_dofs[0].Size() - k;
        int coarse_size = local_dofs.size() * loc_coarse_sz;

        Ps_.emplace_back(fine_size, coarse_size);
        if (k == 0)
        {
            for (int i = 0; i < local_dofs.size(); ++i)
                for (int j = 0; j < loc_coarse_sz; ++j)
                    Ps_[k-1].Add(local_dofs[i][j], i*loc_coarse_sz+j, 1.0);
        }
        else
        {
            for (int i = 0; i < local_dofs.size(); ++i)
                for (int j = 0; j < loc_coarse_sz; ++j)
                    Ps_[k-1].Add(i*loc_fine_sz+j, i*loc_coarse_sz+j, 1.0);
        }
        Ps_[k-1].Finalize();

        if (k < local_dofs[0].Size()-1)
        {
            scaling_coarse.SetSize(coarse_size);
            Ps_[k-1].MultTranspose(scaling_fine, scaling_coarse);
        }
        else
        {
            Ps_[k-1].ScaleRows(scaling_fine);
        }

        mfem::Array<int> coarse_starts;
        ParPartialSums_AssumedPartitionCheck(op.GetComm(), coarse_size, coarse_starts);
        int global_coarse_size = coarse_starts.Last();
        ParallelCSRMatrix parallel_P(op.GetComm(), ops_[k-1]->N(), global_coarse_size,
                                     ops_[k-1]->ColPart(), coarse_starts, &(Ps_[k-1]));

        ops_[k].reset(RAP(ops_[k-1].get(), &parallel_P));
        ops_[k]->CopyRowStarts();
        ops_[k]->CopyColStarts();

        solvers_[k-1] = make_unique<mfem::HypreSmoother>(*ops_[k-1]);

        fine_size = coarse_size;
        scaling_fine = scaling_coarse;
    }

    {
        auto coarse_solver = new mfem::CGSolver(op.GetComm());
        coarse_solver->SetRelTol(1e-1);
        coarse_solver->SetAbsTol(1e-10);
        coarse_solver->SetMaxIter(10);
//        coarse_solver->SetPrintLevel(1);
        coarse_solver->SetOperator(*(ops_.back()));

        coarse_prec_ = make_unique<mfem::HypreBoomerAMG>(*(ops_.back()));
        coarse_prec_->SetPrintLevel(-1);
        coarse_solver->SetPreconditioner(*coarse_prec_);
        coarse_solver->iterative_mode = false;

        solvers_.back().reset(coarse_solver);
    }
}

void pMultigrid::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    mfem::Vector residual(x), correction(y.Size());

    solvers_[level]->Mult(residual, y);
    ops_[level]->Mult(-1.0, y, 1.0, residual);

    mfem::Vector x_c(Ps_[level].NumCols()), y_c(Ps_[level].NumCols());
    y_c = 0.0;

    Ps_[level].MultTranspose(residual, x_c);
    if (level == solvers_.size()-2)
    {
        solvers_.back()->Mult(x_c, y_c);
    }
    else
    {
        level++;
        Mult(x_c, y_c);
        level--;
    }
    Ps_[level].Mult(y_c, correction);

    ops_[level]->Mult(-1.0, correction, 1.0, residual);
    y += correction;

    correction = 0.0;
    solvers_[level]->Mult(residual, correction);
    y += correction;
}

PCG::PCG(std::unique_ptr<ParallelCSRMatrix> op,
         std::unique_ptr<mfem::Solver> prec,
         ParameterList& params)
    : mfem::Solver(op->NumRows(), false),
      op_(std::move(op)),
      prec_(std::move(prec)),
      cg_(op_->GetComm())
{
    int print_level = params.Get("Print level", 1);
    int max_iter = params.Get("Maximum iterations", 5000);
    double rel_tol = params.Get("Relative tolerance", 1e-6);
    double abs_tol = params.Get("Absolute tolerance", 1e-8);

    cg_.SetPrintLevel(print_level);
    cg_.SetMaxIter(max_iter);
    cg_.SetRelTol(rel_tol);
    cg_.SetAbsTol(abs_tol);
    cg_.SetOperator(*op_);
    cg_.SetPreconditioner(*prec_);
    cg_.iterative_mode = false;
}

}// namespace parelag
