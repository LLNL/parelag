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

        mfem::Array<int> Offsets(3);
    	Offsets[0] = 0;
    	Offsets[1] = sequence.GetNumberOfDofs(forms[0]);
    	Offsets[2] = Offsets[1] + sequence.GetNumberOfDofs(forms[1]);

        // Copy the hybridized matrix and eliminate the boundary condition
        // for the matrix. Note that at this stage no rhs is given, so the
        // original hybridized matrix (before elimination) is kept so that
        // later it can be used to finish the elimination process (for rhs)
        mfem::SparseMatrix HB_mat_copy(*hybridization->GetHybridSystem());
        DofHandler* dofhandler = hybridization->GetDofMultiplier();
        SharingMap& mult_dofTrueDof = dofhandler->GetDofTrueDof();

        // Eliminate essential multiplier dofs (1-1 map to natural Hdiv dofs)
        for (int i = 0; i < HB_mat_copy.Size(); i++)
            if (hybridization->GetEssentialMultiplierDofs()[i])
                HB_mat_copy.EliminateRowCol(i);
        auto pHB_mat = Assemble(mult_dofTrueDof, HB_mat_copy, mult_dofTrueDof);

        const int rescale_iter = state.GetExtraParameter("RescaleIteration", -20);
        auto scaling_vector = rescale_iter < 0 ? hybridization->GetRescaling() :
                              _get_scaling_by_smoothing(*pHB_mat, rescale_iter);

        int* scale_i = new int[pHB_mat->Height()+1];
        int* scale_j = new int[pHB_mat->Height()];
        double* scale_data = new double[pHB_mat->Height()];
        std::iota(scale_i, scale_i+pHB_mat->Height()+1, 0);
        std::iota(scale_j, scale_j+pHB_mat->Height(), 0);
        std::copy_n(scaling_vector.GetData(), pHB_mat->Height(), scale_data);
        auto D_Scale = make_shared<mfem::SparseMatrix>(
                    scale_i, scale_j, scale_data,
                    pHB_mat->Height(), pHB_mat->Height());

        const auto facet = AgglomeratedTopology::FACET;
        auto& facet_truefacet = sequence.GetTopology()->EntityTrueEntity(facet);
        const int num_facets = facet_truefacet.GetLocalSize();

        std::shared_ptr<mfem::Solver> solver;

        if (!IsSameOrient || (num_facets == dofhandler->GetNDofs()))
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

            const int num_truefacets = facet_truefacet.GetTrueLocalSize();
            std::vector<mfem::Array<int>> local_dofs(num_truefacets);

            auto facet_dof = dofhandler->GetEntityDofTable(facet);

            mfem::SparseMatrix diag;
            facet_truefacet.get_entity_trueEntity()->GetDiag(diag);
            mfem::SparseMatrix f_tf_diag(diag); f_tf_diag = 1.0;
            mult_dofTrueDof.get_entity_trueEntity()->GetDiag(diag);
            mfem::SparseMatrix d_td_diag(diag); d_td_diag = 1.0;
            std::unique_ptr<mfem::SparseMatrix> tf_td(RAP(f_tf_diag, facet_dof, d_td_diag));

            mfem::SparseMatrix PV_map(d_td_diag.NumCols(), num_truefacets);
            for (int i = 0; i < num_truefacets; ++i)
            {
                local_dofs[i].MakeRef(tf_td->GetRowColumns(i), tf_td->RowSize(i));
                PV_map.Add(local_dofs[i][0], i, 1.0);
            }
            PV_map.Finalize();
            PV_map.ScaleRows(scaling_vector);

            solver = make_unique<AuxSpaceCG>(std::move(pHB_mat), local_dofs, PV_map);
            *D_Scale = 1.0;
        }

        solver->iterative_mode = false;

        std::unique_ptr<mfem::Solver> hybrid_solve =
                make_unique<HybridizationSolver>(std::move(hybridization),
                                                 std::move(solver), Offsets,
                                                 std::move(D_Scale));

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

AuxiliarySpacePreconditioner::AuxiliarySpacePreconditioner(
        ParallelCSRMatrix& op,
        const std::vector<mfem::Array<int> >& local_dofs,
        const SerialCSRMatrix& aux_map)
    : mfem::Solver(op.NumRows(), false),
      op_(op),
      smoother(op_, mfem::HypreSmoother::l1GS, 1),
      local_dofs_(local_dofs.size()),
      aux_map_(aux_map),
      local_ops_(local_dofs.size()),
      local_solvers_(local_dofs.size()),
      aux_cg_(op_.GetComm()),
      middle_map(local_dofs[0].Size()-2),
      middle_op_(local_dofs[0].Size()-2),
      middle_solver_(local_dofs[0].Size()-2)
{
    // Set up local solvers
//    SerialCSRMatrix op_diag;
//    op.GetDiag(op_diag);
    for (int i = 0; i < local_dofs.size(); ++i)
    {
//        local_ops_[i].SetSize(local_dofs[i].Size());
//        op_diag.GetSubMatrix(local_dofs[i], local_dofs[i], local_ops_[i]);
//        local_solvers_[i].Compute(local_ops_[i]);
        local_dofs[i].Copy(local_dofs_[i]);
    }

    for (int k = 0; k < local_dofs[0].Size() - 2; ++k)
    {
        mfem::SparseMatrix coarsen_map(op_.NumRows(), local_dofs.size()*(k+2));

        for (int i = 0; i < local_dofs.size(); ++i)
        {
            for (int j = 0; j < k + 2; ++j)
            {
                coarsen_map.Add(local_dofs[i][j], i*(k+2)+j, 1.0);
            }
        }
        coarsen_map.Finalize();
        middle_map[k].Swap(coarsen_map);

        int num_local_adofs = middle_map[k].NumCols();
        mfem::Array<int> adof_starts;
        ParPartialSums_AssumedPartitionCheck(op.GetComm(), num_local_adofs, adof_starts);
        int num_global_adofs = adof_starts.Last();
        ParallelCSRMatrix parallel_aux_map(op.GetComm(), op.N(), num_global_adofs,
                                           op.ColPart(), adof_starts, &(middle_map[k]));

        middle_op_[k].reset(RAP(&op, &parallel_aux_map));
        middle_op_[k]->CopyRowStarts();
        middle_op_[k]->CopyColStarts();

        middle_solver_[k] = make_unique<mfem::HypreSmoother>(*middle_op_[k]);
    }

    // Set up auxilary space solver
    int num_local_adofs = aux_map.NumCols();
    mfem::Array<int> adof_starts;
    ParPartialSums_AssumedPartitionCheck(op.GetComm(), num_local_adofs, adof_starts);
    int num_global_adofs = adof_starts.Last();
    ParallelCSRMatrix parallel_aux_map(op.GetComm(), op.N(), num_global_adofs,
                                       op.ColPart(), adof_starts, &aux_map_);

    aux_op_.reset(RAP(&op, &parallel_aux_map));
    aux_op_->CopyRowStarts();
    aux_op_->CopyColStarts();

    aux_solver_ = make_unique<mfem::HypreBoomerAMG>(*aux_op_);
    aux_solver_->SetPrintLevel(-1);

    aux_cg_.SetRelTol(1e-1);
    aux_cg_.SetAbsTol(1e-10);
    aux_cg_.SetMaxIter(50);
//    aux_cg_.SetPrintLevel(1);
    aux_cg_.SetOperator(*aux_op_);
    aux_cg_.SetPreconditioner(*aux_solver_);

}

void AuxiliarySpacePreconditioner::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    y = 0.0;

    mfem::Vector x_aux, y_aux, residual(x), correction(y.Size());

    Smoothing(x, y);
    op_.Mult(-1.0, y, 1.0, residual);



    for (int k = local_dofs_[0].Size() - 3; k > -1 ; --k)
    {
        x_aux.SetSize(middle_map[k].NumCols());
        y_aux.SetSize(middle_map[k].NumCols());
        x_aux = 0.0;
        y_aux = 0.0;
        middle_map[k].MultTranspose(residual, x_aux);
        middle_solver_[k]->Mult(x_aux, y_aux);
        correction = 0.0;
        middle_map[k].Mult(y_aux, correction);

        op_.Mult(-1.0, correction, 1.0, residual);
        y += correction;
    }



    x_aux.SetSize(aux_map_.NumCols());
    y_aux.SetSize(aux_map_.NumCols());
    x_aux = 0.0;
    y_aux = 0.0;
    aux_map_.MultTranspose(residual, x_aux);
    aux_cg_.Mult(x_aux, y_aux);
    correction = 0.0;
    aux_map_.Mult(y_aux, correction);

    op_.Mult(-1.0, correction, 1.0, residual);
    y += correction;



    for (int k = 0; k < local_dofs_[0].Size() - 2; ++k)
    {
        x_aux.SetSize(middle_map[k].NumCols());
        y_aux.SetSize(middle_map[k].NumCols());
        x_aux = 0.0;
        y_aux = 0.0;
        middle_map[k].MultTranspose(residual, x_aux);
        middle_solver_[k]->Mult(x_aux, y_aux);
        correction = 0.0;
        middle_map[k].Mult(y_aux, correction);

        op_.Mult(-1.0, correction, 1.0, residual);
        y += correction;
    }



    Smoothing(residual, correction);
    y += correction;
}

void AuxiliarySpacePreconditioner::Smoothing(const mfem::Vector& x, mfem::Vector& y) const
{
    y = 0.0;
    smoother.Mult(x, y);

//    mfem::Vector x_local, y_local, residual(x);

//    auto loop_content = [&](int i)
//    {
//        residual.GetSubVector(local_dofs_[i], x_local);
//        y_local.SetSize(x_local.Size());
//        local_solvers_[i].Mult(x_local, y_local);
//        y.AddElementVector(local_dofs_[i], y_local);

//        residual = x;
//        op_.Mult(-1.0, y, 1.0, residual);
//    };

//    for (int i = 0; i < local_dofs_.size(); ++i)
//    {
//        loop_content(i);
//    }

//    for (int i = local_dofs_.size()-1; i > -1; --i)
//    {
//        loop_content(i);
//    }
}

AuxSpaceCG::AuxSpaceCG(std::unique_ptr<ParallelCSRMatrix> op,
                       const std::vector<mfem::Array<int> >& local_dofs,
                       const SerialCSRMatrix& aux_map)
    : mfem::Solver(op->NumRows(), false),
      op_(std::move(op)),
      prec_(*op_, local_dofs, aux_map),
      cg_(op_->GetComm())
{
    cg_.SetPrintLevel(1);
    cg_.SetMaxIter(500);
    cg_.SetRelTol(1e-9);
    cg_.SetAbsTol(1e-12);
    cg_.SetOperator(*op_);
    cg_.SetPreconditioner(prec_);
    cg_.iterative_mode = false;
}

}// namespace parelag
