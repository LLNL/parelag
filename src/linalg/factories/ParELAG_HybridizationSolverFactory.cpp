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
        auto hybridization =
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
        SharingMap& mult_dofTrueDof = hybridization->
                GetDofMultiplier()->GetDofTrueDof();

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

        auto pD_Scale = make_unique<mfem::HypreParMatrix>(
                    pHB_mat->GetComm(), pHB_mat->N(), pHB_mat->RowPart(),
                    D_Scale.get());

        auto Scaled_pHB = ToUnique(RAP(pHB_mat.get(), pD_Scale.get()));

        auto hybrid_state = std::shared_ptr<SolverState>{
            SolverFact_->GetDefaultState()};
        auto solver = SolverFact_->BuildSolver(std::move(Scaled_pHB),*hybrid_state);
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

}// namespace parelag
