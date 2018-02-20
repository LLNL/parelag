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


#include "linalg/solver_ops/ParELAG_Hierarchy.hpp"

#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"

#include "utilities/MemoryUtils.hpp"

using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

namespace parelag
{

Hierarchy::Hierarchy(std::shared_ptr<mfem::Operator> A, int NumLevels)
    : Solver{A->NumCols(),A->NumRows(),true},
      Levels_(NumLevels),
      CycleMu_(NumLevels,0),
      CoarseResids_(NumLevels),
      CoarseSols_(NumLevels)
{
    PARELAG_ASSERT_DEBUG(NumLevels > 0);

    for (int ii = 0; ii < NumLevels; ++ii)
        Levels_[ii] = std::make_shared<Level>(ii);

    // Set "A" on fine grid
    Levels_.front()->Set("A",std::move(A));
}


Level& Hierarchy::GetLevel(int index)
{
    PARELAG_TEST_FOR_EXCEPTION(
        !(Levels_.at(index)),
        std::runtime_error,
        "Hierarchy::GetLevel(): Level = " << index << " is null. "
        "No reference to return.");

    return *(Levels_[index]);
}


void Hierarchy::SetCycle(int Mu)
{
    PARELAG_TEST_FOR_EXCEPTION(
        Mu >= 0,
        std::runtime_error,
        "Hierarchy::SetCycle(): Cannot set Mu<0 on every level.");

    // Resize to 1
    CycleMu_.resize(1);

    // Set cycle to Mu
    CycleMu_[0] = Mu;
}


void Hierarchy::SetCycle(int Mu, int LevelID)
{
    // Verify that this is a valid LevelID
    PARELAG_TEST_FOR_EXCEPTION(
        LevelID > static_cast<int>(Levels_.size()),
        std::runtime_error,
        "Hierarchy::SetCycle(): LevelID=" << LevelID <<
        " greater than current number of levels (" <<
        Levels_.size() << ").");

    // If CycleMu_.size() is not the right size, create a new
    // vector based on CycleMu_[0].
    if (CycleMu_.size() != Levels_.size())
    {
        const int CurrentMu = CycleMu_[0];
        std::vector<int>(Levels_.size(),CurrentMu).swap(CycleMu_);
    }

    // Set the right CycleMu_ entry to Mu
    CycleMu_[LevelID] = Mu;
}


void Hierarchy::SetCycle(std::vector<int> Mus)
{
    // Check that Mus has the right size
    PARELAG_TEST_FOR_EXCEPTION(
        Mus.size() != Levels_.size(),
        std::runtime_error,
        "Hierarchy::SetCycle(): Input vector has wrong size (" <<
        Mus.size() << "). Correct size is " << CycleMu_.size() << ".");

    // Set the new values
    CycleMu_.swap(Mus);
}


void Hierarchy::Mult(const mfem::Vector& rhs, mfem::Vector& sol) const
{
    std::unique_ptr<mfem::Vector> rhs_view;
    std::unique_ptr<mfem::Vector> sol_view;

    if (this->IsPreconditioner())
    {
        sol_view = make_unique<mfem::Vector>(sol.GetData(),sol.Size());
        rhs_view = make_unique<mfem::Vector>(rhs.GetData(),rhs.Size());
    }
    else
    {
        rhs_view = make_unique<mfem::Vector>(rhs.Size());
        sol_view = make_unique<mfem::Vector>(sol.Size());

        Levels_.front()->Get<std::shared_ptr<mfem::Operator>>("A")
            ->Mult(sol,*rhs_view);
        *rhs_view *= -1.0;
        *rhs_view += rhs;
    }

    *sol_view = 0.0;

    this->Iterate(*rhs_view,*sol_view,0,1,false);

    if (not this->IsPreconditioner())
        sol += *sol_view;
}


void Hierarchy::Iterate(const mfem::Vector& RHS,
                        mfem::Vector& SOL,
                        int StartLevel,
                        int NumIterations,
                        bool CycleThisLevel) const
{
    using Op_Ptr = shared_ptr<mfem::Operator>;

    for (int iteration = 1; iteration <= NumIterations; ++iteration)
    {
        // Check if we're on the coarsest level
        if (StartLevel == static_cast<int>(Levels_.size())-1)
        {
            // This is the coarse level; Solve and move on.
            Level& Coarse = *Levels_[StartLevel];

            if (Coarse.IsValidKey("CoarseSolver"))
            {
                auto& Solver = Coarse.Get<Op_Ptr>("CoarseSolver");
                Solver->Mult(RHS,SOL);
            }
            else
            {
                if (Coarse.IsValidKey("PreSmoother"))
                {
                    auto& PreSmoo = Coarse.Get<Op_Ptr>("PreSmoother");
                    PreSmoo->Mult(RHS,SOL);
                }
                if (Coarse.IsValidKey("PostSmoother"))
                {
                    auto& PostSmoo = Coarse.Get<Op_Ptr>("PostSmoother");
                    PostSmoo->Mult(RHS,SOL);
                }
            }
        }
        else
        {
            Level& Fine = *Levels_[StartLevel];
            Level& Coarse = *Levels_[StartLevel+1];

            // Pre-smoothing
            if (Fine.IsValidKey("PreSmoother"))
            {
                auto& PreSmoo = Fine.Get<Op_Ptr>("PreSmoother");
                PreSmoo->Mult(RHS,SOL);
            }

            // Compute residual
            tmp_resid_.SetSize(RHS.Size());
            {
                auto& A = Fine.Get<Op_Ptr>("A");

                tmp_correct_.SetSize(tmp_resid_.Size());
                tmp_correct_ = 0.;
                A->Mult(SOL,tmp_correct_);
                add(RHS,-1.0,tmp_correct_,tmp_resid_);
            }

            // Restrict residual

            if ((ImplicitTranspose_) || (!Fine.IsKey("R")))
            {
                auto& P = Coarse.Get<Op_Ptr>("P");
                P->MultTranspose(tmp_resid_,CoarseResids_[StartLevel+1]);
            }
            else
            {
                auto& R = Fine.Get<Op_Ptr>("R");
                R->Mult(tmp_resid_,CoarseResids_[StartLevel+1]);
            }

            // Coarse-grid solve

            CoarseSols_[StartLevel+1] = 0.0;
            int cycle_times = 0;
            if (CycleThisLevel)
            {
                if (CycleMu_.size() > 1)
                    cycle_times = CycleMu_[StartLevel];
                else
                    cycle_times = 0;
            }

            // Recursive call here
            for (int cycle = 0; cycle <= cycle_times; ++cycle)
            {
                Iterate(CoarseResids_[StartLevel+1],
                        CoarseSols_[StartLevel+1],StartLevel+1,1,false);
            }

            // Interpolate error correction

            auto& P = Coarse.Get<Op_Ptr>("P");
            tmp_correct_.SetSize(P->Height());

            P->Mult(CoarseSols_[StartLevel+1],tmp_correct_);

            // Add correction

            SOL += tmp_correct_;

            // Post-smoothing

            // If PostSmoother is a valid key, it is either null or
            // non-null. If the latter, use it.
            if (Fine.IsValidKey("PostSmoother"))
            {
                // If a PostSmoother exists and is not null, use it.
                auto& PostSmoo = Fine.Get<Op_Ptr>("PostSmoother");
                PostSmoo->Mult(RHS,SOL);
            }// Post-smoothing

        }// Fine level iteration
    }// For NumIterations
}// Iterate

void Hierarchy::Finalize()
{
    using Op_Ptr = std::shared_ptr<mfem::Operator>;

    auto lbegin = Levels_.cbegin();
    auto rbegin = CoarseResids_.begin(),
        sbegin = CoarseSols_.begin();
    for ( ; lbegin != Levels_.end(); ++lbegin, ++rbegin, ++sbegin)
    {
        rbegin->SetSize((*lbegin)->Get<Op_Ptr>("A")->Height());
        sbegin->SetSize((*lbegin)->Get<Op_Ptr>("A")->Width());
    }
}


void Hierarchy::_do_set_operator(std::shared_ptr<mfem::Operator> const& op)
{
    PARELAG_ASSERT(op->NumCols() == this->Height());
    PARELAG_ASSERT(op->NumRows() == this->Width());

    Levels_.front()->Set("A",op);
}


#ifndef DOXYGEN_SHOULD_SKIP_THIS


unique_ptr<Hierarchy> buildHierarchyFromDeRhamSequence(
    const shared_ptr<mfem::Operator>& A_in,
    const DeRhamSequence& Sequence,
    std::vector<int>& label_ess,
    const int form,
    const int MaxNumLevels=-1)
{
    using Op_Ptr = shared_ptr<mfem::Operator>;

    auto A = A_in;

    // Default to the specified max levels
    int NumLevels = MaxNumLevels;

    // Compute the number of levels that can be in the hierarchy based
    // on the hierarchy of DeRhamSequence objects.
    {
        int NumLevels_actual = 0;
        const DeRhamSequence* tmpPtr = &Sequence;
        while (tmpPtr)
        {
            tmpPtr = tmpPtr->ViewCoarserSequence();
            ++NumLevels_actual;
        }
        if ((MaxNumLevels < 1) || (MaxNumLevels > NumLevels_actual))
            NumLevels = NumLevels_actual;
    }

    // Create the hierarchy with the right number of levels
    auto myHierarchy = make_unique<Hierarchy>(A,NumLevels);
    myHierarchy->SetImplicitTranspose(true);

    // Get a pointer to the current sequence.
    const DeRhamSequence* SequencePtr = &Sequence;
    for (const auto& level : *myHierarchy)
    {
        // Get this level's ID and skip if it's the finest level
        Level& Coarse = *level;
        auto levelID = Coarse.GetLevelID();
        if (levelID == 0) continue;

        // Get P from the level (Remember, DeRhamSequence stores the P
        // that goes from CoarserSequence to ThisSeqence,
        // i.e. Coarse->Fine).
        //
        // Note: We use shared_ptr here to ensure a uniformity
        // across the interface. unique_ptr is actually
        // appropriate for this case, but if we were building a serial
        // version of this hierarchy, with "P" as stored by
        // DeRhamSequence, we would need to share ownership of that P
        // with the sequence. For this reason, "Iterate()" calls
        // "Get<shared_ptr<mfem::Operator>>".
        shared_ptr<mfem::HypreParMatrix> P;

        if (label_ess.size() > 0)
        {
            mfem::Array<int> tmp(label_ess.data(),label_ess.size());
            P = SequencePtr->ComputeTrueP(form,tmp);
        }
        else
            P = SequencePtr->ComputeTrueP(form);

        PARELAG_TEST_FOR_EXCEPTION(
            not (A->Width() == P->Height()),
            std::runtime_error,
            "buildHierarchyFromDeRhamSequence(...): "
            "A and P do not have compatible sizes to compute P^T*A*P!\n"
            "A = " << A->Height() << "x" << A->Width() << "\n"
            "P = " << P->Height() << "x" << P->Width() << "\n"
            "Perhaps you have passed the wrong form (form = " << form << ") "
            "for your operator?\n");

        // Reset A to be Ac=P^t*A*P
        auto A_hypre = dynamic_cast<mfem::HypreParMatrix *>(A.get());

        // Check that the input matrix is actually a HypreParMatrix
        if (levelID == 1)
            PARELAG_TEST_FOR_EXCEPTION(
                !A_hypre,
                std::runtime_error,
                "buildHierarchyFromDeRhamSequence(...): "
                "Fine-grid A is not a HypreParMatrix.");

        A.reset(mfem::RAP(A_hypre,P.get()));
        if (label_ess.size() > 0)
        {
            A_hypre = dynamic_cast<mfem::HypreParMatrix *>(A.get());
            PARELAG_ASSERT_HYPRE_ERROR_FLAG(
                hypre_ParCSRMatrixFixZeroRows(*A_hypre));
        }

        // Set the coarse P and Ac to the Hierarchy
        Coarse.Set("P",Op_Ptr{std::move(P)});
        Coarse.Set("A",Op_Ptr{A});

        // Move on to the next sequence
        SequencePtr = SequencePtr->ViewCoarserSequence();
    }
    myHierarchy->Finalize();

    return myHierarchy;
}


// Note: I explicitly require that A be block square (NxN blocks) and
// implicitly assume that the diagonal blocks of A are square
// matrices.
//
// FIXME: I need to do something for cases (like Stokes) in which
// certain forms have boundary conditions and others don't!
//
// Assumptions:
//
//   * label_ess.size() == forms.size()
//   * label_ess[i].size() > 0 --> this form has BC's
//   * A_in casts to shared_ptr<BlockOperator>
//   * A_in has forms.size() blocks
//   * all forms in 'forms' have been built by Sequence
unique_ptr<Hierarchy> buildBlockedHierarchyFromDeRhamSequence(
    const shared_ptr<mfem::Operator>& A_in,
    const DeRhamSequence& Sequence,
    std::vector<std::vector<int>>& label_ess,
    const std::vector<int>& forms,
    int MaxNumLevels)
{
    using Op_Ptr = shared_ptr<mfem::Operator>;

    auto A = A_in;

    // Verify that we're using a blocked operator
    auto A_blocked = std::dynamic_pointer_cast<MfemBlockOperator>(A);
    PARELAG_TEST_FOR_EXCEPTION(
        !A_blocked,
        std::runtime_error,
        "buildBlockedHierarchyFromDeRhamSequence(...): "
        "A is not a MfemBlockOperator.");

    // Get the block dimensions
    const int NumBlockRows = A_blocked->GetNumBlockRows();
    const int NumBlockCols = A_blocked->GetNumBlockCols();

    PARELAG_TEST_FOR_EXCEPTION(
        NumBlockRows != NumBlockCols,
        std::runtime_error,
        "buildBlockedHierarchyFromDeRhamSequence(...): "
        "A is not block-square!");

    PARELAG_TEST_FOR_EXCEPTION(
        static_cast<int>(forms.size()) != NumBlockRows,
        std::runtime_error,
        "buildBlockedHierarchyFromDeRhamSequence(...): "
        "The forms vector has the wrong size (" << forms.size() <<
        "). Should be " << NumBlockRows << ".");

    PARELAG_TEST_FOR_EXCEPTION(
        forms.size() != label_ess.size(),
        std::runtime_error,
        "buildBlockedHierarchyFromDeRhamSequence(...): "
        "Forms vector and boundary condition information are inconsistent.")

    // Default to the specified max levels
    int NumLevels = MaxNumLevels;

    // Compute the number of levels that can be in the hierarchy based
    // on the hierarchy of DeRhamSequence objects.
    {
        int NumLevels_actual = 0;
        const DeRhamSequence* tmpPtr = &Sequence;
        while (tmpPtr)
        {
            tmpPtr = tmpPtr->ViewCoarserSequence();
            ++NumLevels_actual;
        }
        if ((MaxNumLevels < 1) || (MaxNumLevels > NumLevels_actual))
            NumLevels = NumLevels_actual;
    }

    // Create the hierarchy with the right number of levels
    auto myHierarchy = make_unique<Hierarchy>(A,NumLevels);
    myHierarchy->SetImplicitTranspose(true);

    // Get a pointer to the current sequence.
    const DeRhamSequence* SequencePtr = &Sequence;
    for (auto& level : *myHierarchy)
    {
        // Get this level's ID and skip if it's the finest level
        Level& Coarse = *level;
        auto levelID = Coarse.GetLevelID();
        if (levelID == 0) continue;

        // Grab coarse sequence too
        const DeRhamSequence* CoarseSequencePtr
            = SequencePtr->ViewCoarserSequence();

        // Figure out the right sizes for the blocked operators
        std::vector<int> CoarseOffSets(NumBlockRows+1);

        CoarseOffSets[0] = 0;
        for (int ii = 0; ii < NumBlockRows; ++ii)
            CoarseOffSets[ii+1] = CoarseOffSets[ii] +
                CoarseSequencePtr->GetNumberOfTrueDofs(forms[ii]);

        // Create P and Ac at the same time
        auto Ac = make_shared<MfemBlockOperator>(CoarseOffSets);
        auto P = make_shared<MfemBlockOperator>(
            A_blocked->CopyRowOffsets(), Ac->CopyRowOffsets());

        // Get the P blocks
        std::vector<unique_ptr<mfem::HypreParMatrix>> P_blocks(NumBlockRows);
        for (int ii = 0; ii < NumBlockRows; ++ii)
            if (label_ess[ii].size() > 0)
            {
                mfem::Array<int> tmp(label_ess[ii].data(),label_ess[ii].size());
                P_blocks[ii] = SequencePtr->ComputeTrueP(forms[ii],tmp);
            }
            else
                P_blocks[ii] = SequencePtr->ComputeTrueP(forms[ii]);

        // Do the RAP
        for (int row = 0; row < NumBlockRows; ++row)
        {
            for (int col = 0; col < NumBlockCols; ++col)
            {
                if (not A_blocked->IsZeroBlock(row,col))
                {
                    mfem::HypreParMatrix& Aij_Hypre =
                        dynamic_cast<mfem::HypreParMatrix&>(
                            A_blocked->GetBlock(row,col));

                    auto tmp = shared_ptr<mfem::HypreParMatrix>{
                        mfem::RAP(P_blocks[row].get(),
                                  &Aij_Hypre, P_blocks[col].get())};

                    if ((row == col) && (label_ess[row].size() > 0))
                    {
                        PARELAG_ASSERT_HYPRE_ERROR_FLAG(
                            hypre_ParCSRMatrixFixZeroRows(*tmp));
                    }

                    Ac->SetBlock(row,col,std::move(tmp));
                }
            }
        }

        // Set the block-diagonal elements of P
        for (int row = 0; row < NumBlockRows; ++row)
            P->SetBlock(row,row,std::move(P_blocks[row]));

        // Reset A to be Ac=P^t*A*P
        A = Ac;
        A_blocked = std::dynamic_pointer_cast<MfemBlockOperator>(A);

        // Set the coarse P and Ac to the Hierarchy
        Coarse.Set("P",Op_Ptr{std::move(P)});
        Coarse.Set("A",Op_Ptr{std::move(A)});

        // Move on to the next sequence
        SequencePtr = CoarseSequencePtr;
    }
    myHierarchy->Finalize();

    return myHierarchy;
}

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
}// namespace parelag
