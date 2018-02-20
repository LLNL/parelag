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


#ifndef PARELAG_SOLVERSTATE_HPP_
#define PARELAG_SOLVERSTATE_HPP_

#include <unordered_map>
#include <memory>

#include <mfem.hpp>

#include "utilities/elagError.hpp"
#include "utilities/ParELAG_ParameterList.hpp"

namespace parelag
{

class DeRhamSequence;

/** \class SolverState
 *  \brief General class to store extra information for a solver.
 *
 *  The default SolverState stores additional operators that might be
 *  needed, keyed by user-defined strings (just make sure your
 *  SolverFactory is looking for the right strings later on...).
 *
 *  A user can subclass this class to include any additional
 *  information that a solver or smoother might require. For example,
 *  Hiptmair smoothing requires an operator to project from the
 *  primary space to the auxiliary space -- this operator is "extra
 *  information" that one might store in a SolverState object.
 *
 *  The idea is that this is a place to dump (non-POD) data that is
 *  truly characteristic of a solver's state; things like relaxation
 *  parameters for SOR or Vanka should go on the respective parameter
 *  lists.
 *
 *  \todo This class kind of got out of control and too big. It really
 *        should just be a place to dump extra data that persists
 *        and/or updates over time, either simulation time or real
 *        time or both.
 */
class SolverState
{
public:

    /** \brief Virtual destructor. */
    virtual ~SolverState() {}

    /** \brief Set the specified Operator. */
    void SetOperator(const std::string& name,
                     const std::shared_ptr<mfem::Operator>& op)
    { Operators_[name] = op; }

    /** \brief Get the specified Operator. Returns nullptr if "name"
     *         is not a valid key.
     */
    std::shared_ptr<mfem::Operator>
    GetOperator(const std::string& name) const noexcept
    {
        auto result = Operators_.find(name);
        if (result != Operators_.end())
            return result->second;
        return nullptr;
    }

    /** \brief Test if name refers to a known operator. */
    bool IsOperator(const std::string& name) const noexcept
    {
        auto result = Operators_.find(name);
        if (result != Operators_.end())
            return true;
        return false;
    }

    /** \brief Set the specified Vector. */
    void SetVector(const std::string& name,
                   const std::shared_ptr<mfem::Vector>& vec)
    { Vectors_[name] = vec; }

    /** \brief Get the specified Vector. Returns nullptr if "name" is
     *         not a valid key.
     */
    std::shared_ptr<mfem::Vector>
    GetVector(const std::string& name) const noexcept
    {
        auto result = Vectors_.find(name);
        if (result != Vectors_.end())
            return result->second;
        return nullptr;
    }

    /** \brief Test if name refers to a known vector. */
    bool IsVector(const std::string& name) const noexcept
    {
        auto result = Vectors_.find(name);
        if (result != Vectors_.end())
            return true;
        return false;
    }

    /** \brief Set the boundary labels -- stl only version. */
    void SetBoundaryLabels(std::vector<std::vector<int>> labels) noexcept
    {
        BoundaryLabels_ = std::move(labels);
    }

    /** \brief Set the boundary labels -- mfem::Array version. */
    void SetBoundaryLabels(const std::vector<mfem::Array<int>>& labels) noexcept
    {
        using size_type = std::decay<decltype(labels)>::type::size_type;

        BoundaryLabels_.resize(labels.size());
        for (size_type ii = 0; ii < labels.size(); ++ii)
        {
            BoundaryLabels_[ii].resize(labels[ii].Size());
            for (int jj = 0; jj < labels[ii].Size(); ++jj)
                BoundaryLabels_[ii][jj] = labels[ii][jj];
        }
    }

    /** \brief Get the boundary labels. */
    std::vector<std::vector<int>>& GetBoundaryLabels() noexcept
    {
        return BoundaryLabels_;
    }

    /** \brief Get the boundary labels for a given block. */
    std::vector<int>& GetBoundaryLabels(int blockID)
    {
        return BoundaryLabels_.at(blockID);
    }

    /** \brief Set the forms. */
    void SetForms(std::vector<int> Forms)
    {
        Forms_ = std::move(Forms);
    }

    /** \brief Get the forms. */
    std::vector<int>& GetForms() noexcept
    {
        return Forms_;
    }

    /** \brief Set the DeRhamSequence. */
    void SetDeRhamSequence(
        const std::shared_ptr<DeRhamSequence>& sequence) noexcept
    { Sequence_ = sequence; }

    /** \brief Get the specified DeRhamSequence. */
    DeRhamSequence& GetDeRhamSequence()
    { PARELAG_ASSERT(Sequence_); return *Sequence_; }

    std::shared_ptr<DeRhamSequence> GetDeRhamSequencePtr() const noexcept
    { return Sequence_; }

    /** \brief Test if there is a DeRhamSequence attached to the state. */
    bool HasDeRhamSequence() const noexcept
    {
        return (bool)Sequence_;
    }

    /** \brief If *this is missing a field that rhs has, copy from rhs. */
    void MergeState(const SolverState& rhs)
    {
        _do_merge_state(rhs);
    }

    /** \brief If *this is missing a field that rhs has, move from rhs. */
    void MergeState(SolverState&& rhs)
    {
        _do_merge_state(std::move(rhs));
    }

    /** \brief Set extra parameters to the list. */
    template <typename T>
    void SetExtraParameter(const std::string& name,T&& value)
    {
        ExtraParams_.Set(name,std::forward<T>(value));
    }

    /** \brief Get parameter from the list. */
    template <typename T>
    T& GetExtraParameter(const std::string& name,const T& default_value)
    {
        return ExtraParams_.Get(name,default_value);
    }

protected:
    /** \brief A map of strings to operators. */
    std::unordered_map<std::string,std::shared_ptr<mfem::Operator>> Operators_;

    /** \brief A map of strings to extra vectors.
     *  (think: BCs or nonlinear residuals, etc)
     */
    std::unordered_map<std::string,std::shared_ptr<mfem::Vector>> Vectors_;

    /** \brief A map from block ID to essential boundary labels for
     *         that block.
     */
    std::vector<std::vector<int>> BoundaryLabels_;

    /** \brief A map from block ID to the corresponding form ID. */
    std::vector<int> Forms_;

    /** \brief DeRhamSequence for this state object. */
    std::shared_ptr<DeRhamSequence> Sequence_{nullptr};

    /** \brief Parameter List. */
    ParameterList ExtraParams_;

protected:

    virtual void _do_merge_state(const SolverState& rhs)
    {
        if (this->Operators_.size() == 0)
            this->Operators_ = rhs.Operators_;
        if (this->BoundaryLabels_.size() == 0)
            this->BoundaryLabels_ = rhs.BoundaryLabels_;
        if (this->Forms_.size() == 0)
            this->Forms_ = rhs.Forms_;
        if (!this->Sequence_)
            this->Sequence_ = rhs.Sequence_;
        ExtraParams_.Merge(rhs.ExtraParams_);
    }

    virtual void _do_merge_state(SolverState&& rhs)
    {
        if (this->Operators_.size() == 0)
            this->Operators_ = std::move(rhs.Operators_);
        if (this->BoundaryLabels_.size() == 0)
            this->BoundaryLabels_ = std::move(rhs.BoundaryLabels_);
        if (this->Forms_.size() == 0)
            this->Forms_ = std::move(rhs.Forms_);
        if (!this->Sequence_)
            this->Sequence_ = std::move(rhs.Sequence_);
        ExtraParams_.Merge(rhs.ExtraParams_);
    }
};// class SolverState


/** \class NestedSolverState
 *  \brief Solver state for building solvers/smoothers that depend
 *         on other solvers/smoothers.
 *
 *  Examples of these solvers/smoothers include as Hybrid smoothers
 *  (Hiptmair) and block-factorization smoothers. Essentially, one can
 *  store sub-states in this state to use when constructing these
 *  other solvers.
 */
class NestedSolverState : public SolverState
{
public:

    /** \brief Virtual destructor. */
    virtual ~NestedSolverState() {}

    /** \brief Assign a state to a string label. */
    void SetSubState(std::string const& name,
                     std::shared_ptr<SolverState> const& state)
    { States_[name] = state; }

    /** \brief Return the substate with the specified name. */
    std::shared_ptr<SolverState> GetSubState(std::string const& name) noexcept
    {
        auto result = States_.find(name);
        if (result != States_.end())
            return result->second;
        return nullptr;
    }

    /** \brief Test if the given name is the name of a known substate. */
    bool IsSubState(std::string const& name) const noexcept
    {
        auto result = States_.find(name);
        if (result != States_.end())
            return true;
        return false;
    }

protected:

    virtual void _do_merge_state(const SolverState& rhs)
    {
        auto nested_rhs = dynamic_cast<const NestedSolverState*>(&rhs);
        if (nested_rhs && (this->States_.size() == 0))
            this->States_ = nested_rhs->States_;

        SolverState::_do_merge_state(rhs);
    }

    virtual void _do_merge_state(SolverState&& rhs)
    {
        auto nested_rhs = dynamic_cast<NestedSolverState*>(&rhs);
        if (nested_rhs && (this->States_.size() == 0))
            this->States_ = std::move(nested_rhs->States_);

        SolverState::_do_merge_state(std::move(rhs));
    }

protected:

    std::unordered_map<std::string,std::shared_ptr<SolverState>> States_;

};// class NestedSolverState
}// namespace parelag
#endif /* PARELAG_SOLVERSTATE_HPP_ */
