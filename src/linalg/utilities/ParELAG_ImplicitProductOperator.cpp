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


#include "ParELAG_ImplicitProductOperator.hpp"

#include <memory>

namespace parelag
{

void ImplicitProductOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    PARELAG_ASSERT(Ops_.size());
    PARELAG_ASSERT(Ops_.size() == Vecs_.size() + 1);
    PARELAG_ASSERT(x.Size() == width);
    PARELAG_ASSERT(y.Size() == height);

    auto vec_iter = Vecs_.begin();
    const mfem::Vector * tmpL = std::addressof(x);
    mfem::Vector * tmpR;

    for (auto& op : Ops_)
    {
        if (vec_iter == Vecs_.end())
            tmpR = std::addressof(y);
        else
            tmpR = std::addressof(*vec_iter);

        // Figure out if we need to start with a zero RHS
        auto solver_ptr = dynamic_cast<mfem::Solver *>(op.get());
        if (solver_ptr)
            if (solver_ptr->iterative_mode)
                *tmpR = 0.0;

        op->Mult(*tmpL,*tmpR);

        tmpL = tmpR;
        ++vec_iter;
    }
}


void ImplicitProductOperator::MultTranspose(
    const mfem::Vector& x, mfem::Vector& y) const
{
    PARELAG_ASSERT(Ops_.size());
    PARELAG_ASSERT(Ops_.size() == Vecs_.size() + 1);
    PARELAG_ASSERT(x.Size() == height);
    PARELAG_ASSERT(y.Size() == width);

    auto vec_iter = Vecs_.rbegin();
    const mfem::Vector * tmpL = &x;
    mfem::Vector * tmpR;

    for (auto op_it = Ops_.rbegin(); op_it != Ops_.rend(); ++op_it)
    {
        if (vec_iter == Vecs_.rend())
            tmpR = std::addressof(y);
        else
            tmpR = std::addressof(*vec_iter);

        // Figure out if we need to start with a zero RHS
        auto solver_ptr = dynamic_cast<mfem::Solver *>(op_it->get());
        if (solver_ptr)
            if (solver_ptr->iterative_mode)
                *tmpR = 0.0;

        (*op_it)->MultTranspose(*tmpL,*tmpR);

        tmpL = tmpR;
        ++vec_iter;
    }

}


ImplicitProductOperator& ImplicitProductOperator::PreMultiply(
    const std::shared_ptr<mfem::Operator>& op)
{
    Ops_.push_back(op);
    height = op->Height();
    if (Ops_.size() > 1)
        Vecs_.emplace_back(op->Width());
    return *this;
}


ImplicitProductOperator& ImplicitProductOperator::PostMultiply(
    const std::shared_ptr<mfem::Operator>& op)
{
    Ops_.push_front(op);
    width = op->Width();
    if (Ops_.size() > 1)
        Vecs_.emplace_front(op->Height());
    return *this;
}


}// namespace parelag
