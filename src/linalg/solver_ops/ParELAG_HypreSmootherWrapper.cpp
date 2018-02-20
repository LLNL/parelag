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


#include "ParELAG_HypreSmootherWrapper.hpp"

namespace parelag
{

HypreSmootherWrapper::HypreSmootherWrapper(
    const std::shared_ptr<mfem::Operator>& op, int type, ParameterList& params)
    : Solver{op->NumCols(),op->NumRows(),true}
{
    A_ = std::dynamic_pointer_cast<mfem::HypreParMatrix>(op);

    PARELAG_ASSERT(A_);

    smoo_ = make_unique<mfem::HypreSmoother>(
        *A_, type, params.Get<int>("Sweeps",1),
        params.Get<double>("Damping Factor",1.0),
        params.Get<double>("Omega",1.0),
        params.Get<int>("Cheby Poly Order",2),
        params.Get<double>("Cheby Poly Fraction",0.3));
    smoo_->iterative_mode = true;
}


void HypreSmootherWrapper::_do_set_operator(
    const std::shared_ptr<mfem::Operator>& op)
{
    auto tmp = std::dynamic_pointer_cast<mfem::HypreParMatrix>(op);

    PARELAG_ASSERT(tmp);

    A_ = tmp;

    smoo_->SetOperator(*op);
}

}// namespace parelag
