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

#ifndef AUXHYPRESMOOTHER_HPP_
#define AUXHYPRESMOOTHER_HPP_

#include <mfem.hpp>

namespace parelag
{
class AuxHypreSmoother : public mfem::Solver
{
public:
    AuxHypreSmoother(const mfem::HypreParMatrix &_A, const mfem::HypreParMatrix &_C, int type = mfem::HypreSmoother::l1GS,
                     int relax_times = 1, double relax_weight = 1.0,
                     double omega = 1.0, int poly_order = 2,
                     double poly_fraction = .3);

    void SetOperator(const mfem::Operator & A_);
    void Mult(const mfem::Vector & x, mfem::Vector & y) const;

    virtual ~AuxHypreSmoother();

private:

    const mfem::HypreParMatrix * A;
    const mfem::HypreParMatrix * C;

    mfem::HypreParMatrix * CtAC;
    mfem::HypreSmoother * S;

    mutable mfem::Vector X;
    mutable mfem::Vector Y;
    mutable mfem::Vector res;
};
}
#endif /* AUXHYPRESMOOTHER_HPP_ */
