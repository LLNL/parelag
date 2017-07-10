/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#ifndef AUXHYPRESMOOTHER_HPP_
#define AUXHYPRESMOOTHER_HPP_

class AuxHypreSmoother : public Solver
{
public:
	AuxHypreSmoother(const HypreParMatrix &_A, const HypreParMatrix &_C, int type = HypreSmoother::l1GS,
            int relax_times = 1, double relax_weight = 1.0,
            double omega = 1.0, int poly_order = 2,
            double poly_fraction = .3);

	void SetOperator(const Operator & A_);
	void Mult(const Vector & x, Vector & y) const;

	virtual ~AuxHypreSmoother();

private:

	const HypreParMatrix * A;
	const HypreParMatrix * C;

	HypreParMatrix * CtAC;
	HypreSmoother * S;

	mutable Vector X;
	mutable Vector Y;
	mutable Vector res;
};

#endif /* AUXHYPRESMOOTHER_HPP_ */
