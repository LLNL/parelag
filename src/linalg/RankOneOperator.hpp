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

#ifndef RANKONEOPERATOR_HPP_
#define RANKONEOPERATOR_HPP_

class RankOneOperator : public Operator
{
public:
	RankOneOperator(MPI_Comm comm, const Vector & u, const Vector & v);
	void Mult(const Vector & x, Vector & y) const;
	void MultTranspose(const Vector & x, Vector & y) const;
	virtual ~RankOneOperator();

private:

	double dot(const Vector & a, const Vector & b) const;

	MPI_Comm comm;
	const Vector * u;
	const Vector * v;
};

class RankOnePerturbation : public Operator
{
public:
	RankOnePerturbation(MPI_Comm comm, const Operator & A, const Vector & u, const Vector & v);
	void Mult(const Vector & x, Vector & y) const;
	void MultTranspose(const Vector & x, Vector & y) const;
	virtual ~RankOnePerturbation();

private:

	double dot(const Vector & a, const Vector & b) const;

	MPI_Comm comm;
	const Operator * A;
	const Vector * u;
	const Vector * v;
};

#endif /* RANKONEOPERATOR_HPP_ */
