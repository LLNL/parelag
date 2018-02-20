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

#ifndef RANKONEOPERATOR_HPP_
#define RANKONEOPERATOR_HPP_

#include <mfem.hpp>

namespace parelag
{
class RankOneOperator : public mfem::Operator
{
public:
    RankOneOperator(MPI_Comm comm, const mfem::Vector & u, const mfem::Vector & v);
    void Mult(const mfem::Vector & x, mfem::Vector & y) const;
    void MultTranspose(const mfem::Vector & x, mfem::Vector & y) const;
    virtual ~RankOneOperator();

private:

    double dot(const mfem::Vector & a, const mfem::Vector & b) const;

    MPI_Comm comm;
    const mfem::Vector * u;
    const mfem::Vector * v;
};

class RankOnePerturbation : public mfem::Operator
{
public:
    RankOnePerturbation(MPI_Comm comm, const mfem::Operator & A, const mfem::Vector & u, const mfem::Vector & v);
    void Mult(const mfem::Vector & x, mfem::Vector & y) const;
    void MultTranspose(const mfem::Vector & x, mfem::Vector & y) const;
    virtual ~RankOnePerturbation();

private:

    double dot(const mfem::Vector & a, const mfem::Vector & b) const;

    MPI_Comm comm;
    const mfem::Operator * A;
    const mfem::Vector * u;
    const mfem::Vector * v;
};
}//namespace parelag
#endif /* RANKONEOPERATOR_HPP_ */
