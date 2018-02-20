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

#ifndef GHOSTMAP_HPP_
#define GHOSTMAP_HPP_

#include <memory>

#include <mfem.hpp>

#include "structures/SharingMap.hpp"

namespace parelag
{

//! @class GhostMap
/**
   @brief based on SharingMap, this class is able to communicate data
   between processors needed in e.g. Discontinuous Galerkin.

   An example of usage, where the Vector f is the number of coarse
   faces long:

   GhostMap gm(face_trueFace);
   Vector true_sum_f(face_trueFace.GetTrueLocalSize());
   Vector true_diff_f(face_trueFace.GetTrueLocalSize());
   gm.AssemblePlus(f1,true_sum_f);
   gm.AssembleMinus(f1,true_diff_f);

   Vector sum_f(nAF);
   Vector diff_f(nAF);

   gm.DistributePlus(true_sum_f, sum_f);
   gm.DistributeMinus(true_diff_f, diff_f);

   // Upwinding:
   double f_star = 0.5*(un*(sum_f(face))+std::fabs(un)*(diff_f(face)));

   Here f is upwinded based on the direction of u*n
*/
class GhostMap
{
public:
    GhostMap(const SharingMap & in);
    ~GhostMap();

    int AssemblePlus(const mfem::Vector & data, mfem::Vector & trueData);
    int AssembleMinus(const mfem::Vector & data, mfem::Vector & trueData);

    int DistributePlus(const mfem::Vector & trueData, mfem::Vector & data);
    int DistributeMinus(const mfem::Vector & trueData, mfem::Vector & data);

private:

    std::unique_ptr<SharingMap> Map_;

    int assemble(const mfem::Vector & data,
                 double offd_values,
                 mfem::Vector & trueData);
    int distribute(const mfem::Vector & trueData,
                   double offd_values,
                   mfem::Vector & data);
};
}//namespace parelag
#endif /*GHOSTMAP_HPP_ */
