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

#ifndef PARELAG_VECTORTRAITS_HPP_
#define PARELAG_VECTORTRAITS_HPP_

#include <mfem.hpp>
#include <vector>

namespace parelag
{

template <typename VectorT>
struct VectorTraits
{
    using value_type = typename VectorT::value_type;
    using size_type = typename VectorT::size_type;
};

template <typename T>
struct VectorTraits<mfem::Array<T>>
{
    using value_type = T;
    using size_type = int;
};

template<>
struct VectorTraits<mfem::Vector>
{
    using value_type = double;
    using size_type = int;
};

}// namespace parelag
#endif /* PARELAG_VECTORTRAITS_HPP_ */
