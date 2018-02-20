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

#ifndef HYPRETRAITS_HPP_
#define HYPRETRAITS_HPP_

#include <memory>

// Hypre includes
#include <seq_mv.h>
#include <_hypre_parcsr_mv.h>
#include <_hypre_parcsr_ls.h>

namespace parelag
{

template<typename T>
struct UndefinedHypreTraits
{
    static inline T undefined()
    {
        return T::this_type_is_missing_specialization();
    }
};
    
/// \brief This aids the use of unique_ptr with some fundamental hypre
/// types (namely parallel and serial CSR matrices and vectors).
///
/// The unspecialized struct should fail to create a deleter for its
/// type T and *should* print a reasonably helpful compiler
/// error. Thus, specializations must be written for each type
/// actually used. Currently hypre_CSRMatrix, hypre_ParCSRMatrix,
/// hypre_Vector, and hypre_ParVector are supported.
///
/// \note <ol>
///
/// <li> These only specify a unique_ptr type because the deleter is a
/// template parameter to unique_ptr, but not shared_ptr. Creating a
/// shared_ptr to these objects is trivial since the deleter is the
/// second argument to the constructor and not part of its type.
///
/// <li> You must explicitly construct a unique_ptr_t for these
/// objects as make_unique (both my version and the C++14 standard
/// version) do not recognize deleters in the variadic template.
///
/// </ol>
template<typename T>
struct HypreTraits;

template<>
struct HypreTraits<hypre_CSRMatrix>
{
    using deleter_t = decltype(&hypre_CSRMatrixDestroy);
    using unique_ptr_t = std::unique_ptr<hypre_CSRMatrix, deleter_t>;
};

template<>
struct HypreTraits<hypre_ParCSRMatrix>
{
    using deleter_t = decltype(&hypre_ParCSRMatrixDestroy);
    using unique_ptr_t = std::unique_ptr<hypre_ParCSRMatrix,deleter_t>;
};

template<>
struct HypreTraits<hypre_ParVector>
{
    using deleter_t = decltype(&hypre_ParVectorDestroy);
    using unique_ptr_t = std::unique_ptr<hypre_ParVector,deleter_t>;
};

// I don't think parelag uses hypre_Vector currently, but I figured
// this should exist for completeness.
template<>
struct HypreTraits<hypre_Vector>
{
    using deleter_t = decltype(&hypre_SeqVectorDestroy);
    using unique_ptr_t = std::unique_ptr<hypre_Vector,deleter_t>;
};

}//namespace parelag
#endif

