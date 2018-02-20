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


#ifndef PARELAG_SUPERLUDIST_HELPERS_HPP_
#define PARELAG_SUPERLUDIST_HELPERS_HPP_

#include <memory>

// Basically, this is all the metaprogramming/scaffolding code that
// makes SuperLUDist interface work seamlessly for the supported
// types: double, double complex.

// I use a TypeTraits class basically to handle the Dtype and to map
// std::complex into SLU::DCX::doublecomplex. The SLUDIST_Caller
// template should be instantiated using, e.g.,
//
//   SLUDIST_Caller<SLUDIST_TypeTraits<Scalar>::type> slu_caller_name;
//
namespace parelag
{

/** \namespace parelag::SLUDIST
 *  \brief SuperLUDist symbols.
 *
 *  This keeps the SuperLUDist functions out of the main parelag
 *  namespace directly. This is necessary in case multiple flavors of
 *  SuperLU are built.
 */
namespace SLUDIST
{
// Need to make sure we can include the Dist version, even if we have
// included the sequential version (note the different namespace, so
// this is ok).
#undef __SUPERLU_SUPERMATRIX
#undef __SUPERLU_ENUM_CONSTS
#undef __SUPERLU_UTIL
#undef CHECK_MALLOC
#undef NO_MEMTYPE
#include <util_dist.h>
#include <superlu_defs.h>

/** \namespace parelag::SLUDIST::DBL
 *  \brief Enable double-precision real support for SuperLUDist.
 */
namespace DBL
{
#include <superlu_ddefs.h>
}//namespace DBL

#ifdef PARELAG_ENABLE_COMPLEX
/** \namespace parelag::SLUDIST::DCX
 *  \brief Enable double-precision complex support for SuperLUDist.
 */
namespace DCX
{
#include <superlu_zdefs.h>
}//namespace DCX
#endif /* PARELAG_ENABLE_COMPLEX */

}//namespace SLUDIST


/** \struct SLUDIST_TypeTraits
 *  \brief Type traits for the 2 types supported by SuperLUDist.
 */
template <typename T>
struct SLUDIST_TypeTraits;

/** \struct SLUDIST_Caller
 *  \brief Handles the type resolution for calls to SuperLUDist.
 */
template <typename T>
struct SLUDIST_Caller;

template <>
struct SLUDIST_TypeTraits<double>
{
    static constexpr SLUDIST::Dtype_t Dtype = SLUDIST::SLU_D;
    using type = double;
    using LUstruct_t = SLUDIST::DBL::LUstruct_t;
    using SOLVEstruct_t = SLUDIST::DBL::SOLVEstruct_t;
};

template <>
struct SLUDIST_Caller<double>
{
    template <typename... Ts>
    static double plangs(Ts&&... Args)
    {return SLUDIST::DBL::pdlangs(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static SLUDIST::int_t gstrf(Ts&&... Args)
    {return SLUDIST::DBL::pdgstrf(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void gstrs(Ts&&... Args)
    {SLUDIST::DBL::pdgstrs(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void gstrs_Bglobal(Ts&&... Args)
    {SLUDIST::DBL::pdgstrs_Bglobal(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void gssvx(Ts&&... Args)
    {SLUDIST::DBL::pdgssvx(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static float dist_psymbtonum(Ts&&... Args)
    {return SLUDIST::DBL::ddist_psymbtonum(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void Create_CompCol_Matrix(Ts&&... Args)
    {SLUDIST::DBL::dCreate_CompCol_Matrix_dist(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void Create_CompRowLoc_Matrix(Ts&&... Args)
    {SLUDIST::DBL::dCreate_CompRowLoc_Matrix_dist(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void Create_Dense_Matrix(Ts&&... Args)
    {SLUDIST::DBL::dCreate_Dense_Matrix_dist(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void CompRow_loc_to_CompCol_global(Ts&&... Args)
    {SLUDIST::DBL::pdCompRow_loc_to_CompCol_global(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void Permute_Dense_Matrix(Ts&&... Args)
    {SLUDIST::DBL::pdPermute_Dense_Matrix(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void LUstructInit(Ts&&... Args)
    {SLUDIST::DBL::LUstructInit(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void LUstructFree(Ts&&... Args)
    {SLUDIST::DBL::LUstructFree(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void Destroy_LU(Ts&&... Args)
    {SLUDIST::DBL::Destroy_LU(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void SolveInit(Ts&&... Args)
    {SLUDIST::DBL::dSolveInit(std::forward<Ts>(Args)...);}

    template <typename... Ts>
    static void SolveFinalize(Ts&&... Args)
    {SLUDIST::DBL::dSolveFinalize(std::forward<Ts>(Args)...);}
};

#ifdef PARELAG_ENABLE_COMPLEX
template <>
struct SLUDIST_TypeTraits<std::complex<double>>
{
    static constexpr SLUDIST::Dtype_t Dtype = SLUDIST::SLU_Z;
    using type = SLUDIST::DCX::doublecomplex;
    using LUstruct_t = SLUDIST::DCX::LUstruct_t;
    using SOLVEstruct_t = SLUDIST::DCX::SOLVEstruct_t;
};

template <>
struct SLUDIST_TypeTraits<SLUDIST::DCX::doublecomplex>
{
    static constexpr SLUDIST::Dtype_t Dtype = SLUDIST::SLU_Z;
    using type = SLUDIST::DCX::doublecomplex;
    using LUstruct_t = SLUDIST::DCX::LUstruct_t;
    using SOLVEstruct_t = SLUDIST::DCX::SOLVEstruct_t;
};

template <>
struct SLUDIST_Caller<SLUDIST::DCX::doublecomplex>
{
    template <typename... Ts>
    static void gstrf(Ts&&... Args)
    { parelag::SLUDIST::DCX::zgstrf(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gstrs(Ts&&... Args)
    { parelag::SLUDIST::DCX::zgstrs(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gssv(Ts&&... Args)
    { parelag::SLUDIST::DCX::zgssv(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gssvx(Ts&&... Args)
    { parelag::SLUDIST::DCX::zgssvx(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_CompCol_Matrix_dist(Ts&&... Args)
    { parelag::SLUDIST::DCX::zCreate_CompCol_Matrix_dist(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_CompRow_Matrix_dist(Ts&&... Args)
    { parelag::SLUDIST::DCX::zCreate_CompRow_Matrix_dist(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_Dense_Matrix_dist(Ts&&... Args)
    { parelag::SLUDIST::DCX::zCreate_Dense_Matrix_dist(std::forward<Ts>(Args)...); }
};
#endif /* PARELAG_ENABLE_COMPLEX */

}// namespace parelag
#endif /* PARELAG_SUPERLUDIST_HELPERS_HPP_ */
