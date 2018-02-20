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


#ifndef PARELAG_SUPERLU_HELPERS_HPP_
#define PARELAG_SUPERLU_HELPERS_HPP_

#include <memory>

// Basically, this is all the metaprogramming/scaffolding code that
// makes the SuperLU interface work seamlessly for the four supported
// types: single, double, single complex, double complex.

// I use a TypeTraits class basically to handle the Dtype and to map
// std::complex into SLU::SCX::complex and
// SLU::DCX::doublecomplex. The SLU_Caller template should be
// instantiated using, e.g.,
//
//   SLU_Caller<SLU_TypeTraits<Scalar>::type> slu_caller_name;
//
namespace parelag
{


/** \namespace parelag::SLU
 *  \brief SuperLU symbols.
 *
 *  This keeps the SuperLU functions out of the main parelag namespace
 *  directly. This is necessary in case multiple flavors of SuperLU
 *  are built.
 */
namespace SLU
{

using int_t = int;
#undef __SUPERLU_SUPERMATRIX
#undef __SUPERLU_ENUM_CONSTS
#undef __SUPERLU_UTILS
#include <supermatrix.h>
#include <slu_util.h>

/** \namespace parelag::SLU::SGL
 *  \brief Enable single-precision real support for SuperLU.
 */
namespace SGL
{
#include <slu_sdefs.h>
}//namespace SGL

/** \namespace parelag::SLU::DBL
 *  \brief Enable double-precision real support for SuperLU.
 */
namespace DBL
{
#include <slu_ddefs.h>
}//namespace DBL

#ifdef PARELAG_ENABLE_COMPLEX

/** \namespace parelag::SLU::SCX
 *  \brief Enable single-precision complex support for SuperLU.
 */
namespace SCX
{
#include <slu_cdefs.h>
}//namespace SCX

/** \namespace parelag::SLU::DCX
 *  \brief Enable double-precision complex support for SuperLU.
 */
namespace DCX
{
#include <slu_zdefs.h>
}//namespace DCX

#endif /* PARELAG_ENABLE_COMPLEX */

}//namespace SLU


/** \struct SLU_TypeTraits
 *  \brief Type traits for the 4 types supported by SuperLU.
 */
template <typename T>
struct SLU_TypeTraits;


/** \struct SLU_Caller
 *  \brief Handles the type resolution for calls to SuperLU.
 */
template <typename T>
struct SLU_Caller;


template <>
struct SLU_TypeTraits<float>
{
    static constexpr SLU::Dtype_t Dtype = SLU::SLU_S;
    using type = float;
};

template <>
struct SLU_Caller<float>
{
    template <typename... Ts>
    static void gstrf(Ts&&... Args)
    { parelag::SLU::SGL::sgstrf(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gstrs(Ts&&... Args)
    { parelag::SLU::SGL::sgstrs(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gssv(Ts&&... Args)
    { parelag::SLU::SGL::sgssv(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gssvx(Ts&&... Args)
    { parelag::SLU::SGL::sgssvx(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_CompCol_Matrix(Ts&&... Args)
    { parelag::SLU::SGL::sCreate_CompCol_Matrix(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_CompRow_Matrix(Ts&&... Args)
    { parelag::SLU::SGL::sCreate_CompRow_Matrix(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_Dense_Matrix(Ts&&... Args)
    { parelag::SLU::SGL::sCreate_Dense_Matrix(std::forward<Ts>(Args)...); }
};


template <>
struct SLU_TypeTraits<double>
{
    static constexpr SLU::Dtype_t Dtype = SLU::SLU_D;
    using type = double;
};

template <>
struct SLU_Caller<double>
{
    template <typename... Ts>
    static void gstrf(Ts&&... Args)
    { parelag::SLU::DBL::dgstrf(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gstrs(Ts&&... Args)
    { parelag::SLU::DBL::dgstrs(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gssv(Ts&&... Args)
    { parelag::SLU::DBL::dgssv(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gssvx(Ts&&... Args)
    { parelag::SLU::DBL::dgssvx(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_CompCol_Matrix(Ts&&... Args)
    { parelag::SLU::DBL::dCreate_CompCol_Matrix(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_CompRow_Matrix(Ts&&... Args)
    { parelag::SLU::DBL::dCreate_CompRow_Matrix(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_Dense_Matrix(Ts&&... Args)
    { parelag::SLU::DBL::dCreate_Dense_Matrix(std::forward<Ts>(Args)...); }
};


#ifdef PARELAG_ENABLE_COMPLEX


template <>
struct SLU_TypeTraits<std::complex<float>>
{
    static constexpr SLU::Dtype_t Dtype = SLU::SLU_C;
    using type = SLU::SCX::complex;
};

template <>
struct SLU_TypeTraits<SLU::SCX::complex>
{
    static constexpr SLU::Dtype_t Dtype = SLU::SLU_C;
    using type = SLU::SCX::complex;
};

template <>
struct SLU_Caller<SLU::SCX::complex>
{
    template <typename... Ts>
    static void gstrf(Ts&&... Args)
    { parelag::SLU::SCX::cgstrf(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gstrs(Ts&&... Args)
    { parelag::SLU::SCX::cgstrs(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gssv(Ts&&... Args)
    { parelag::SLU::SCX::cgssv(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gssvx(Ts&&... Args)
    { parelag::SLU::SCX::cgssvx(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_CompCol_Matrix(Ts&&... Args)
    { parelag::SLU::SCX::cCreate_CompCol_Matrix(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_CompRow_Matrix(Ts&&... Args)
    { parelag::SLU::SCX::cCreate_CompRow_Matrix(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_Dense_Matrix(Ts&&... Args)
    { parelag::SLU::SCX::cCreate_Dense_Matrix(std::forward<Ts>(Args)...); }
};


template <>
struct SLU_TypeTraits<std::complex<double>>
{
    static constexpr SLU::Dtype_t Dtype = SLU::SLU_Z;
    using type = SLU::DCX::doublecomplex;
};

template <>
struct SLU_TypeTraits<SLU::DCX::doublecomplex>
{
    static constexpr SLU::Dtype_t Dtype = SLU::SLU_Z;
    using type = SLU::DCX::doublecomplex;
};

template <>
struct SLU_Caller<SLU::DCX::doublecomplex>
{
    template <typename... Ts>
    static void gstrf(Ts&&... Args)
    { parelag::SLU::DCX::zgstrf(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gstrs(Ts&&... Args)
    { parelag::SLU::DCX::zgstrs(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gssv(Ts&&... Args)
    { parelag::SLU::DCX::zgssv(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void gssvx(Ts&&... Args)
    { parelag::SLU::DCX::zgssvx(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_CompCol_Matrix(Ts&&... Args)
    { parelag::SLU::DCX::zCreate_CompCol_Matrix(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_CompRow_Matrix(Ts&&... Args)
    { parelag::SLU::DCX::zCreate_CompRow_Matrix(std::forward<Ts>(Args)...); }

    template <typename... Ts>
    static void Create_Dense_Matrix(Ts&&... Args)
    { parelag::SLU::DCX::zCreate_Dense_Matrix(std::forward<Ts>(Args)...); }
};
#endif /* PARELAG_ENABLE_COMPLEX */

}// namespace parelag
#endif /* PARELAG_SUPERLU_HELPERS_HPP_ */
