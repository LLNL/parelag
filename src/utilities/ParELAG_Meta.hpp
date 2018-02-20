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


#ifndef PARELAG_META_HPP_
#define PARELAG_META_HPP_

#include <stdexcept>
#include <type_traits>

namespace parelag
{

/** \name Cast operators */
///@

/** \brief A safe narrowing cast.
 *
 *  Checks that the input value can safely be converted back to its
 *  original value after narrowing.
 *
 *  \tparam T The type to which \c x is being converted.
 *  \tparam U The (deduced) type of \c x.
 *
 *  \param x The value to cast.
 *
 *  \throws std::runtime_error Thrown if x cannot be cast to T without
 *                             narrowing the value.
 *
 *  \return A copy of x with the type narrowed.
 */
template <typename T, typename U>
T narrowing_cast(U const& x)
{
    T as_t = static_cast<T>(x);

    if (static_cast<U>(as_t) != x)
        throw std::runtime_error(
            "narrowing_cast: input value cannot be safely narrowed.");

    return as_t;
}

/** \name Convenience typedefs */
///@{

/// Enables function if B=true and gives type T; ignores otherwise
template <bool B, typename T=void>
using EnableIf = typename std::enable_if<B,T>::type;

/// Selects type T if B=true, type U if B=false
template <bool B, typename T, typename U>
using Conditional = typename std::conditional<B,T,U>::type;

template <typename T>
using Decay = typename std::decay<T>::type;

///@}
/** \name Type predicates */
///@{

/// Predicate to test if a clase is constructible with a given set of
/// argument types
template <typename T, typename... Args>
constexpr bool IsConstructible()
{
    return std::is_constructible<T,Args...>::value;
}

/// Predicate to test a class has a default ctor
template <typename T>
constexpr bool IsDefaultConstructible()
{
    return std::is_default_constructible<T>::value;
}

/// Predicate to test a class has a default ctor
template <typename T>
constexpr bool IsNoThrowDefaultConstructible()
{
    return std::is_nothrow_default_constructible<T>::value;
}

/// Predicate to test if T is the base clase of U
template <typename T, typename U>
constexpr bool IsBaseOf()
{
    return std::is_base_of<T,U>::value;
}

/// Predicate to test if T and U are the same type
template <typename T, typename U>
constexpr bool IsSame()
{
    return std::is_same<T,U>::value;
}

/// Predicate to test if T has const qualifiers
template <typename T>
constexpr bool IsConst()
{
    return std::is_const<T>::value;
}


///@}

}// namespace parelag

#endif /* PARELAG_META_HPP_ */
