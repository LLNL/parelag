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

#ifndef PARELAG_MEMORYUTILS_HPP_
#define PARELAG_MEMORYUTILS_HPP_

#include <memory>

namespace parelag
{

#if __cplusplus > 201103L
using std::make_unique;
#else
/// make_shared is part of the C++11 standard; make_unique did not
/// make it in until the C++14 standard. This implements the
/// single-object version that appears in that standard (i.e. the
/// return type is std::unique_ptr<T>); the array version is not
/// supported; however, it is not difficult to add if needed.
template<typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&&... params)
{
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}
#endif

/// Short-hand for creating a unique_ptr from a raw pointer object.
/// This is useful for dealing with MFEM objects.
template<typename T>
std::unique_ptr<T> ToUnique(T* raw_ptr)
{
    return std::unique_ptr<T>{raw_ptr};
}

// This is a purposefully undefined templated class used for
// compile-time debugging of auto-deduced types. PLEASE: do not
// remove it just because it doesn't appear in committed code --
// such code would deliberately fail to compile and thus would
// (hopefully) not be committed. But this is a useful tool to have
// lying around and it should never actually cause object code to
// be written. Thanks!
//
// If you want to know the deduced type of something, your
// compiler can help you. Try:
//
// =====
//   auto myvar = SomeFunction(Args...);
//
//   parelag::TypeChecker<decltype(myvar)> myvartype;
// =====
//
// Your compiler will not have a definition for this class,
// obviously, and it will print the type of 'myvar' in its error
// message output. It may print a mangled name; if this is the
// case, run the output through 'c++filt' to demangle it.
template<typename T>
class TypeChecker;

}// namespace parelag

#endif
