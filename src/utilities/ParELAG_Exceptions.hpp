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

#ifndef PARELAG_EXCEPTIONS_HPP_
#define PARELAG_EXCEPTIONS_HPP_

#include <stdexcept>

namespace parelag
{

/// \class bad_var_cast
/// \brief Exception thrown when getting data from variable container fails
struct bad_var_cast : std::runtime_error
{
    template <typename T>
    bad_var_cast(T&& what_arg)
        : std::runtime_error(std::forward<T>(what_arg))
    {}
};

/// \class not_implemented_error
/// \brief Exception thrown when function declared but not fully implemented
struct not_implemented_error : std::logic_error
{
    template <typename T>
    not_implemented_error(T&& what_arg)
        : std::logic_error(std::forward<T>(what_arg))
    {}
};

/// \class hypre_runtime_error
/// \brief Exception thrown when hypre's error code is found to be set
struct hypre_runtime_error : std::runtime_error
{
    template <typename T>
    hypre_runtime_error(T&& what_arg)
        : std::runtime_error(std::forward<T>(what_arg))
    {}
};

}
#endif
