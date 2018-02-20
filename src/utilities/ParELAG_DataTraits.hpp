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

#ifndef PARELAG_DATATRAITS_HPP_
#define PARELAG_DATATRAITS_HPP_

#include <complex>

namespace parelag
{

template <typename T> struct is_complex_type : std::false_type {};

template <typename T>
struct is_complex_type<std::complex<T>> : std::true_type {};

template <typename value_t>
struct DataTraits {};

template <>
struct DataTraits<int>
{
    static constexpr int zero() { return 0; }
    static constexpr int one() { return 1; }
};

template <>
struct DataTraits<unsigned>
{
    static constexpr unsigned zero() { return 0u; }
    static constexpr unsigned one() { return 1u; }
};

template <>
struct DataTraits<float>
{
    using real_type = float;

    static constexpr float zero() { return 0.0f; }
    static constexpr float one() { return 1.0f; }
};

template <>
struct DataTraits<double>
{
    using real_type = double;

    static constexpr double zero() { return 0.0; }
    static constexpr double one() { return 1.0; }
};

template <>
struct DataTraits<std::complex<float>>
{
    using real_type = float;

    static constexpr std::complex<float> zero()
    {
        return std::complex<float>{0.0f,0.0f};
    }
    static constexpr std::complex<float> one()
    {
        return std::complex<float>{1.0f,0.0f};
    }
};

template <>
struct DataTraits<std::complex<double>>
{
    using real_type = double;

    static constexpr std::complex<double> zero()
    {
        return std::complex<double>{0.0f,0.0f};
    }
    static constexpr std::complex<double> one()
    {
        return std::complex<double>{1.0f,0.0f};
    }
};

}// namespace parelag
#endif /* PARELAG_DATATRAITS_HPP_ */
