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

#include "Coefficient.hpp"

namespace parelag
{
using namespace mfem;

void fillCoefficientArray(int nDim, int polyorder,
                          Array<Coefficient *> & coeffs)
{
    // FIXME (trb 12/23/15): Why does this return and the others throw?
    if(polyorder < 0)
    {
        coeffs.SetSize(0);
        return;
    }

    switch(nDim)
    {
    case 2:
        fill2DCoefficientArray(polyorder, coeffs);
        break;
    case 3:
        fill3DCoefficientArray(polyorder, coeffs);
        break;
    default:
        PARELAG_TEST_FOR_EXCEPTION(
            true,
            std::invalid_argument,
            "fillCoefficientArray: Invalid nDim.");
    }
}


void fillVectorCoefficientArray(int nDim, int polyorder,
                                Array<VectorCoefficient *> & coeffs)
{
    // FIXME (trb 12/23/15): Why does this return and the others throw?
    if(polyorder < 0)
    {
        coeffs.SetSize(0);
        return;
    }

    switch(nDim)
    {
    case 2:
        fill2DVectorCoefficientArray(polyorder, coeffs);
        break;
    case 3:
        fill3DVectorCoefficientArray(polyorder, coeffs);
        break;
    default:
        PARELAG_TEST_FOR_EXCEPTION(
            true,
            std::invalid_argument,
            "fillVectorCoefficientArray: Invalid nDim.");
    }
}

void fillRTVectorCoefficientArray(int nDim, int polyorder,
                                  Array<VectorCoefficient *> & coeffs)
{
    PARELAG_TEST_FOR_EXCEPTION(
        polyorder < 0,
        std::invalid_argument,
        "fillRTVectorCoefficientArray(): polyorder must be >= 0.");

    switch(nDim)
    {
    case 2:
        fillRT2DVectorCoefficientArray(polyorder, coeffs);
        break;
    case 3:
        fillRT3DVectorCoefficientArray(polyorder, coeffs);
        break;
    default:
        PARELAG_TEST_FOR_EXCEPTION(
            true,
            std::invalid_argument,
            "fillRTVectorCoefficientArray(): Invalid nDim.");
    }
}

void fill2DCoefficientArray(int polyorder, Array<Coefficient *> & coeffs)
{
    PARELAG_TEST_FOR_EXCEPTION(
        polyorder < 0,
        std::invalid_argument,
        "fill2DCoefficientArray(): polyorder must be >= 0." );

    const int size = (polyorder+1)*(polyorder+2)/2;
    coeffs.SetSize(size);
    Coefficient ** c = coeffs.GetData();

    int order_x, order_y;

    for(int order_max(0); order_max <= polyorder; ++order_max)
        for(order_x = 0; order_x <= order_max; ++order_x)
        {
            order_y = order_max - order_x;
            *(c++) = new PolynomialCoefficient(order_x, order_y);
        }
}

void fill3DCoefficientArray(int polyorder, Array<Coefficient *> & coeffs)
{
    PARELAG_TEST_FOR_EXCEPTION(
        polyorder < 0,
        std::invalid_argument,
        "fill3DCoefficientArray(): polyorder must be >= 0." );

    int size = 0;
    int order_x, order_y, order_z;
    for(int order_max(0); order_max <= polyorder; ++order_max)
        for(order_x = 0; order_x <= order_max; ++order_x)
            for(order_y = 0; order_y <= order_max-order_x; ++order_y)
                ++size;

    coeffs.SetSize(size);
    Coefficient ** c = coeffs.GetData();

    for(int order_max(0); order_max <= polyorder; ++order_max)
        for(order_x = 0; order_x <= order_max; ++order_x)
            for(order_y = 0; order_y <= order_max-order_x; ++order_y)
            {
                order_z = order_max - order_x - order_y;
                *(c++) = new PolynomialCoefficient(order_x, order_y, order_z);
            }
}

void fill2DVectorCoefficientArray(int polyorder,
                                  Array<VectorCoefficient *> & coeffs)
{

    PARELAG_TEST_FOR_EXCEPTION(
        polyorder < 0,
        std::invalid_argument,
        "fill2DVectorCoefficientArray(): polyorder must be >= 0." );

    constexpr int ncomp = 2;

    const int size = ncomp*(polyorder+1)*(polyorder+2)/2;
    coeffs.SetSize(size);
    VectorCoefficient ** c = coeffs.GetData();

    int order_x, order_y;
    for(int order_max(0); order_max <= polyorder; ++order_max)
        for(order_x = 0; order_x <= order_max; ++order_x)
            for(int icomp(0); icomp < ncomp; ++icomp)
            {
                order_y = order_max - order_x;
                *(c++) = new VectorPolynomialCoefficient(2, icomp, order_x, order_y);
            }
}

void fill3DVectorCoefficientArray(int polyorder,
                                  Array<VectorCoefficient *> & coeffs)
{

    PARELAG_TEST_FOR_EXCEPTION(
        polyorder < 0,
        std::invalid_argument,
        "fill3DVectorCoefficientArray(): polyorder must be >= 0." );

    constexpr int ncomp = 3;

    int size = 0;
    int order_x, order_y, order_z;
    for(int order_max(0); order_max <= polyorder; ++order_max)
        for(order_x = 0; order_x <= order_max; ++order_x)
            for(order_y = 0; order_y <= order_max-order_x; ++order_y)
                ++size;

    size *= ncomp;
    coeffs.SetSize(size);
    VectorCoefficient ** c = coeffs.GetData();

    for(int order_max(0); order_max <= polyorder; ++order_max)
        for(order_x = 0; order_x <= order_max; ++order_x)
            for(order_y = 0; order_y <= order_max-order_x; ++order_y)
                for(int icomp(0); icomp < ncomp; ++icomp)
                {
                    order_z = order_max - order_x - order_y;
                    *(c++) = new VectorPolynomialCoefficient(3, icomp, order_x, order_y, order_z);
                }
}

void fillRT2DVectorCoefficientArray(int polyorder,
                                    Array<VectorCoefficient *> & coeffs)
{

    PARELAG_TEST_FOR_EXCEPTION(
        polyorder < 0,
        std::invalid_argument,
        "fillRT2DVectorCoefficientArray(): polyorder must be >= 0." );

    constexpr int ncomp = 2;

    int size = ncomp*(polyorder+1)*(polyorder+2)/2 + polyorder + 1;
    coeffs.SetSize(size);
    VectorCoefficient ** c = coeffs.GetData();

    int order_x, order_y;

    for(int order_max(0); order_max <= polyorder; ++order_max)
        for(order_x = 0; order_x <= order_max; ++order_x)
            for(int icomp(0); icomp < ncomp; ++icomp)
            {
                order_y = order_max - order_x;
                *(c++) = new VectorPolynomialCoefficient(2, icomp, order_x, order_y);
            }

    for(order_x = 0; order_x <= polyorder; ++order_x)
    {
        order_y = polyorder - order_x;
        *(c++) = new RTPullbackCoefficient(2, order_x, order_y);
    }
}

void fillRT3DVectorCoefficientArray(int polyorder,
                                    Array<VectorCoefficient *> & coeffs)
{
    PARELAG_TEST_FOR_EXCEPTION(
        polyorder < 0,
        std::invalid_argument,
        "fillRT3DVectorCoefficientArray(): polyorder must be >= 0." );

    constexpr int ncomp = 3;

    int size = 0;
    int order_x, order_y, order_z;
    for(int order_max(0); order_max <= polyorder; ++order_max)
        for(order_x = 0; order_x <= order_max; ++order_x)
            for(order_y = 0; order_y <= order_max-order_x; ++order_y)
                ++size;
    size *= ncomp;

    for(order_x = 0; order_x <= polyorder; ++order_x)
        for(order_y = 0; order_y <= polyorder-order_x; ++order_y)
            ++size;

    coeffs.SetSize(size);
    VectorCoefficient ** c = coeffs.GetData();

    for(int order_max(0); order_max <= polyorder; ++order_max)
        for(order_x = 0; order_x <= order_max; ++order_x)
            for(order_y = 0; order_y <= order_max-order_x; ++order_y)
                for(int icomp(0); icomp < ncomp; ++icomp)
                {
                    order_z = order_max - order_x - order_y;
                    *(c++) = new VectorPolynomialCoefficient(3, icomp, order_x, order_y, order_z);
                }

    for(order_x = 0; order_x <= polyorder; ++order_x)
        for(order_y = 0; order_y <= polyorder-order_x; ++order_y)
        {
            order_z = polyorder - order_x - order_y;
            *(c++) = new RTPullbackCoefficient(3, order_x, order_y, order_z);
        }
}
}//namespace parelag
