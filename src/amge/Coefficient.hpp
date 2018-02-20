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

#ifndef AMGE_COEFFICIENT_HPP_
#define AMGE_COEFFICIENT_HPP_

#include <cmath>

#include <mfem.hpp>

#include "utilities/elagError.hpp"

namespace parelag
{
class PolynomialCoefficient : public mfem::Coefficient
{
public:
    PolynomialCoefficient(int orderx, int ordery, int orderz = 0):
        order_x(orderx),
        order_y(ordery),
        order_z(orderz)
    {}

    virtual double Eval(mfem::ElementTransformation &T,
                        const mfem::IntegrationPoint &ip)
    {
        using std::pow;
        double x[3];
        mfem::Vector transip(x, 3);

        T.Transform(ip, transip);

        return pow(x[0],order_x)*pow(x[1],order_y)*pow(x[2],order_z);
    }
    virtual void Read(std::istream &in){ in >> order_x >> order_y >> order_z;}

private:
    int order_x;
    int order_y;
    int order_z;
};


class VectorPolynomialCoefficient : public mfem::VectorCoefficient
{

public:
    VectorPolynomialCoefficient(int VDim,
                                int component_,
                                int orderx,
                                int ordery,
                                int orderz=0):
        mfem::VectorCoefficient(VDim),
        component(component_),
        order_x(orderx),
        order_y(ordery),
        order_z(orderz)
    {
        PARELAG_TEST_FOR_EXCEPTION(
            component >= VDim,
            std::logic_error,
            "VectorPolynomialCoefficient: component >= VDim");
    };


    using mfem::VectorCoefficient::Eval;
    virtual void Eval(mfem::Vector &V,
                      mfem::ElementTransformation &T,
                      const mfem::IntegrationPoint &ip)
    {
        using std::pow;
        double x[3]; x[2] = 1.0;
        //Just to make sure that pow(x[2], ORDER_Z) is always ok
        mfem::Vector transip(x, 3);

        T.Transform (ip, transip);

        V.SetSize (vdim);
        V = 0.0;
        V(component) = pow(x[0],order_x)*pow(x[1],order_y)*pow(x[2],order_z);
    }

    virtual ~VectorPolynomialCoefficient() { }

private:
    int component;
    int order_x;
    int order_y;
    int order_z;
};

class RTPullbackCoefficient : public mfem::VectorCoefficient
{
public:
    RTPullbackCoefficient(int VDim,
                          int order_x_,
                          int order_y_,
                          int order_z_ = 0):
        mfem::VectorCoefficient(VDim),
        order_x(order_x_),
        order_y(order_y_),
        order_z(order_z_)
    {
    }

    using mfem::VectorCoefficient::Eval;
    virtual void Eval(mfem::Vector &V,
                      mfem::ElementTransformation &T,
                      const mfem::IntegrationPoint &ip)
    {
        using std::pow;

        double x[3]; x[2] = 1.0;
        //Just to make sure that pow(x[2], ORDER_Z) is always ok
        mfem::Vector transip(x, 3);

        T.Transform (ip, transip);

        V.SetSize (vdim);

        double scaling = pow(x[0],order_x)*pow(x[1],order_y)*pow(x[2],order_z);

        for(int comp(0); comp < vdim; ++comp)
            V(comp) = scaling * x[comp];
    }

private:
    int order_x;
    int order_y;
    int order_z;
};

void fillCoefficientArray(int nDim,
                          int polyorder,
                          mfem::Array<mfem::Coefficient *> & coeffs);
void fill2DCoefficientArray(int polyorder,
                            mfem::Array<mfem::Coefficient *> & coeffs);
void fill3DCoefficientArray(int polyorder,
                            mfem::Array<mfem::Coefficient *> & coeffs);

void fillVectorCoefficientArray(
    int nDim,int polyorder,
    mfem::Array<mfem::VectorCoefficient *> & coeffs);
void fill2DVectorCoefficientArray(
    int polyorder,
    mfem::Array<mfem::VectorCoefficient *> & coeffs);
void fill3DVectorCoefficientArray(
    int polyorder,
    mfem::Array<mfem::VectorCoefficient *> & coeffs);
void fillRTVectorCoefficientArray(
    int nDim,int polyorder,
    mfem::Array<mfem::VectorCoefficient *> & coeffs);
void fillRT2DVectorCoefficientArray(
    int polyorder,
    mfem::Array<mfem::VectorCoefficient *> & coeffs);
void fillRT3DVectorCoefficientArray(
    int polyorder,
    mfem::Array<mfem::VectorCoefficient *> & coeffs);

template<class T>
void freeCoeffArray(mfem::Array<T*> & coeffs)
{
    T** c = coeffs.GetData();
    T** end = c + coeffs.Size();

    for( ; c != end; ++c)
        delete *c;
}
}
#endif /* AMGE_COEFFICIENT_HPP_ */
