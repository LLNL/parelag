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

#include "bilinIntegrators.hpp"

#include "utilities/elagError.hpp"

namespace parelag
{
using namespace mfem;

void VectorFEtracesMassIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
    int nd = el.GetDof();
    // int dim = el.GetDim();
    double w;

    elmat.SetSize(nd);
#ifdef ParELAG_ENABLE_OPENMP
    Vector shape(nd);
#else
    shape.SetSize(nd);
#endif

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
        // int order = 2 * el.GetOrder();
        int order = 2 * el.GetOrder() + Trans.OrderW();

        if (el.Space() == FunctionSpace::rQk)
            ir = &RefinedIntRules.Get(el.GetGeomType(), order);
        else
            ir = &IntRules.Get(el.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcShape(ip, shape);

        Trans.SetIntPoint (&ip);
        w = ip.weight/Trans.Weight();
        if (Q)
            w *= Q -> Eval(Trans, ip);

        AddMult_a_VVt(w, shape, elmat);
    }
}

void VolumetricFEMassIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
    int nd = el.GetDof();
    // int dim = el.GetDim();
    double w;

    elmat.SetSize(nd);
#ifdef ParELAG_ENABLE_OPENMP
    Vector shape(nd);
#else
    shape.SetSize(nd);
#endif
    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
        // int order = 2 * el.GetOrder();
        int order = 2 * el.GetOrder() + Trans.OrderW();

        if (el.Space() == FunctionSpace::rQk)
            ir = &RefinedIntRules.Get(el.GetGeomType(), order);
        else
            ir = &IntRules.Get(el.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcShape(ip, shape);

        Trans.SetIntPoint (&ip);
        w = ip.weight/Trans.Weight();
        if (Q)
            w *= Q -> Eval(Trans, ip);

        AddMult_a_VVt(w, shape, elmat);
    }
}


void ND_3D_FacetMassIntegrator::AssembleElementMatrix(
    const FiniteElement &el,
    ElementTransformation &Trans,
    DenseMatrix &elmat)
{
   int dof  = el.GetDof();
   IsoparametricTransformation * tmp = dynamic_cast<IsoparametricTransformation *>(&Trans);
   elag_assert(tmp);
   int dim  = tmp->GetPointMat().Height();

    double w;

#ifdef ParELAG_ENABLE_OPENMP
    Vector D(VQ ? VQ->GetVDim() : 0);
    DenseMatrix vshape(dof, dim);
#else
    vshape.SetSize(dof,dim);
#endif

    elmat.SetSize(dof);
    elmat = 0.0;

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
        // int order = 2 * el.GetOrder();
        int order = Trans.OrderW() + 2 * el.GetOrder();
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        Trans.SetIntPoint (&ip);

        el.CalcVShape(Trans, vshape);

        w = ip.weight * Trans.Weight();
        if (VQ)
        {
            VQ->Eval(D, Trans, ip);
            D *= w;
            AddMultADAt(vshape, D, elmat);
        }
        else
        {
            if (Q)
                w *= Q -> Eval (Trans, ip);
            AddMult_a_AAt (w, vshape, elmat);
        }
    }
}

void ND_3D_FacetMassIntegrator::AssembleElementMatrix2(
    const FiniteElement &trial_fe, const FiniteElement &test_fe,
    ElementTransformation &Trans, DenseMatrix &elmat)
{
    // assume test_fe is scalar FE and trial_fe is vector FE
    IsoparametricTransformation * tmp =
        dynamic_cast<IsoparametricTransformation *>(&Trans);
    elag_assert(tmp);

    int dim  = tmp->GetPointMat().Height();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    double w;

    PARELAG_TEST_FOR_EXCEPTION(
        VQ,
        parelag::not_implemented_error,
        "ND_3D_FacetMassIntegrator::AssembleElementMatrix2(...):\n"
        "Not implemented for vector permeability.");

#ifdef ParELAG_ENABLE_OPENMP
    DenseMatrix vshape(trial_dof, dim);
    Vector shape(test_dof);
#else
    vshape.SetSize(trial_dof, dim);
    shape.SetSize(test_dof);
#endif

    elmat.SetSize (dim*test_dof, trial_dof);

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
        int order = (Trans.OrderW() + test_fe.GetOrder() + trial_fe.GetOrder());
        ir = &IntRules.Get(test_fe.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        Trans.SetIntPoint (&ip);

        trial_fe.CalcVShape(Trans, vshape);
        test_fe.CalcShape(ip, shape);

        w = ip.weight * Trans.Weight();
        if (Q)
            w *= Q -> Eval (Trans, ip);

        for (int d = 0; d < dim; d++)
        {
            for (int j = 0; j < test_dof; j++)
            {
                for (int k = 0; k < trial_dof; k++)
                {
                    elmat(d * test_dof + j, k) += w * shape(j) * vshape(k, d);
                }
            }
        }
    }
}


SpectralLumpedIntegrator::SpectralLumpedIntegrator(
    BilinearFormIntegrator * _bfi, int _own_bfi)
    : bfi(_bfi),
      own_bfi(_own_bfi)
#ifndef	ParELAG_ENABLE_OPENMP
    , elmat_f{}
#endif

{
}

void SpectralLumpedIntegrator::AssembleElementMatrix(
    const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
    int nd = el.GetDof();
    elmat.SetSize(nd);
    elmat = 0.;
#ifdef ParELAG_ENABLE_OPENMP
    DenseMatrix elmat_f(nd);
#endif
    bfi->AssembleElementMatrix(el, Trans, elmat_f);
    double d, s;

    for(int i(0); i < nd; ++i)
    {
        d = elmat(i,i) = elmat_f(i,i);
        s = 1./sqrt(d);
        for(int j(0); j <nd; ++j)
        {
            elmat_f(i,j) *= s;
            elmat_f(j,i) *= s;
        }
    }

    DenseMatrixEigensystem eigs(elmat_f);
    eigs.Eval();
    s = eigs.Eigenvalues().Min();

    elmat *= s;
}

SpectralLumpedIntegrator::~SpectralLumpedIntegrator()
{
    if(own_bfi)
        delete bfi;
}


DiagonalIntegrator::DiagonalIntegrator(
    BilinearFormIntegrator * _bfi, int _own_bfi)
    : bfi(_bfi),
      own_bfi(_own_bfi)
{
}

void DiagonalIntegrator::AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
    int nd = el.GetDof();
    bfi->AssembleElementMatrix(el, Trans, elmat);
    for(int i(0); i < nd; ++i)
        for(int j(0); j < i; ++j)
        {
            elmat(i,j) = 0.;
            elmat(j,i) = 0.;
        }
}

DiagonalIntegrator::~DiagonalIntegrator()
{
    if(own_bfi)
        delete bfi;
}

void DivergenceInterpolator2::AssembleElementMatrix2(const FiniteElement &dom_fe,
                                                     const FiniteElement &ran_fe,
                                                     ElementTransformation &Trans,
                                                     DenseMatrix &elmat)
{

    if( Trans.OrderW() < 2)
        ran_fe.ProjectDiv(dom_fe, Trans, elmat);
    else
    {
        mass_int.AssembleElementMatrix(ran_fe, Trans, elmat_mass);
        div_int.AssembleElementMatrix2(dom_fe, ran_fe,Trans,elmat_div);
        mass_inv.Compute(elmat_mass);
        elmat.SetSize(elmat_div.Height(), elmat_div.Width() );
        mass_inv.Mult(elmat_div, elmat);
    }

}

void LoseInterpolators(DiscreteLinearOperator & l)
{
    l.GetDI()->SetSize(0);
}
}//namespace parelag
