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

#ifndef BILININTEGRATORS_HPP_
#define BILININTEGRATORS_HPP_

#include <mfem.hpp>

#include "linalg/dense/ParELAG_LDLCalculator.hpp"

namespace parelag
{
/// Class for local mass matrix assemblying a(u,v) := (Q u, v)
class VectorFEtracesMassIntegrator: public mfem::BilinearFormIntegrator
{
private:
#ifndef	ParELAG_ENABLE_OPENMP
    mfem::Vector shape, te_shape;
#endif
    mfem::Coefficient *Q;

public:
    VectorFEtracesMassIntegrator(const mfem::IntegrationRule *ir = nullptr)
        : mfem::BilinearFormIntegrator(ir),
#ifndef ParELAG_ENABLE_OPENMP
        shape{},
        te_shape{},
#endif
        Q{nullptr}
    { }

    /// Construct a mass integrator with coefficient q
    VectorFEtracesMassIntegrator(mfem::Coefficient &q,
                                 const mfem::IntegrationRule *ir = nullptr)
        : mfem::BilinearFormIntegrator(ir),
#ifndef ParELAG_ENABLE_OPENMP
        shape{},
        te_shape{},
#endif
        Q(&q) { }

    VectorFEtracesMassIntegrator(VectorFEtracesMassIntegrator const&) = delete;
    VectorFEtracesMassIntegrator(VectorFEtracesMassIntegrator&&) = delete;

    VectorFEtracesMassIntegrator& operator=(
        VectorFEtracesMassIntegrator const&) = delete;
    VectorFEtracesMassIntegrator& operator=(
        VectorFEtracesMassIntegrator&&) = delete;

    /** Given a particular Finite Element
        computes the element mass matrix elmat. */
    virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
                                       mfem::ElementTransformation &Trans,
                                       mfem::DenseMatrix &elmat);

    virtual void AssembleElementMatrix2(mfem::FiniteElement const&,
                                        mfem::FiniteElement const&,
                                        mfem::ElementTransformation &,
                                        mfem::DenseMatrix &)
    { mfem::mfem_error("NA");}
};

/// Class for local mass matrix assemblying a(u,v) := (Q u, v)
class VolumetricFEMassIntegrator: public mfem::BilinearFormIntegrator
{
private:
#ifndef	ParELAG_ENABLE_OPENMP
    mfem::Vector shape, te_shape;
#endif
    mfem::Coefficient *Q;

public:
    VolumetricFEMassIntegrator(const mfem::IntegrationRule *ir = nullptr)
        : mfem::BilinearFormIntegrator(ir),
#ifndef ParELAG_ENABLE_OPENMP
        shape{},
        te_shape{},
#endif
        Q{nullptr}
    { }

    /// Construct a mass integrator with coefficient q
    VolumetricFEMassIntegrator(mfem::Coefficient &q, const mfem::IntegrationRule *ir = nullptr)
        : mfem::BilinearFormIntegrator(ir),
#ifndef ParELAG_ENABLE_OPENMP
        shape{},
        te_shape{},
#endif
        Q{&q}
    { }

    VolumetricFEMassIntegrator(VolumetricFEMassIntegrator const&) = delete;
    VolumetricFEMassIntegrator(VolumetricFEMassIntegrator&&) = delete;

    VolumetricFEMassIntegrator& operator=(
        VolumetricFEMassIntegrator const&) = delete;
    VolumetricFEMassIntegrator& operator=(
        VolumetricFEMassIntegrator&&) = delete;

    /** Given a particular Finite Element computes the element
        mass matrix elmat. */
    virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
                                       mfem::ElementTransformation &Trans,
                                       mfem::DenseMatrix &elmat);
    virtual void AssembleElementMatrix2(mfem::FiniteElement const&,
                                        mfem::FiniteElement const&,
                                        mfem::ElementTransformation &,
                                        mfem::DenseMatrix &)
    { mfem::mfem_error("NA");}
};

/// Class for local mass matrix assemblying a(u,v) := (Q u, v)
class PointFEMassIntegrator: public mfem::BilinearFormIntegrator
{
private:

public:
    PointFEMassIntegrator()
        : mfem::BilinearFormIntegrator() {  }

    /** Given a particular Finite Element computes the element
        mass matrix elmat. */
    virtual void AssembleElementMatrix(mfem::FiniteElement const&,
                                       mfem::ElementTransformation &,
                                       mfem::DenseMatrix &elmat)
    {
        elmat.SetSize(1,1);
        elmat(0,0) = 1;
    }

    virtual void AssembleElementMatrix2(mfem::FiniteElement const&,
                                        mfem::FiniteElement const&,
                                        mfem::ElementTransformation &,
                                        mfem::DenseMatrix &)
    { mfem::mfem_error("NA");}
};

/// Integrator for (Q u, v) for VectorFiniteElements
class ND_3D_FacetMassIntegrator: public mfem::BilinearFormIntegrator
{
private:
    mfem::Coefficient *Q;
    mfem::VectorCoefficient *VQ;

#ifndef ParELAG_ENABLE_OPENMP
    mfem::Vector shape;
    mfem::Vector D;
    mfem::DenseMatrix vshape;
#endif

public:
    ND_3D_FacetMassIntegrator()
        : Q{nullptr}, VQ{nullptr}
#ifndef ParELAG_ENABLE_OPENMP
        ,shape{}, D{}, vshape{}
#endif
    {}

    ND_3D_FacetMassIntegrator(mfem::Coefficient *_q)
        : Q{_q}, VQ{nullptr}
#ifndef ParELAG_ENABLE_OPENMP
        ,shape{}, D{}, vshape{}
#endif
    {}
    ND_3D_FacetMassIntegrator(mfem::Coefficient &q)
        : Q{&q}, VQ{nullptr}
#ifndef ParELAG_ENABLE_OPENMP
        ,shape{}, D{}, vshape{}
#endif
    {}
    ND_3D_FacetMassIntegrator(mfem::VectorCoefficient *_vq)
        : Q{nullptr}, VQ{_vq}
#ifndef ParELAG_ENABLE_OPENMP
        ,shape{}, D{}, vshape{}
#endif
    {}
    ND_3D_FacetMassIntegrator(mfem::VectorCoefficient &vq)
        : Q{nullptr}, VQ{&vq}
#ifndef ParELAG_ENABLE_OPENMP
        ,shape{}, D{}, vshape{}
#endif
    {}

    ND_3D_FacetMassIntegrator(ND_3D_FacetMassIntegrator const&) = delete;
    ND_3D_FacetMassIntegrator(ND_3D_FacetMassIntegrator&&) = delete;

    ND_3D_FacetMassIntegrator& operator=(
        ND_3D_FacetMassIntegrator const&) = delete;
    ND_3D_FacetMassIntegrator& operator=(
        ND_3D_FacetMassIntegrator&&) = delete;

    virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
                                       mfem::ElementTransformation &Trans,
                                       mfem::DenseMatrix &elmat);
    virtual void AssembleElementMatrix2(const mfem::FiniteElement &trial_fe,
                                        const mfem::FiniteElement &test_fe,
                                        mfem::ElementTransformation &Trans,
                                        mfem::DenseMatrix &elmat);
};

class SpectralLumpedIntegrator : public mfem::BilinearFormIntegrator
{
private:
    mfem::BilinearFormIntegrator *bfi;
    int own_bfi;
#ifndef	ParELAG_ENABLE_OPENMP
    mfem::DenseMatrix elmat_f;
#endif

public:
    SpectralLumpedIntegrator(mfem::BilinearFormIntegrator * _bfi,
                             int _own_bfi = 1);

    SpectralLumpedIntegrator(SpectralLumpedIntegrator const&) = delete;
    SpectralLumpedIntegrator(SpectralLumpedIntegrator&&) = delete;

    SpectralLumpedIntegrator& operator=(
        SpectralLumpedIntegrator const&) = delete;
    SpectralLumpedIntegrator& operator=(
        SpectralLumpedIntegrator&&) = delete;

    virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
                                       mfem::ElementTransformation &Trans,
                                       mfem::DenseMatrix &elmat);
    virtual ~SpectralLumpedIntegrator();
};

class DiagonalIntegrator : public mfem::BilinearFormIntegrator
{
private:
    mfem::BilinearFormIntegrator *bfi;
    int own_bfi;

public:
    DiagonalIntegrator(mfem::BilinearFormIntegrator * _bfi,
                       int _own_bfi = 1);

    DiagonalIntegrator(DiagonalIntegrator const&) = delete;
    DiagonalIntegrator(DiagonalIntegrator&&) = delete;

    DiagonalIntegrator& operator=(
        DiagonalIntegrator const&) = delete;
    DiagonalIntegrator& operator=(
        DiagonalIntegrator&&) = delete;

    virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
                                       mfem::ElementTransformation &Trans,
                                       mfem::DenseMatrix &elmat);
    virtual ~DiagonalIntegrator();
};

/** Class for constructing the (local) discrete divergence matrix
    which can be used as an integrator in a DiscreteLinearOperator
    object to assemble the global discrete divergence matrix.

    Note: Since the dofs in the L2_FECollection are nodal values,
    the local discrete divergence matrix (with an RT-type domain
    space) will depend on the transformation. On the other hand,
    the local matrix returned by VectorFEDivergenceIntegrator is
    independent of the transformation.

    this does the same thing as mfem::DivergenceInterpolotor for
    orders 0, 1
*/
class DivergenceInterpolator2 : public mfem::DiscreteInterpolator
{
public:
    DivergenceInterpolator2() = default;

    virtual void AssembleElementMatrix2(const mfem::FiniteElement &dom_fe,
                                        const mfem::FiniteElement &ran_fe,
                                        mfem::ElementTransformation &Trans,
                                        mfem::DenseMatrix &elmat);
private:
    mfem::VectorFEDivergenceIntegrator div_int;
    mfem::MassIntegrator mass_int;
    LDLCalculator mass_inv;
    mfem::DenseMatrix elmat_div;
    mfem::DenseMatrix elmat_mass;
};

void LoseInterpolators(mfem::DiscreteLinearOperator & l);
}//namespace parelag
#endif // BILININTEGRATORS_HPP_
