/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#ifndef BILININTEGRATORS_HPP_
#define BILININTEGRATORS_HPP_

/** Class for local mass matrix assemblying a(u,v) := (Q u, v) */
class VectorFEtracesMassIntegrator: public BilinearFormIntegrator
{
private:
#ifndef	ELEMAGG_USE_OPENMP
   Vector shape, te_shape;
#endif
   Coefficient *Q;

public:
   VectorFEtracesMassIntegrator(const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir) { Q = NULL; }
   /// Construct a mass integrator with coefficient q
   VectorFEtracesMassIntegrator(Coefficient &q, const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), Q(&q) { }

   /** Given a particular Finite Element
       computes the element mass matrix elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat){ mfem_error("NA");}
};

/** Class for local mass matrix assemblying a(u,v) := (Q u, v) */
class VolumetricFEMassIntegrator: public BilinearFormIntegrator
{
private:
#ifndef	ELEMAGG_USE_OPENMP
   Vector shape, te_shape;
#endif
   Coefficient *Q;

public:
   VolumetricFEMassIntegrator(const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir) { Q = NULL; }
   /// Construct a mass integrator with coefficient q
   VolumetricFEMassIntegrator(Coefficient &q, const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), Q(&q) { }

   /** Given a particular Finite Element
       computes the element mass matrix elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat){ mfem_error("NA");}
};

/** Class for local mass matrix assemblying a(u,v) := (Q u, v) */
class PointFEMassIntegrator: public BilinearFormIntegrator
{
private:

public:
   PointFEMassIntegrator()
      : BilinearFormIntegrator() {  }

   /** Given a particular Finite Element
       computes the element mass matrix elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat)
   {
	   elmat.SetSize(1,1);
	   elmat(0,0) = 1;
   }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat){ mfem_error("NA");}
};

/// Integrator for (Q u, v) for VectorFiniteElements
class ND_3D_FacetMassIntegrator: public BilinearFormIntegrator
{
private:
   Coefficient *Q;
   VectorCoefficient *VQ;

#ifndef ELEMAGG_USE_OPENMP
   Vector shape;
   Vector D;
   DenseMatrix vshape;
#endif

public:
   ND_3D_FacetMassIntegrator() { Q = NULL; VQ= NULL; }
   ND_3D_FacetMassIntegrator(Coefficient *_q) { Q = _q; VQ= NULL; }
   ND_3D_FacetMassIntegrator(Coefficient &q) { Q = &q; VQ= NULL; }
   ND_3D_FacetMassIntegrator(VectorCoefficient *_vq) { VQ = _vq; Q = NULL; }
   ND_3D_FacetMassIntegrator(VectorCoefficient &vq) { VQ = &vq; Q = NULL; }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

class SpectralLumpedIntegrator : public BilinearFormIntegrator
{
private:
	BilinearFormIntegrator *bfi;
	int own_bfi;
#ifndef	ELEMAGG_USE_OPENMP
	DenseMatrix elmat_f;
#endif

public:
	SpectralLumpedIntegrator(BilinearFormIntegrator * _bfi, int _own_bfi = 1);
	virtual void AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat);
	virtual ~SpectralLumpedIntegrator();
};

class DiagonalIntegrator : public BilinearFormIntegrator
{
private:
	BilinearFormIntegrator *bfi;
	int own_bfi;

public:
	DiagonalIntegrator(BilinearFormIntegrator * _bfi, int _own_bfi = 1);
	virtual void AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat);
	virtual ~DiagonalIntegrator();
};

/** Class for constructing the (local) discrete divergence matrix which can
    be used as an integrator in a DiscreteLinearOperator object to assemble
    the global discrete divergence matrix.

    Note: Since the dofs in the L2_FECollection are nodal values, the local
    discrete divergence matrix (with an RT-type domain space) will depend on
    the transformation. On the other hand, the local matrix returned by
    VectorFEDivergenceIntegrator is independent of the transformation. */
class DivergenceInterpolator2 : public DiscreteInterpolator
{
public:
   virtual void AssembleElementMatrix2(const FiniteElement &dom_fe,
                                       const FiniteElement &ran_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
private:
   VectorFEDivergenceIntegrator div_int;
   MassIntegrator mass_int;
   LDLCalculator mass_inv;
   DenseMatrix elmat_div;
   DenseMatrix elmat_mass;
};

void LoseInterpolators(DiscreteLinearOperator & l);

#endif /* BILININTEGRATORS_HPP_ */
