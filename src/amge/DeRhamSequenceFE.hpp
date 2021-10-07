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

#ifndef DERHAMSEQUENCEFE2_HPP_
#define DERHAMSEQUENCEFE2_HPP_

#include <array>
#include <memory>

#include <mfem.hpp>

#include "amge/DeRhamSequence.hpp"
#include "linalg/dense/ParELAG_MultiVector.hpp"
#include "topology/Topology.hpp"
#include "amge/Coefficient.hpp"

namespace parelag
{
/// \class DeRhamSequenceFE
/// \brief A DeRhamSequence based on FE spaces.
///
/// There has been a change to the public interface. The Build()
/// method has been abruptly depreciated and all evidence thereof has
/// been erased. This was done to enforce ownership policies more
/// clearly. In particular, a DeRhamSequenceFE is responsible for its
/// data members (among other reasons, because DeRhamSequence is
/// responsible for its data members) and now owns its data
/// uniquely. Previously, the default was to assume no ownership of
/// the data and allow Build() to pass in user-defined forms,
/// interpolators, etc.
///
/// In practice, however, DeRhamSequenceFE is not a complete class --
/// there is at least one pure virtual function. Thus, a user could
/// not create an object explicitly of type DeRhamSequenceFE and then
/// build it using the previously existing Build() function
/// anyway. Thus, the new canonical method for constructing a
/// DeRhamSequenceFE object is to write a complete subclass using the
/// integrators that are needed for whatever the application. The
/// current complete subclasses, DeRhamSequence3D_FE and
/// DeRhamSequence2D_Hdiv_FE, *DO* own their data, and thus additional
/// user-defined subclasses must as well.
///
/// Moreover, methods like ReplaceMassIntegrator() now assume
/// ownership of the Integrator that is passed in, for example.
///
/// If there is a *good* reason to, say, share ownership of
/// data, please log a complaint on JIRA and we can discuss it.
class DeRhamSequenceFE : public DeRhamSequence
{
public:

    /// \name Constructors and destructor
    ///@{

    /// Constructor
    DeRhamSequenceFE(
        const std::shared_ptr<AgglomeratedTopology>& topo,int nSpaces);

    /// Destructor
    virtual ~DeRhamSequenceFE();

    ///@}
    /// \name Deleted special functions
    ///@{

    DeRhamSequenceFE(DeRhamSequenceFE const&) = delete;
    DeRhamSequenceFE(DeRhamSequenceFE&&) = delete;

    DeRhamSequenceFE& operator=(DeRhamSequenceFE const&) = delete;
    DeRhamSequenceFE& operator=(DeRhamSequenceFE&&) = delete;

    ///@}
    /// \name Getters
    ///@{

    /// Return the finite element space for a given form
    mfem::FiniteElementSpace * GetFeSpace(int jform)
    {
        return FESpace_[jform].get();
    }

    /// Return the finite element collection for a given form
    mfem::FiniteElementCollection * GetFeColl(int jform)
    {
        return FEColl_[jform].get();
    }

    ///@}
    /// \name Math stuff
    ///@{
    void ReplaceMassIntegrator(
        AgglomeratedTopology::Entity ientity,
        int jform,
        std::unique_ptr<mfem::BilinearFormIntegrator> m,
        bool recompute = true);

    std::unique_ptr<MultiVector> InterpolateScalarTargets(
        int jform, const mfem::Array<mfem::Coefficient *> & scalars);
    std::unique_ptr<MultiVector> InterpolateVectorTargets(
        int jform, const mfem::Array<mfem::VectorCoefficient *> & vectors);

    ///@}
    /// \name (Re)implementation of DeRhamSequence interface
    ///@{

    /// TODO
    virtual std::unique_ptr<mfem::SparseMatrix>
    ComputeProjectorFromH1ConformingSpace(int jform) const override;

    /// display the coarse basis functions interactively with GLVis
    virtual void showP(int jform,
                       mfem::SparseMatrix & P,
                       mfem::Array<int> & parts) override;

    /// interactive visualization of v with GLVis
    virtual void show(int jform, MultiVector & v) override;

    /// same as show, except v is assumed to be "true"
    virtual void ShowTrueData(int jform, MultiVector & true_v) override;

    /// save a meshfile and vector file for visualization with GLVis
    virtual void ExportGLVis(int jform,
                             mfem::Vector & v,
                             std::ostream & os) override;

    /// Return a pointer to the FE sequence
    virtual DeRhamSequenceFE * FemSequence() override
    {
        return this;
    }

    /// TODO document this
    virtual void ProjectCoefficient(int jform,
                                    mfem::Coefficient & c,
                                    mfem::Vector & v) override;

    /// TODO document this
    virtual void ProjectVectorCoefficient(int jform,
                                          mfem::VectorCoefficient & c,
                                          mfem::Vector & v) override;

    /// \brief Add (global) polynomial targets to sequence so they will be
    ///        included in coarse spaces.
    ///
    /// This code previously lived in various example/test codes and is
    /// collected here mostly for code reuse.
    void SetUpscalingTargets(int nDimensions, int upscalingOrder,
                             int form_start=-1);

    ///@}
    /// \name Debugging
    ///@{
    /// never called, probably DEPRECATED
    void DEBUG_CheckNewLocalMassAssembly();
    ///@}

    /// Update FE spaces and reassemble.
    void Update();

protected:

    /// \name Protected functions
    ///@{

    /// TODO document this
    void buildDof();

    /// TODO document this
    void assembleLocalMass();

    /// TODO document this
    void assembleLocalMass_ser();

    /// never really used, probably DEPRECATED
    void assembleLocalMass_old();

    /// TODO document this
    void assembleLocalMass_omp();

    /// TODO document this
    void assembleDerivative();

    /// TODO document this
    virtual void computePVTraces(
        AgglomeratedTopology::Entity icodim,
        mfem::Vector & PVinAgg) override = 0;

    ///@}
    /// \name Protected data members
    ///@{
    // FIXME (trb 12/15/15): Why so many protected data members??

    std::array<std::unique_ptr<mfem::DiscreteInterpolator>,3> di_;
    std::array<std::unique_ptr<mfem::FiniteElementCollection>,4> FEColl_;
    std::array<std::unique_ptr<mfem::FiniteElementSpace>,4> FESpace_;

    // (trb 12/15/15): I'm hoping this gives some efficiency over an
    // std::vector. In 2D this will waste 4 std::unique_ptrs holding
    // nullptr, but we should really be targeting 3D anyway. And
    // std::unique_ptr has negligible overhead over a raw pointer...
    std::array<std::unique_ptr<mfem::BilinearFormIntegrator>,10> mi_;
    // IMPORTANT!!!
    // This represents the weird 2D array:
    // Codim/Form   0 1 2 3
    //      0     [ 6 3 1 0 ]
    //      1     [ 7 4 2   ]
    //      2     [ 8 5     ]
    //      3     [ 9       ]
    //
    // This is stored column-wise from right to left, top to bottom,
    // as indicated by the values in the array. Thus, to obtain
    // (codim,form)=(2,1), one would access mi[5].
    //
    // Alternatively,
    //   array2D(form,codim) = mi[(dim - form)*(nforms - form)/2 + codim]
    //
    // (This expression is valid in 3D for the 2D array above and in
    // 2D for the analogous 2D array here:
    // Codim/Form   0 1 2
    //      0     [ 3 1 0 ]
    //      1     [ 4 2   ]
    //      2     [ 5     ]
    //
    // This needs to change if we ever decide to size everything
    // dynamically and only allocate space for the spaces we build.)
    //
    // Any library functions that access the array are expected to
    // expose the (form,codim)-type interface to the user and compute
    // the index manually.
    //
    // This assumes that nforms = ndim + 1.
    //
    // Functions may want to validate assumptions (form>jformstart)
    // and (form+codim<nforms), preferably only in debug mode.


    // This should eventually be a shared_ptr probably
    mfem::ParMesh * Mesh_;
    ///@}

private:

    /// \name Private functions
    ///@{

    const mfem::FiniteElement * GetFE(int ispace,
                                      int ientity_type,
                                      int ientity) const;

    mfem::ElementTransformation * GetTransformation(int ientity_type,
                                                    int ientity) const;

    void GetTransformation(int ientity_type,
                           int ientity,
                           mfem::IsoparametricTransformation & tr) const;
    ///@}
};


/// \class DeRhamSequence3D_FE
/// \brief Represents a DeRhamSequenceFE in three dimensions
class DeRhamSequence3D_FE : public DeRhamSequenceFE
{
public:

    /// Constructor.
    DeRhamSequence3D_FE(const std::shared_ptr<AgglomeratedTopology>& topo,
                        mfem::ParMesh * mesh,
                        int order, bool assemble=true, bool assemble_mass=true);

    /// Destructor
    virtual ~DeRhamSequence3D_FE();

protected:

    /// TODO
    virtual void computePVTraces(AgglomeratedTopology::Entity icodim,
                                 mfem::Vector & pv) override;
};

/// \class DeRhamSequence2D_Hdiv_FE
/// \brief Represents the HDIV-based sequence in 2D
/// (i.e. H1->HDiv->L2, using RT for the HDiv subspace)
class DeRhamSequence2D_Hdiv_FE : public DeRhamSequenceFE
{
public:
    /// Constructor
    DeRhamSequence2D_Hdiv_FE(const std::shared_ptr<AgglomeratedTopology>& topo,
                             mfem::ParMesh * mesh,
                             int order, bool assemble=true, bool assemble_mass=true);

    /// Destructor
    virtual ~DeRhamSequence2D_Hdiv_FE();

protected:

    /// TODO
    virtual void computePVTraces(
        AgglomeratedTopology::Entity icodim, mfem::Vector & pv) override;
};

void InterpolatePV_L2(const mfem::FiniteElementSpace * fespace,
                      const mfem::SparseMatrix & AE_element,
                      mfem::Vector & AE_Interpolant);

void InterpolatePV_HdivTraces(const mfem::FiniteElementSpace * fespace,
                              const mfem::SparseMatrix & AF_facet,
                              mfem::Vector & AF_Interpolant);

void InterpolatePV_HcurlTraces(const mfem::FiniteElementSpace * fespace,
                               const mfem::SparseMatrix & AR_ridge,
                               mfem::Vector & AR_Interpolant);

void InterpolatePV_H1Traces(const mfem::FiniteElementSpace * fespace,
                            const mfem::SparseMatrix & AP_peak,
                            mfem::Vector & AP_Interpolant);

}//namespace parelag
#endif /* DERHAMSEQUENCEFE_HPP_ */
