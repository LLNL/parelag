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

/*
  DeRhamSequence  -->  Define interface
     |
     -->  GeneralDeRhamSequence<Dim>
                    |
                    -->  DeRhamSequenceAlg<Dim>
                    |
                    -->  DeRhamSequenceFE<Dim>
                            |
                            -->  DeRhamSequence3D_FE<3>
                            |
                            -->  DeRhamSequence2D_Hdive_FE<2>
*/

#ifndef AMGE_DERHAMSEQUENCE_HPP_
#define AMGE_DERHAMSEQUENCE_HPP_

#include <array>
#include <memory>
#include <vector>

#include <mfem.hpp>

#include "amge/CochainProjector.hpp"
#include "amge/DofHandler.hpp"
#include "amge/DOFAgglomeration.hpp"
#include "amge/ElementalMatricesContainer.hpp"
#include "linalg/dense/ParELAG_MultiVector.hpp"

namespace parelag
{

class DeRhamSequenceFE;

/// \class DeRhamSequence
/// \brief The key class for the whole project, keeps track of all the
/// algebraic relationships at a given mesh level of the hierarchy.
///
/// This is normally created in two ways: from an MFEM ParMesh
/// (wrapped as an AgglomeratedTopology), or by calling Coarsen() from
/// another DeRhamSequence object.
class DeRhamSequence : public std::enable_shared_from_this<DeRhamSequence>
{
public:

    friend class DeRhamSequenceAlg;

    /// Spaces in the de Rham sequence
    enum Space {H1, HCURL, HDIV, L2, SPACE_SENTINEL};

    /// \name Constructor and destructor
    ///@{

    /// Constructor given an \b AgglomeratedTopology object and the
    /// number of spaces (3 for 2D, 4 for 3D)
    DeRhamSequence(const std::shared_ptr<AgglomeratedTopology>& topo,int nspaces);

    /// Destructor
    virtual ~DeRhamSequence();

    ///@}
    /// \name Member getter functions
    ///@{

    /// Return the form corresponding to the space identifier and dimensions.
    static int GetForm(int nDim, Space space)
    {
        PARELAG_ASSERT(space < SPACE_SENTINEL && nDim <= 3);

        switch (space)
        {
        case H1:
            return 0;
        case HCURL:
            return 1;
        case HDIV:
            return nDim - 1;
        case L2:
            return nDim;
        default:
            PARELAG_ASSERT(false);
            return -1;
        }
    }

    /// Return the number of forms represented in the sequence
    int GetNumForms() const noexcept
    {
        return nForms_;
    }

    /// Return the number of forms represented in the sequence
    int GetNumberOfForms() const noexcept
    {
        return this->GetNumForms();
    }

    // FIXME (trb 12/24/15): Do we care about bounds checking at all?
    //std::vector::at() will do it and emit an "out_of_range"
    //exception. operator[] will never throw, but causes "undefined
    //behavior" if the input is greater than the size.

    /// Return the number of local dofs (true+shared)
    int GetNumDofs(int jform) const
    {
#ifdef ELAG_DEBUG
        return Dof_.at(jform)->GetNDofs();
#else
        return Dof_[jform]->GetNDofs();
#endif
    }

    /// Return the number of local dofs (true+shared)
    int GetNumberOfDofs(int jform) const
    {
        return this->GetNumDofs(jform);
    }

    /// Return the number of local true dofs
    int GetNumTrueDofs(int jform) const
    {
#ifdef ELAG_DEBUG
        return Dof_.at(jform)->GetDofTrueDof().GetTrueLocalSize();
#else
        return Dof_[jform]->GetDofTrueDof().GetTrueLocalSize();
#endif
    }

    /// Return the number of local true dofs
    int GetNumberOfTrueDofs(int jform) const
    {
        return this->GetNumTrueDofs(jform);
    }

    mfem::SparseMatrix * GetD(int jform)
    {
        return D_[jform].get();
    }

    void SetD(int jform, std::unique_ptr<mfem::SparseMatrix> D)
    {
        D_[jform] = std::move(D);
    }

    mfem::SparseMatrix * GetDerivativeOperator(int jform)
    {
        return GetD(jform);
    }

    mfem::SparseMatrix * GetP(int jform)
    {
        return P_[jform].get();
    }

    void SetP(int jform, std::unique_ptr<mfem::SparseMatrix> P)
    {
        P_[jform] = std::move(P);
    }

    // FIXME!!
    const mfem::SparseMatrix& GetAEntityDof(
        AgglomeratedTopology::Entity etype, int jform)
    {
        return DofAgg_[jform]->GetAEntityDof(etype);
    }

    CochainProjector * GetPi(int jform)
    {
        return Pi_[jform].get();
    }

    DofHandler * GetDofHandler(int jform)
    {
        return Dof_[jform].get();
    }

    DofAgglomeration * GetDofAgg(int jform)
    {
        return DofAgg_[jform].get();
    }

    /// Returns local (element) mass matrices as one "DG-like" matrix.
    mfem::SparseMatrix * GetM(AgglomeratedTopology::Entity etype,
                              int jform)
    {
        const int nDim = Topo_->Dimensions();
        const int idx = (nDim-jform)*(nForms_-jform)/2 + etype;
        return M_[idx].get();
    }

    // FIXME (trb 04/14/2016): Can return const ref?
    AgglomeratedTopology * GetTopology() {return Topo_.get();}

    virtual DeRhamSequenceFE * FemSequence() = 0;

    ///@}
    /// \name Member setter functions
    ///@{

    /// Set the (global) targets.
    ///
    /// The sequence does a deep copy of the targets.
    void SetTargets(const mfem::Array<MultiVector *> & targets);
    // FIXME (trb 12/17/2015): Is there a reason the targets are
    // copied instead of moved? What are the implications of moving
    // them in, and offering views of them through the sequence? Same
    // for SetLocalTargets() below.

    /// This is very similar to \b SetTargets() in the sense that it
    /// just copies what the caller provides into \b
    /// localtargets(codim, jform).
    void SetLocalTargets(AgglomeratedTopology::Entity codim,
                         int jform,
                         const mfem::Array<MultiVector *>& localfunctions);


    /// This is very similar to \b SetLocalTargets() in the sense that
    /// it just sets \b localtargets(codim, jform) = \a
    /// localfunctions. This means that the provided array becomes
    /// owned by this class and must not be deleted by someone else.
    void OwnLocalTargets(
        AgglomeratedTopology::Entity codim,
        int jform,
        std::vector<std::unique_ptr<MultiVector>> localfunctions);


    /// Is this the first form that is defined?
    void SetjformStart(int start)
    {
        elag_assert(start >= Topo_->Dimensions() - Topo_->Codimensions());
        jformStart_ = start;
    }

    /// Set the SVD tolerance
    void SetSVDTol(double tol)
    {
        SVD_Tolerance_ = tol;
    }

    ///@}
    /// \name Math-y stuff
    ///@{

    /// \brief Create a coarse representation of the current
    /// DeRhamSequence.
    ///
    /// User is responsible for deleting the created object.
    std::shared_ptr<DeRhamSequence> Coarsen();

    /// Fills in dofAgg.
    void AgglomerateDofs();

    ///@}
    /// \name Helper functions
    ///@{

    /// Once the \b localtargets(0, 0) are set, this function can be
    /// called to populate all "traces" in H1 on all "codimensions".
    ///
    /// It also computes local gradients in H(curl) and populates all
    /// H(curl) "codimensions" with their "traces".  That is, it fills
    /// in \b localtargets for "forms" 0 and 1.
    ///
    /// \warning Currently only implemented for jform == 0
    void PopulateLocalTargetsFromForm(const int jform);

    /// TODO
    void ComputeSpaceInterpolationError(int jform,
                                        const MultiVector & fineVector);

    /// TODO
    virtual void ProjectCoefficient(int jform,
                                    mfem::Coefficient & c,
                                    mfem::Vector & v) = 0;

    /// TODO
    virtual void ProjectVectorCoefficient(int jform,
                                          mfem::VectorCoefficient & c,
                                          mfem::Vector & v) = 0;


    ///@}
    /// \name Visualization
    ///@{

    /// TODO
    void ShowProjector(int jform);

    /// TODO
    void ShowDerProjector(int jform);

    /// TODO
    virtual void showP(int jform,
                       mfem::SparseMatrix & P_t,
                       mfem::Array<int> & parts) = 0;

    /// TODO
    virtual void show(int jform, MultiVector & v) = 0;

    /// TODO
    virtual void ShowTrueData(int jform, MultiVector & true_v) = 0;

    /// TODO
    virtual void ExportGLVis(int jform, mfem::Vector & v, std::ostream & os) = 0;

    ///@}
    /// \name Factory functions
    ///
    /// These functions all create a *new* object that is returned to
    /// the caller. Thus, the naming convention "ComputeThing()" is
    /// used to contrast nonowning returns that are used by the
    /// similar "GetThing()".
    ///@{

    /// Returns a copy of parallel-ized D[jform]
    std::unique_ptr<ParallelCSRMatrix> ComputeTrueD(int jform) const;

    /// Returns a copy of parallel-ized D[jform] with boundary
    /// conditions applied
    std::unique_ptr<ParallelCSRMatrix>
    ComputeTrueD(int jform, mfem::Array<int> & ess_label);

    /// Calls the appropraite GetTrueD() to keep the previous function
    /// valid.
    ///
    /// If you don't know the syntax, you should learn it, but it
    /// doesn't matter -- it does what it's supposed to do.
    ///
    /// \warning This function will be depreciated soon-ish
    /// probably/hopefully
    template <typename... Ts>
    std::unique_ptr<ParallelCSRMatrix>
    ComputeTrueDerivativeOperator(Ts&&... args)
    {
        return ComputeTrueD(std::forward<Ts>(args)...);
    }

    /// Returns the parallel-ized mass matrix for the given form
    std::unique_ptr<ParallelCSRMatrix> ComputeTrueM(int jform);

    /// Returns the parallel-ized mass matrix for the given form
    std::unique_ptr<ParallelCSRMatrix> ComputeTrueM(
        int jform, mfem::Vector & elemMatrixScaling);

    /// Calls the appropraite GetTrueM() to keep the previous function
    /// valid.
    template <typename... Ts>
    std::unique_ptr<ParallelCSRMatrix>
    ComputeTrueMassOperator(Ts&&... args)
    {
        return ComputeTrueM(std::forward<Ts>(args)...);
    }

    /// Returns the parallel-ized P matrix for the given form
    std::unique_ptr<ParallelCSRMatrix> ComputeTrueP(int jform) const;

    const ParallelCSRMatrix& GetTrueP(int jform) const;

    /// Returns the parallel-ized P matrix for the given form with
    /// boundary conditions applied.
    std::unique_ptr<ParallelCSRMatrix> ComputeTrueP(
        int jform, mfem::Array<int> & ess_label) const;

    /// Returns the parallel-ized cochain projector matrix for a given
    /// form.
    std::unique_ptr<ParallelCSRMatrix> ComputeTruePi(int jform);

    const ParallelCSRMatrix& GetTruePi(int jform);

    /// TODO
    std::unique_ptr<ParallelCSRMatrix>
    ComputeTrueProjectorFromH1ConformingSpace(int jform) const;

    /// TODO

    // I realize this signature is a little strange. In an ideal
    // world, this would return an n-tuple (with n = dim), but we
    // can't overload on return type, so we pass in 3 unique_ptr
    // references. The reason these are unique_ptrs is because new
    // memory is allocated for which the caller becomes responsible at
    // the call exit. Note that this will destroy the current object
    // held by each unique_ptr (if any).
    void ComputeTrueProjectorFromH1ConformingSpace(
        int jform,
        std::unique_ptr<ParallelCSRMatrix>& Pix,
        std::unique_ptr<ParallelCSRMatrix>& Piy,
        std::unique_ptr<ParallelCSRMatrix>& Piz ) const;

    /// Return a copy of D[form] with boundary conditions applied
    std::unique_ptr<mfem::SparseMatrix>
    ComputeDerivativeOperator(int jform, mfem::Array<int> & ess_label);

    /// Create and return the mass operator
    std::unique_ptr<mfem::SparseMatrix> ComputeMassOperator(int jform)
    {
        const int nDim = Topo_->Dimensions();
        const int idx = (nDim-jform)*(nForms_-jform)/2
            + AgglomeratedTopology::ELEMENT;

        return Assemble(AgglomeratedTopology::ELEMENT,
                        *M_[idx],
                        *Dof_[jform],
                        *Dof_[jform]);
    }

    /// TODO
    std::unique_ptr<mfem::SparseMatrix> ComputeLumpedMassOperator(int jform);

    /// TODO
    std::unique_ptr<mfem::SparseMatrix>
    ComputeMassOperator(int jform, mfem::Vector & elemMatrixScaling);

    /// TODO
    std::unique_ptr<mfem::SparseMatrix>
    ComputeLumpedMassOperator(int jform, mfem::Vector & elemMatrixScaling);

    /// TODO
    virtual std::unique_ptr<mfem::SparseMatrix>
    ComputeProjectorFromH1ConformingSpace(int jform) const = 0;

    /// Calls ComputeP(int,mfem::Array<int>), to be consistent with
    /// "GetDerivativeOperator" and "ComputeDerivativeOperator"
    ///
    /// \warning This function will be deprecated soon
    std::unique_ptr<mfem::SparseMatrix>
    GetP(int jform, mfem::Array<int> & ess_label) const;

    /// Compute a copy of P[jform] with boundary conditions applied
    std::unique_ptr<mfem::SparseMatrix>
    ComputeP(int jform, mfem::Array<int> & ess_label) const;


    /// Compute the elemental mass matrices
    std::unique_ptr<ElementalMatricesContainer>
    GetElementalMassMatrices(int iform,
                             AgglomeratedTopology::Entity icodim);

    /// Get the representation of constant one function in L2 (dim-form)
    const mfem::Vector& GetL2ConstRepresentation() const { return L2_const_rep_; }
    mfem::Vector& GetL2ConstRepresentation() { return L2_const_rep_; }

    ///@}
    /// \name Exploit the linked list
    ///@{
    const DeRhamSequence * ViewCoarserSequence() const noexcept
    { return CoarserSequence_.lock().get(); }

    const DeRhamSequence * ViewFinerSequence() const noexcept
    { return FinerSequence_.lock().get(); }

    std::shared_ptr<DeRhamSequence> CoarserSequence() const noexcept
    { return CoarserSequence_.lock(); }

    std::shared_ptr<DeRhamSequence> FinerSequence() const noexcept
    { return FinerSequence_.lock(); }

    ///@}
    /// \name Debugging help
    ///@{

    /// TODO
    void DumpD();

    /// TODO
    void DumpP();

    /// Sanity check the DeRhamSequence
    ///
    /// This calls the following protected functions
    ///   1. CheckCoarseMassMatrix();
    ///   2. CheckTrueCoarseMassMatrix();
    ///   3. CheckD();
    ///   4. CheckTrueD();
    ///   5. CheckDP();
    ///   6. CheckTrueDP();
    ///   7. CheckCoarseDerivativeMatrix();
    ///   8. CheckTrueCoarseDerivativeMatrix();
    ///   9. CheckPi();
    ///
    void CheckInvariants();

    ///@}
    /// \name Public data members
    ///@{

    static std::stringstream DeRhamSequence_os;

    ///@}

protected:
    /// \name Routines called by CheckInvariants()
    ///@{
    /// Test matrix multiplication, check (SparseMatrix) M_c = P^T M_f P
    void CheckCoarseMassMatrix();
    /// Test matrix multiplication, check (HypreParMatrix) M_c - P^T M_f P
    void CheckTrueCoarseMassMatrix();
    /// Test exactness of sequence for local SparseMatrix, D_{i+1} D_i = 0
    void CheckD();
    /// Test exactness of sequence for HypreParMatrix
    void CheckTrueD();
    /// Test commutativity, for local SparseMatrix
    void CheckDP();
    /// Test commutativity for HypreParMatrix
    void CheckTrueDP();
    /// Test if coarse D_c = \Pi D_f P, (kind of commutativity)
    void CheckCoarseDerivativeMatrix();
    /// Same for HypreParMatrix multiplication
    void CheckTrueCoarseDerivativeMatrix();
    /// Test that chosen targets are actually represented on coarse space
    void CheckPi();
    ///@}

    int EstimateUpperBoundNCoarseDof(int jform);
    void ComputeCoarseTraces(int jform);
    void ComputeCoarseTracesNoTargets(int jform);
    void ComputeCoarseTracesWithTargets(int jform);
    void Compute0formCoarseTraces();

    /// Split localtargets into interior and boundary components, append it
    /// to (already partially filled) localtargets_interior, _boundary
    /// This is some shared code between hFacetExtension, hRidgeExtension
    void PartitionLocalTargets(
        int nTargets, int nlocaltargets,
        MultiVector& localtargets, int bdr_size, int internal_size,
        MultiVector& localtargets_interior, MultiVector& localtargets_boundary);

    /// Extracts some carefully chosen parts of Projector and returns them in
    /// a MultiVector that's good for handing to CreateDofFunctional()
    ///
    /// Shared code from all the hEntityExtension() functions, though the
    /// calling sequence for this function is almost as long as the code it
    /// replaces (some day some of these parameters will live in an object?)
    std::unique_ptr<MultiVector> BuildCochainProjector(
        int nrhs_RangeT, int nrhs_Null,
        const mfem::Array<int>& coarseUDof_InternalRangeT,
        const mfem::Array<int>& coarseUDof_InternalNull,
        mfem::SparseMatrix& Projector,
        const mfem::Array<int>& fineUDof_Internal,
        int usize) const;

    /// more shared code from hEntityExtension
    std::unique_ptr<mfem::DenseMatrix> CoarsenMassMatrixPart(
        int nlocalbasis, const mfem::Array<int>& coarseUDof_on_Bdr,
        const mfem::Array<int>& coarseUDof_InternalRangeT,
        const mfem::Array<int>& coarseUDof_InternalNull,
        mfem::SparseMatrix& Projector, const mfem::Array<int>& fineUDof,
        const mfem::SparseMatrix& M_aa) const;

    /// some ELAG_DEBUG code shared between hEntityExtension routines
    void CheckLocalExactnessCommuting(
        int jform, AgglomeratedTopology::Entity codim_dom);

    /// collect data for the C block in hRidgeExtension, hPeakExtension
    /// (after combining them this routine is less attractive)
    std::unique_ptr<mfem::SparseMatrix> GetMinusC(
        int jform, AgglomeratedTopology::Entity codim_dom);
    void hFacetExtension(int jform);
    void hRidgePeakExtension(
        int jform, const AgglomeratedTopology::Entity codim_dom);

    virtual void computePVTraces(AgglomeratedTopology::Entity icodim,
                                 mfem::Vector & PVinAgg) = 0;

    std::unique_ptr<mfem::SparseMatrix> getUnextendedInterpolator(int jform);

    /// Distributes \b localtargets(0, jform) among all other \a
    /// jform "codimensions". That is, fills in the rest of the
    /// "form" \a jform column of \b LocalTargets_.
    void populateLowerCodims(const int jform);

    void populateLowerCodimsMultivec(const int jform);

    /// Computes gradients of \b localtargets(0, 0) on the
    /// agglomerated elements and fills in \b localtargets(0, 1). See
    /// \b PopulateLocalTargetsFromForm.
    ///
    /// \warning Currently only implemented for jform == 1, ie,
    /// derivatives from H1 being put IN H(curl)
    void targetDerivativesInForm(const int jform);

    /// The MPI communicator across which this DeRhamSequence is
    /// distributed.
    MPI_Comm Comm_;

    // This is the first form that is supposed to be used. That is, nforms
    // should always equal the correct number of forms for the *DIMENSION*
    // (4 for 3D, 3 for 2D), and then this should specify the first element of
    // the sequence the user wants (i.e. in 3D, jformStart=2 implies HDIV
    // and L2 only, jformStart=1 implies HCURL, HDIV, and L2, and
    // jformStart=0 implies all spaces). THIS NEEDS TO BE CLARIFIED
    // BOTH IN CODE AND IN DOCUMENTATION.
    int jformStart_;

    // FIXME (trb 12/14/15): std::shared_ptr? No Sequence without a
    // Topology; Topology without Sequence ok.
    std::shared_ptr<AgglomeratedTopology> Topo_;

    /// Number of Spaces Involved (4 in 3D, 3 in 2D)
    int nForms_;

    /// DOF Handlers
    std::vector<std::unique_ptr<DofHandler>> Dof_;

    /// DofAgglomeration handlers: dofAgg[i]==NULL until \b
    /// AgglomerateDofs() is called. This usually happens whenever \b
    /// Coarsen() or \b PopulateLocalTargets() is called.
    std::vector<std::unique_ptr<DofAgglomeration>> DofAgg_;

    /// Derivative operators
    std::vector<std::unique_ptr<mfem::SparseMatrix>> D_;

    /// Local Mass Matrices (LOWER TRIANGULAR DATA STRUCTURE)
    std::array<std::unique_ptr<mfem::SparseMatrix>,10> M_;
    // for (codim,form), idx=(dim-form)(nspaces-form)/2 + codim

    /// Global targets
    std::vector<std::unique_ptr<MultiVector>> Targets_;
    // FIXME (trb 12/14/15): This could be a
    // std::array<std::unique_ptr<MultiVector>,4>, probably

    /// Pasciak-Vassilevski traces, apparently
    std::vector<std::unique_ptr<mfem::Vector>> PV_Traces_;
    // FIXME (trb 12/14/15): This could be a
    // std::array<std::unique_ptr<MultiVector>,4>, probably

    /// In general this contains for each "form" and "codimension"
    /// local targets' restrictions on all dofs on each agglomerated
    /// entity.
    ///
    /// NOTE: \b localtargets(0, 0) are (possibly spectral)
    ///    functions defined on each AE on the current level. They
    ///    provide local targets.
    ///
    /// NOTE: This must be done in such a way that for each
    ///    agglomerated entity all possible local targets, that are
    ///    supported on it, are taken into account (collected
    ///    together), i.e. from all "directions", and also the local
    ///    within agglomerated entity numbering must be respected
    ///    (consistent).  Also, the within processor orientation is
    ///    the one that must be used all local targets' dofs.  This
    ///    especially extends when an agglomerated entity falls on the
    ///    boundary of two processors (i.e. it is shared) -- all local
    ///    targets's restrictions on the shared agglomerated entity
    ///    should be collected from all processors having it and all
    ///    other entities sharing it.  Also, all these combined
    ///    "collections" of restrictions need to be distributed among
    ///    all processors that share the agglomerated entity in
    ///    question.  Surely, when distributing, the respective
    ///    processor/local numberings of the dofs within the shared
    ///    agglomerated entity must be respected but they may not be
    ///    the same across processors. Also the within processor dof
    ///    orientation must be respected but they may not be the same
    ///    across processors. See \b PopulateLocalTergetsFromForm.
    std::array<std::vector<std::unique_ptr<MultiVector>>,10> LocalTargets_;
    // for (codim,form), idx=(dim-form)(nspaces-form)/2 + codim

    /*
     * Lower triangular matrix P_t are the matrices that represent the
     * new De-Rham sequence in term of the one on the previous level:
     *
     * I hope you are reading this with monospaced font...
     *
     * In 3D we have:
     *
     *  0 --> H1 --> Hcurl --> Hdiv --> L2 --> 0
     *
     *           0-forms           1-forms            2-forms           3-forms
     * 0-codim [ H1                Hcurl              Hdiv              L2     ]
     * 1-codim [ H1_face_traces    Hcurl_face_traces  Hdiv_face_traces         ]
     * 2-codim [ H1_edge_traces    Hcurl_edge_traces                           ]
     * 3-codim [ H1_vertex_traces                                              ]
     *
     * In 2D we have:
     *
     *  0 --> H1 --> Hcurl --> L2 --> 0
     *
     *           0-forms             1-forms             2-forms
     * 0-codim [ H1                  Hcurl               L2     ]
     * 1-codim [ H1_edges_traces     Hcurl_edges_traces         ]
     * 2-codim [ H1_vertices_traces                             ]
     */

    /// P is the interpolation matrix from the coarser level to this
    std::vector<std::unique_ptr<mfem::SparseMatrix>> P_;

    // FIXME: (trb 12/14/2015): I think it would be good if this went the
    // other way. That is, P should be the interpolation matrix from
    // this level to the finer. With that configuration, if x is a
    // vector represented in this DeRhamSequence, then Px is a valid
    // operation. (In the current configuration, (P^T)x is a valid
    // operation.) This is obviously a larger change since

    /// The cochain projector from this level to the coarser one
    std::vector<std::unique_ptr<CochainProjector>> Pi_;

    mutable std::vector<std::unique_ptr<mfem::HypreParMatrix>> trueP_;

    mutable std::vector<std::unique_ptr<mfem::HypreParMatrix>> truePi_;

    /// Representation of constant one function in L2 (dim-form)
    mfem::Vector L2_const_rep_;

    /// The next coarsest sequence in the hierarchy
    std::weak_ptr<DeRhamSequence> CoarserSequence_;

    /// The next finest sequence in the hierarchy
    std::weak_ptr<DeRhamSequence> FinerSequence_;

    /// Relative tolerance to be used in the SVD
    double SVD_Tolerance_;

    /// The smallest entry allowed in the interpolation
    /// matrix. Potential entries that are smaller will be dropped.
    double SmallestEntry_;
};

/// @class DeRhamSequenceAlg
/**
   @brief Subclass of DeRhamSequence to use if there is no mesh, if
   this is coarsened
*/
class DeRhamSequenceAlg : public DeRhamSequence
{
public:
    DeRhamSequenceAlg(
        const std::shared_ptr<AgglomeratedTopology>& topo,int nspaces);

    virtual ~DeRhamSequenceAlg() = default;

    virtual std::unique_ptr<mfem::SparseMatrix>
    ComputeProjectorFromH1ConformingSpace(int jform) const override;

    virtual void showP(int jform,
                       mfem::SparseMatrix & P,
                       mfem::Array<int> & parts) override;

    virtual void show(int jform,
                      MultiVector & v) override;

    virtual void ShowTrueData(int jform, MultiVector & true_v) override;

    virtual void ExportGLVis(int jform,
                             mfem::Vector & v,
                             std::ostream & os) override;

    virtual DeRhamSequenceFE * FemSequence() override
    {
        return FinerSequence_.lock()->FemSequence();
    }

    virtual void ProjectCoefficient(int jform,
                                    mfem::Coefficient & c,
                                    mfem::Vector & v) override;

    virtual void ProjectVectorCoefficient(int jform,
                                          mfem::VectorCoefficient & c,
                                          mfem::Vector & v) override;

protected:
    void computePVTraces(AgglomeratedTopology::Entity icodim,
                         mfem::Vector & PVinAgg) override;

};
}//namespace parelag
#endif /* DERHAMSEQUENCE_HPP_ */
