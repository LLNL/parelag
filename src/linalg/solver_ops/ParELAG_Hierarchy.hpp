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


#ifndef PARELAG_HIERARCHY_HPP_
#define PARELAG_HIERARCHY_HPP_

#include <memory>
#include <vector>

#include "amge/DeRhamSequence.hpp"
#include "linalg/solver_core/ParELAG_Level.hpp"
#include "linalg/solver_core/ParELAG_Solver.hpp"

namespace parelag
{

/** \class Hierarchy
 *  \brief This manages multigrid cycles using pre-defined operators.
 *
 *  This is essentially geometric multigrid; all pieces are expected
 *  to exist prior to creating this object. There is no mechanism in
 *  place for building missing operators on-the-fly (though it would
 *  be possible to add this, if needed). The operators used by this
 *  class are only expected to have an Op-Vec multiply defined (i.e.,
 *  they satisfy mfem::Operator). The operators that are needed are as
 *  follows:
 *
 *     * A (needed on every level): system operator for given level.
 *
 *     * P (needed on every level but the finest): interpolation
 *       operator from level to finer.
 *
 *     * R (needed on every level but the coarsest): restriction
 *       operator from level to coarser.
 *
 *     * Pre-Smoother (any levels): PreSmoothing operator. May be
 *       null.
 *
 *     * Post-Smoother (any levels): PostSmoothing operator. May be
 *       null.
 *
 *     * Coarse Solver (coarsest level): Solver to be called on
 *       coarsest grid system.
 *
 *  A note on these objects. The paradigm is this:
 *
 *     * The only functionality required of any of these operators is
 *       a Operator-Vector product. (Also possibly required is an
 *       Operator-Transposed-Vector product, see rules for \f$P\f$ and
 *       \f$R\f$ below.)
 *
 *     * A consequence of this is that Galerkin coarse-grid operators,
 *       \f$A_c=RAP\f$, cannot be computed on-the-fly. Therefore,
 *       coarse-grid operators must be constructed before Mult() is
 *       called.
 *
 *     * If \f$P\f$ does not exist or is null but \f$R\f$ exists and
 *       is not null, \f$P=R^T\f$ is used.
 *
 *     * If \f$R\f$ does not exist or is null, \f$R=P^T\f$ is used.
 *
 *     * If at least one of \f$P\f$ and \f$R\f$ does not exist or is
 *       not null, an exception is thrown.
 *
 *     * Due to the above rules, if \f$P\f$ is set on the finest level
 *       but \f$R\f$ is missing or null, \f$R=P^T\f$ will be used for
 *       consistency. Likewise, if \f$R\f$ is set on the coarsest
 *       level but \f$P\f$ is missing or null, \f$P=R^T\f$ will be
 *       used.
 *
 *     * If the Pre-Smoother doesn't exist or is null, then there is
 *       no pre-smoothing.
 *
 *     * If the Post-Smoother doesn't exist or is null, then there is
 *       no post-smoothing.
 *
 *     * If, on the coarsest level, CoarseSolver does not exist or is
 *       null, then the PreSmoother (if any) is called, followed by
 *       the PostSmoother (if any). If a CoarseSolver is found, no
 *       smoothers are searched for.
 *
 *     * If, on the coarsest level, none of CoarseSolver, PreSmoother,
 *       and PostSmoother are valid, nothing will happen on that level
 *       and a correction of zero will be interpolated to next finer
 *       level. This has no significance and is the result of the
 *       above rules.
 *
 *     * The CoarseSolver is only checked on the coarsest level;
 *       setting it on any other level has no effect.
 *
 *  This class can currently handle "\f$\mu\f$-cycles", where
 *  \f$\mu=0\f$ corresponds to a V-cycle and \f$\mu=1\f$ to a
 *  W-cycle.
 *
 *  \todo There are plans to extend to a so-called k-cycle, in which
 *  the cycle can be changed on a per-level basis. Additionally, there
 *  are plans to extend to full multigrid.
 *
 *  \todo Cleanup memory usage, re auxiliary vectors.
 */
class Hierarchy : public Solver
{
    using level_ptr = std::shared_ptr<Level>;
    using iterator = std::vector<level_ptr>::iterator;
    using const_iterator = std::vector<level_ptr>::const_iterator;

public:

    /** \name Constructor and destructor */
    ///@{

    /** \brief Creates a hierarchy with a given number of levels.
     *
     *  The levels are empty, except the finest, to which A is
     *  attached. The finest level has ID 0.
     *
     *  \param A The fine-grid system operator.
     *  \param NumLevels The number of levels to create in the
     *         hierarchy.
     */
    Hierarchy(std::shared_ptr<mfem::Operator> A, int NumLevels = 1);

    /** \brief Destructor. */
    ~Hierarchy() = default;

    ///@}

    /** \name Deleted special functions. */
    ///@{
    /** \brief All special constructors and assignment deleted. */
    Hierarchy() = delete;
    Hierarchy(Hierarchy const&) = delete;
    Hierarchy(Hierarchy&&) = delete;
    Hierarchy& operator=(Hierarchy const&) = delete;
    Hierarchy& operator=(Hierarchy&&) = delete;
    ///@}

    /** \name Getter functions */
    ///@{

    /** \brief Get the number of levels in this hierarchy.
     *
     *  \return The number of levels in this hierarchy.
     *
     *  \todo Fix return type.
     */
    int GetNumLevels() const noexcept { return Levels_.size(); }

    /** \brief Get a level in the hierarchy by ID.
     *
     *  \throws std::out_of_range if index >= GetNumLevels.
     *  \throws std::runtime_error if Levels_[index] is null.
     *
     *  \param index The ID of the level.
     *
     *  \return A reference to the level with ID \c index.
     *
     *  \todo Fix parameter type.
     */
    Level& GetLevel(int index);

    ///@}
    /** \name Setter functions */
    ///@{

    /** \brief Set flag to use implicit transpose by default.
     *
     *  \param UseImplicit \c true will cause \f$R=P^{T}\f$.
     */
    void SetImplicitTranspose(bool UseImplicit)
    { ImplicitTranspose_ = UseImplicit; }


    /** \brief Set the cycle on every level.
     *
     *  Setting mu=0 is a V-cycle; setting mu=1 is a W-cycle.
     *
     *  \note This will not affect fine grid behavior. The cycle is
     *        essentially always 0 on the fine grid. Use a different
     *        number of iterations in the call to Iterate() to change
     *        fine-grid cycling.
     *
     *  \param Mu The cycle to be applied at every level.
     */
    void SetCycle(int Mu);


    /** \brief Set the cycle on a particular level.
     *
     *  Choosing Mu=-1 will skip that level (no smoothing, just
     *  restriction and prolongation).
     *
     *  \note Changing Mu on LevelID=0 will have no effect. Use a
     *        different number of iterations in the call to Iterate()
     *        to change fine-grid cycling.
     *
     *  \param Mu The cycle to applied to a level.
     *  \param LevelID The ID of the level whose cycle is being set.
     */
    void SetCycle(int Mu, int LevelID);


    /** \brief Set the cycle for all levels.
     *
     *  Level i will cycle \c Mus[i] times. Choosing \c Mus[i]=-1 will
     *  skip that level (no smoothing, just restriction and
     *  prolongation).
     *
     *  \note The value of \c Mus[0] will have no effect. Use a
     *        different number of iterations in the call to Iterate()
     *        to change fine-grid cycling.
     *
     *  \param Mus The cycle to applied to each level. It is expected
     *             that \c Mus.size() = GetNumLevels().
     */
    void SetCycle(std::vector<int> Mus);

    ///@}
    /** \name mfem::Operator interface */
    ///@{

    /** \brief Apply the MG operator to a vector. */
    void Mult(mfem::Vector const& rhs, mfem::Vector& sol) const override;

    ///@}
    /** \name The multigrid-specific functions */
    ///@{

    /** \brief The actual recursive MG call.
     *
     *  \param RHS The current level RHS or residual.
     *  \param SOL The current level solution.
     *  \param StartLevel The ID of the current level.
     *  \param NumIterations The number of times to run the iteration.
     *  \param CycleThisLevel Whether or not multiple cycles can run
     *                        from this level. This prevents, for
     *                        example, W-cycles from running "2 Ws"
     *                        from the fine grid.
     */
    void Iterate(const mfem::Vector& RHS, mfem::Vector& SOL,
                 int StartLevel = 0, int NumIterations = 1,
                 bool CycleThisLevel = false) const;

    /** \brief Ensures the hierarchy is ready for use and memory is
     *         pre-allocated.
     */
    void Finalize();

    ///@}
    /** \name STL compatibility */
    ///@{

    /** \brief Get an iterator to the finest level. */
    iterator begin() noexcept { return Levels_.begin(); }

    /** \brief Get an iterator to "one-past-coarsest" level. */
    iterator end() noexcept { return Levels_.end(); }

    /** \brief Get a const iterator to the finest level. */
    const_iterator begin() const noexcept { return Levels_.begin(); }

    /** \brief Get a const iterator to "one-past-coarsest" level. */
    const_iterator end() const noexcept { return Levels_.end(); }

    ///@}

private:

    void _do_set_operator(const std::shared_ptr<mfem::Operator> &op) override;

private:

    /** \brief The collection of levels in the hierarchy. */
    std::vector<level_ptr> Levels_;

    /** \brief The cycle strategy for each level. */
    std::vector<int> CycleMu_;

    /** \brief \c true if we should use \f$R=P^T\f$. */
    bool ImplicitTranspose_;

    /** \name Auxiliary vectors */
    ///@{
    /** \brief Auxiliary vectors. */
    mutable mfem::Vector tmp_resid_, tmp_correct_;

    mutable std::vector<mfem::Vector> CoarseResids_, CoarseSols_;
    ///@}

};// class Hierarchy


/** \brief Non-member constructor that builds a Hierarchy object from
 *         a hierarchy of DeRhamSeqences.
 *
 *  Builds a hierarchy corresponding to the given \c form. This sets P
 *  and A for each level. Thus, it computes \f$A_C = P^T AP\f$ for
 *  each level.
 *
 *  Pass MaxNumLevels=-1 to build a complete hierarchy
 *  (until there is no CoarserSequence).
 *
 *  \param A The fine-grid operator.
 *  \param Sequence The fine-grid DeRhamSequence object.
 *  \param label_ess The list of essential boundary labels.
 *  \param form The form to which A corresponds.
 *  \param MaxNumLevels The number of levels to build.
 *
 *  \return A Hierarchy with no smoothers or coarse-grid solvers, but
 *          all coarse operators and interpolation operators set.
 */
std::unique_ptr<Hierarchy> buildHierarchyFromDeRhamSequence(
    const std::shared_ptr<mfem::Operator>& A,
    const DeRhamSequence& Sequence,
    std::vector<int> & label_ess,
    int form,
    int MaxNumLevels);

/** \brief Non-member constructor that builds a Hierarchy object from
 *         a hierarchy of DeRhamSequences using BlockedOperators.
 *
 *  This constucts a "monolithic multigrid" hierarchy for an operator
 *  that is expressed as
 *
 *  \f[
 *     A = \begin{bmatrix}
 *        A_{00} & \dots  & A_{0n} \\
 *        \vdots & \ddots & \vdots \\
 *        A_{n0} & \dots  & A_{nn}
 *     \end{bmatrix},
 *  \f]
 *
 *  where each block row and column corresponds to a given form. This
 *  correspondence is assumed to be square (row i corresponds to the
 *  same form as column i, so that diagonal blocks are square).
 *
 *  \param A The fine-grid system operator.
 *  \param Sequence The fine-grid DeRhamSequence object.
 *  \param label_ess The list of lists of essential boundary labels
 *                   for each form.
 *  \param forms The ordered list of forms corresponding to A.  The
 *               "forms" vector should have size equal to number of
 *               block rows/columns and each entry is the form to
 *               which that diagonal block corresponds. That is, if
 *               solving an Hdiv-L2-Hcurl-H1 system, "forms" would be
 *               {2, 3, 1, 0}.
 *  \param MaxNumLevels The number of levels to build.
 *
 *  \return A Hierarchy with no smoothers or coarse-grid solvers, but
 *          all coarse operators and interpolation operators set.
 */
std::unique_ptr<Hierarchy> buildBlockedHierarchyFromDeRhamSequence(
    const std::shared_ptr<mfem::Operator>& A,
    const DeRhamSequence& Sequence,
    std::vector<std::vector<int>>& label_ess,
    const std::vector<int>& forms,
    int MaxNumLevels=-1);

}// namespace parelag
#endif /* PARELAG_HIERARCHY_HPP_ */
