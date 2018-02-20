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


#ifndef PARELAG_HIPTMAIRSMOOTHERFACTORY_HPP_
#define PARELAG_HIPTMAIRSMOOTHERFACTORY_HPP_

#include <memory>

#include <mfem.hpp>

#include "linalg/solver_core/ParELAG_SolverFactory.hpp"

namespace parelag
{

/** \class HiptmairSmootherFactory
 *  \brief SolverFactory for creating HiptmairSmoother objects.
 */
class HiptmairSmootherFactory : public SolverFactory
{
public:

    /** \name Constructors and destructor */
    ///@{

    /** \brief Construct the factory with a parameter list.
     *
     *  \param params The list of parameters for the factory.
     */
    HiptmairSmootherFactory(
        ParameterList const& params = ParameterList())
    { SetParameters(params); }

    /** \brief Construct from another SolverFactory.
     *
     *  \param SFact The SolverFactory that will be used to constuct
     *         both the primary and the auxiliary subsolvers.
     *  \param params The list of parameters for the factory.
     */
    HiptmairSmootherFactory(
        std::shared_ptr<SolverFactory> SmootherFact,
        const ParameterList& params = ParameterList())
        : PrimaryFact_{std::move(SmootherFact)},
          AuxiliaryFact_{PrimaryFact_}
    { SetParameters(params); }

    /** \brief Construct from two solver factories.
     *
     *  \param PrimarySolverFactory The SolverFactory for building a
     *         Solver in the Primary Space.
     *  \param AuxiliarySolverFactory THe SolverFactory for building a
     *         Solver in the Auxiliary Space.
     */
    HiptmairSmootherFactory(
        std::shared_ptr<SolverFactory> PrimarySolverFactory,
        std::shared_ptr<SolverFactory> AuxiliarySolverFactory,
        const ParameterList& params = ParameterList())
        : PrimaryFact_{std::move(PrimarySolverFactory)},
          AuxiliaryFact_{std::move(AuxiliarySolverFactory)}
    { SetParameters(params); }

    /** \brief Destructor. */
    ~HiptmairSmootherFactory() {}

    ///@}
    /** \name Nested preconditioning */
    ///@{

    /** \brief Set the SolverFactory to be used in the primary space. */
    void SetPrimaryFactory(std::shared_ptr<SolverFactory> PrimaryFact) noexcept
    {
        PrimaryFact_ = std::move(PrimaryFact);
    }

    /** \brief Set the SolverFactory to be used in the auxiliary space. */
    void SetAuxiliaryFactory(std::shared_ptr<SolverFactory> AuxFact) noexcept
    {
        AuxiliaryFact_ = std::move(AuxFact);
    }

    /** \brief Set the SolverFactoriess to be used for the primary and
     *         auxiliary spaces together. */
    void SetFactories(std::shared_ptr<SolverFactory> PrimaryFact,
                      std::shared_ptr<SolverFactory> AuxFact) noexcept
    {
        SetPrimaryFactory(std::move(PrimaryFact));
        SetAuxiliaryFactory(std::move(AuxFact));
    }

    /** \brief Get the SolverFactory for the primary space. */
    std::shared_ptr<SolverFactory> GetPrimarySmootherFactory() const noexcept
    {
        return PrimaryFact_;
    }

    /** \brief Get the SolverFactory for the auxiliary space. */
    std::shared_ptr<SolverFactory> GetAuxiliarySmootherFactory() const noexcept
    {
        return AuxiliaryFact_;
    }

    ///@}

private:

    /** \brief Computes \f$A_{aux} = D^T A D\f$. */
    std::unique_ptr<mfem::Operator> _do_compute_aux_operator(
        mfem::Operator& A, mfem::Operator& D ) const;

    /** \name SmootherFactory Interface */
    ///@{

    std::unique_ptr<mfem::Solver> _do_build_solver(
        const std::shared_ptr<mfem::Operator>& op,
        SolverState& state ) const override;

    void _do_set_default_parameters() override;

    void _do_initialize(const ParameterList& Params) override;

    std::unique_ptr<SolverState> _do_get_default_state() const override
    { return make_unique<NestedSolverState>(); }

    ///@}

private:

    /** \brief The SolverFactory for the primary space. */
    std::shared_ptr<SolverFactory> PrimaryFact_;

    /** \brief The SolverFactory for the primary space. */
    std::shared_ptr<SolverFactory> AuxiliaryFact_;

};// class HiptmairSmootherFactory
}// namespace parelag
#endif /* PARELAG_HIPTMAIRSMOOTHERFACTORY_HPP_ */
