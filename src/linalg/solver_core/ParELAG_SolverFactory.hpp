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


#ifndef PARELAG_SOLVERFACTORY_HPP_
#define PARELAG_SOLVERFACTORY_HPP_

#include <memory>

#include <mfem.hpp>

#include "ParELAG_SolverState.hpp"

#include "utilities/ParELAG_ParameterList.hpp"

namespace parelag
{

// Forward declarations
class SolverLibrary;


/** \class SolverFactory
 *  \brief An interface for creating more complex solver objects.
 */
class SolverFactory
{
public:

    /** \name Constructors and destructor */
    ///@{

    /** \brief Default constructor */
    SolverFactory()
        : Params_{},
          Lib_{}
    {}

    /** \brief Destructor */
    virtual ~SolverFactory() = default;

    ///@}
    /** \name Main interface functions */
    ///@{

    /** \brief Construct and setup the solver operator.
     *
     *  The Solver object should be fully initialized and ready for use.
     *
     *  \param op     The operator upon which the solver is based.
     *  \param state  Any state associated with this solver.
     */
    std::unique_ptr<mfem::Solver> BuildSolver(
        const std::shared_ptr<mfem::Operator>& op,
        SolverState& state ) const
    {
        return _do_build_solver(op,state);
    }

    /** \brief Get a SolverState of the right type for this Factory
     */
    std::unique_ptr<SolverState> GetDefaultState() const
    {
        return _do_get_default_state();
    }

    ///@}
    /** \name Deal with the solver's parameters. */
    ///@{

    /** \brief Initialize the SolverFactory using the given ParameterList
     *
     *  Use with a default constructor to finish setup of SolverFactories
     *
     *  \param params  The ParameterList for this factory.
     */
    void Initialize(const ParameterList& params)
    {
        SetParameters(params);
        SetDefaultParameters();

        _do_initialize(params);
    }

    /** \brief Sets all parameters for a given SolverFactory to defaults.
     */
    void SetDefaultParameters()
    {
        _do_set_default_parameters();
    }

    /** \brief Merge the input list into the parameter list.
     *
     *  \params params  The parameters to be set.
     */
    void SetParameters(const ParameterList& params)
    {
        Params_.Merge(params);
    }

    /** \brief Get the parameter list for this factory. */
    ParameterList& GetParameters()
    {
        return Params_;
    }

    /** \brief Get the parameter list for this factory (const). */
    ParameterList const& GetParameters() const
    {
        return Params_;
    }

    ///@}
    /** \name Solver Library management */
    ///@{

    /** \brief Get the SolverLibrary associated with this factory.
     *
     *  This is for automating the building of complex nested solvers
     *  using the SolverLibrary interface. If all factories for nested
     *  solvers are provided to the SolverFactory that needs them, the
     *  library will not be referenced and the Lib_ member can safely
     *  be left as nullptr.
     */
    SolverLibrary const& GetSolverLibrary() const
    {
        PARELAG_ASSERT_DEBUG((bool) Lib_);
        return *Lib_;
    }

    /** \brief Set the SolverLibrary associated with this factory.
     *
     *  \param lib  The SolverLibrary object to be used for finding
     *  nested factories.
     */
    void SetSolverLibrary(std::shared_ptr<const SolverLibrary> lib) noexcept
    {
        Lib_ = std::move(lib);
    }

    /** \brief Check that there is a valid solver library object. */
    bool HasValidSolverLibrary() const noexcept
    {
        return (bool) Lib_;
    }

    ///@}

private:

    /** \brief Implementation of BuildSolver(). */
    virtual std::unique_ptr<mfem::Solver> _do_build_solver(
        const std::shared_ptr<mfem::Operator>& op,
        SolverState& state ) const = 0;

    /** \brief Implementation of GetDefaultState(). */
    virtual std::unique_ptr<SolverState> _do_get_default_state() const
    {
        return make_unique<SolverState>();
    }

    /** \brief Implementation of Initialize(). */
    virtual void _do_initialize(const ParameterList& Params) = 0;

    /** \brief Implementation of SetDefaultParameters(). */
    virtual void _do_set_default_parameters() = 0;

private:

    /** \brief The parameter list for this solver factory. */
    ParameterList Params_;

    /** \brief The SolverLibrary used by this Factory. */
    std::shared_ptr<const SolverLibrary> Lib_;

};// class SolverFactory
}// namespace parelag
#endif /* PARELAG_SOLVERFACTORY_HPP_ */
