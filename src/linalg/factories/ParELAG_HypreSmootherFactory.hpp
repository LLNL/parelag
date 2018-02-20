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


#ifndef PARELAG_HYPRESMOOTHERFACTORY_HPP_
#define PARELAG_HYPRESMOOTHERFACTORY_HPP_

#include <memory>

#include <mfem.hpp>

#include "linalg/solver_core/ParELAG_SolverFactory.hpp"

namespace parelag
{

// Forward declarations
class Level;
class ParameterList;

/** \class HypreSmootherFactory
 *  \brief Creates a HypreSmoother object for a given matrix.
 *
 *  The map from string types to numeric types is given here:
 *
 *  String Type                  | Int Type
 *  ---------------------------- | --------
 *   "Jacobi"                    | 0
 *   "L1 Jacobi"                 | 1
 *   "L1 Gauss-Seidel"           | 2
 *   "Kaczmarz"                  | 3
 *   "L1 Gauss-Seidel Truncated" | 4
 *   "Lumped Jacobi"             | 5
 *   "Gauss-Seidel"              | 6
 *   "Chebyshev"                 | 16
 *   "Taubin"                    | 1001
 *   "FIR"                       | 1002
 *   "Hypre Jacobi"              | 413
 *
 *  The meaningful parameters for this class are:
 *
 *  Name                   | Type   | Default value
 *  ---------------------- | ------ | -----------------
 *   "Type"                | string | "L1 Gauss-Seidel"
 *   "Sweeps"              | int    | 1
 *   "Damping Factor"      | double | 1.0
 *   "Omega"               | double | 1.0
 *   "Cheby Poly Order"    | int    | 2
 *   "Cheby Poly Fraction" | double | 0.3
 *
 *  All parameters are effectively forwarded to created
 *  HyperSmootherWrapper objects.
 */
class HypreSmootherFactory : public SolverFactory
{
public:
    /** \name Constructors. */
    ///@{

    /** \brief Default constructor. */
    HypreSmootherFactory() = default;

    /** \brief Construct from ParameterList
     *
     *  The ParameterList is assumed to have a "Type" parameter defined.
     *
     *  \param params The input ParameterList.
     */
    HypreSmootherFactory(const ParameterList& params)
        : Type_{StringTypeToInt(params.Get<std::string>("Type"))}
    {
        this->SetParameters(params);
    }

    /** \brief Construct by string type
     *
     *  \param type The string version of the type. Set to empty
     *              string to pull from the input ParameterList.
     *  \param params The input ParameterList.
     */
    HypreSmootherFactory(const std::string& type,
                         const ParameterList& params = ParameterList())
        : Type_{StringTypeToInt(type)}
    { this->SetParameters(params); }

    /** \brief Construct by numeric type
     *
     *  \param type The numeric value of the type. Set to "-1" to pull
     *              from the input ParameterList.
     *  \param params The input ParameterList.
     */
    HypreSmootherFactory(int type,
                         const ParameterList& params = ParameterList())
        : Type_{type}
    { this->SetParameters(params); }

    ///@}

private:
    int Type_;

private:

    /** \name Implementation of SolverFactory */
    ///@{

    /** \brief Implementation of BuildSolver(). */
    std::unique_ptr<mfem::Solver> _do_build_solver(
        const std::shared_ptr<mfem::Operator>& op,
        SolverState& state ) const override;

    /** \brief Implementation of SetDefaultParameters(). */
    void _do_set_default_parameters() override;

    /** \brief Implementation of Initialize(). */
    void _do_initialize(const ParameterList& Params) override;

    ///@}

    std::string IntTypeToString(int type) const noexcept;
    int StringTypeToInt(const std::string& type) const noexcept;

};// class HypreSmootherFactory
}// namespace parelag
#endif /* PARELAG_HYPRESMOOTHERFACTORY_HPP_ */
