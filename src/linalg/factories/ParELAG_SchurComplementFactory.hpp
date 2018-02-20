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


#ifndef PARELAG_SCHURCOMPLEMENTFACTORY_HPP_
#define PARELAG_SCHURCOMPLEMENTFACTORY_HPP_

#include "linalg/solver_core/ParELAG_SolverState.hpp"
#include "linalg/utilities/ParELAG_MfemBlockOperator.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{

/** \class SchurComplementFactory
 *  \brief Factory for creating Schur complements
 *
 *  \todo See BramblePasciakTransformation and
 *        MonolithicBlockedOperatorFactory for other examples of
 *        "Non-solver Operator Factories". Perhaps these should all
 *        have a unified interface and/or base class or I don't know.
 */
class SchurComplementFactory
{
public:
    /** \name Constructors and destructor */
    ///@{

    /** \brief Constructor taking the type of Schur complement and a
     *         scaling factor.
     *
     *  The types are as follows:
     *
     *  String Type  | Definition
     *  ------------ | -----------
     *  "MASS"       | \f$M_{\text{form}}\f$
     *  "DIAGONAL"   | \f$A_{11} - A_{10}\text{diag}(A_{00})^{-1}A_{01}\f$
     *  "ABSROWSUM"  | \f$A_{11} - A_{10}\text{absrowsum}(A_{00})^{-1}A_{01}\f$
     *
     *  \param schur_complement_type The type of Schur complement to construct.
     *  \param scaling A scaling to apply to the diagonal approximation
     *             of \f$A_{00}\f$ (ignored for "MASS").
     */
    SchurComplementFactory(
        std::string schur_complement_type = "mass",
        double scaling = 1.0);

    /** \brief Destructor. */
    ~SchurComplementFactory() = default;

    ///@}
    /** \name Factory functions */
    ///@{

    /** \brief Build an explicit Schur complement matrix. */
    std::unique_ptr<mfem::Operator> BuildOperator(
        MfemBlockOperator& op, SolverState& state) const;

    ///@}

private:

    /** \brief The string identifier for the type of Schur complement. */
    std::string Type_;

    /** \brief Diagonal scaling factor. */
    double Alpha_;

};// class SchurComplementFactory
}// namespace parelag
#endif /* PARELAG_SCHURCOMPLEMENTFACTORY_HPP_ */
