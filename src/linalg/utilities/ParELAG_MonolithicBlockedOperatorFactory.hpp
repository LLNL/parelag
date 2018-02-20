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


#ifndef PARELAG_MONOLITHICBLOCKEDOPERATORFACTORY_HPP_
#define PARELAG_MONOLITHICBLOCKEDOPERATORFACTORY_HPP_

#include <mfem.hpp>

#include "utilities/MemoryUtils.hpp"

namespace parelag
{

/** \class MonolithicBlockedOperatorFactory
 *  \brief Manages construction of a monolithic mfem::HypreParMatrix
 *         from a blocked operator.
 */
class MonolithicBlockedOperatorFactory
{
    using function_index_type = int;
    using local_index_type = HYPRE_Int;
    using global_index_type = HYPRE_Int;
    using value_type = HYPRE_Int;
public:

    /** \brief Default constructor. */
    MonolithicBlockedOperatorFactory() = default;

    /** \brief Construct and return the monolithic operator.
     *
     *  \tparam BlockedOperatorType May be mfem::BlockOperator or
     *              parelag::MfemBlockOperator.
     *
     *  \param blop The blocked operator in explicit blocked format.
     *
     *  \return The operator as a monolithic matrix.
     *
     *  \warning This will do a deep copy of all of the matrix data.
     */
    template <typename BlockedOperatorType>
    std::unique_ptr<mfem::HypreParMatrix> BuildOperator(
        BlockedOperatorType& blop);

    /** \brief Get a reference to the LOCAL dof-function map.
     *
     *  This is used for constructing certain BoomerAMG preconditioners/solvers.
     */
    std::vector<function_index_type>& GetDofToFunctionMap()
    { return DofToFunction_; }

    /** \brief Get a reference to the LOCAL dof-function map. (const version)
     *
     *  This is used for constructing certain BoomerAMG preconditioners/solvers.
     */
    const std::vector<function_index_type>& GetDofToFunctionMap() const
    { return DofToFunction_; }

private:

    /** \brief The dof to function map. */
    std::vector<function_index_type> DofToFunction_;
    // DofToFunction[idx] = block_id

};// class MonolithicBlockedOperatorFactory
}// namespace parelag
#endif /* PARELAG_MONOLITHICBLOCKEDOPERATORFACTORY_HPP_ */

// NOTE: I don't really know what to do with these "auxiliary
// factories", like "BramblePasciakTransformation",
// "MonolithicBlockedOperatorFactory", and
// "SchurComplementFactory". They kind of fit a generic Builder
// pattern; perhaps refactor to reflect this?
