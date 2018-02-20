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


#ifndef PARELAG_HIPTMAIRSMOOTHER_HPP_
#define PARELAG_HIPTMAIRSMOOTHER_HPP_

#include <mfem.hpp>

#include "linalg/solver_core/ParELAG_Solver.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{

/** \class HiptmairSmoother
 *  \brief Implementation of Hiptmair-style auxiliary space smoothing.
 *
 *  See, for example, Figure 6.2 in \cite RHiptmair_1999a or Algorithm
 *  F.1.1 in \cite PSVassilevski_2008a.
 */
class HiptmairSmoother : public Solver
{
public:
    /** \name Constructor and Destructor */
    ///@{

    /** \brief Constructor that takes all needed operators.
     *
     *  \param A The primary space operator.
     *  \param A_Aux The auxiliary space operator. This is not used
     *               and may be nullptr. It is included in case the
     *               auxiliary smoother object does not maintain the
     *               lifetime of its underlying operator.
     *  \param D_Op The derivative to move from the auxiliary space to
     *              the primary space.
     *  \param PrimarySolver The solver for the primary space.
     *  \param AuxiliarySolver THe solver for the auxiliary space.
     */
    HiptmairSmoother(std::shared_ptr<mfem::Operator> A,
                     std::shared_ptr<mfem::Operator> A_Aux,
                     std::shared_ptr<mfem::Operator> D_Op,
                     std::shared_ptr<mfem::Solver> PrimarySolver,
                     std::shared_ptr<mfem::Solver> AuxiliarySolver);

    /** \brief Destructor. */
    ~HiptmairSmoother() {}

    ///@}
    /** \name mfem::Solver interface */
    ///@{

    /** Applies the Hiptmair smoother in the primary-auxiliary order. */
    void Mult(const mfem::Vector& B, mfem::Vector& X) const override;

    /** Applies the Hiptmair smoother in the auxiliary-primary order. */
    void MultTranspose(const mfem::Vector& B, mfem::Vector& X) const override;

    ///@}

private:

    void _do_set_operator(const std::shared_ptr<mfem::Operator>& op) override
    {
        PARELAG_ASSERT(op);
        A_ = op;
    }

private:

    /** \brief The primary space operator. */
    std::shared_ptr<mfem::Operator> A_;

    /** \brief The auxiliary space operator. */
    std::shared_ptr<mfem::Operator> A_aux_;

    /** \brief The derivative operator. */
    std::shared_ptr<mfem::Operator> D_op_;

    /** \brief The solver in the primary space. */
    std::shared_ptr<mfem::Solver> PrimarySolver_;

    /** \brief The solver in the auxiliary space. */
    std::shared_ptr<mfem::Solver> AuxiliarySolver_;

    /** \name Auxiliary members. */
    ///@{
    /** \brief An auxiliary vector. */
    mutable std::unique_ptr<mfem::Vector> AuxB_;
    mutable std::unique_ptr<mfem::Vector> AuxX_;
    mutable std::unique_ptr<mfem::Vector> PrimaryVec_;
    ///@}

};// class Hiptmair Smoother
}// namespace parelag
#endif /* PARELAG_HIPTMAIRSMOOTHER_HPP_ */
