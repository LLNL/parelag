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


#ifndef PARELAG_BRAMBLEPASCIAKSOLVER_HPP_
#define PARELAG_BRAMBLEPASCIAKSOLVER_HPP_

#include "linalg/solver_core/ParELAG_Solver.hpp"

namespace parelag
{

/** \class BramblePasciakSolver
 *  \brief A wrapper around the solver for a Bramble-Pasciak
 *         transformed system.
 *
 *  We need to transform the "real" right-hand side with the
 *  transformation matrix as well. This class facilitates that step.
 */
class BramblePasciakSolver : public Solver
{
public:

    /** \name Constructors and destructor */
    ///@{

    /** \brief Constructor taking the transformation operator and the
     *         solver object to be used for solving the *transformed*
     *         problem.
     *
     *  \param transform_op The transformation operator, presumably
     *                      from BramblePasciakTransformation. Used to
     *                      transform the system right-hand side prior
     *                      to solve.
     *  \param solver The solver for the transformed system.
     */
    BramblePasciakSolver(std::shared_ptr<mfem::Operator> transform_op,
                         std::shared_ptr<mfem::Solver> solver);

    /** \brief Destructor. */
    ~BramblePasciakSolver() = default;

    ///@}
    /** \name Solver interface */
    ///@{

    /** \brief Apply the solver to the input vector
     *
     *  Apply the transformation to the input RHS and call Mult() on
     *  the solver object.
     */
    void Mult(mfem::Vector const& rhs, mfem::Vector& sol) const override;

    /** \brief Apply the transpose of the solver to the input vector
     *
     *  Apply the transpose of the transformation to the input RHS and
     *  call MultTranspose() on the solver object.
     */
    void MultTranspose(
        mfem::Vector const& rhs, mfem::Vector& sol) const override;

    ///@}
    /** \name Extra methods */
    ///@{

    /** \brief Set the transformation operator.
     *
     *  \param transform_op The Operator used to transform the system.
     */
    void SetTransformationOperator(std::shared_ptr<mfem::Operator> transform_op);

    /** \brief Get the transformation operator for this solver.
     *
     *  \return The transformation operator for this solver. */
    std::shared_ptr<mfem::Operator> GetTransformationOperator() const noexcept;

    ///@}

private:

    /** \name The ParELAG::Solver Interface */
    ///@{

    /** \brief Set the underlying operator.
     *
     *  Setting the operator will forward the operator into the Solver_
     *
     *  \warning If the transformation has changed, it must be changed
     *           by calling the SetTransformation() method as
     *           well. This function has no effect on the
     *           transformation, which will still be applied as usual
     *           during Mult().
     */
    void _do_set_operator(const std::shared_ptr<mfem::Operator>& op) override;

    ///@}

private:

    /** \brief The transformation operator */
    std::shared_ptr<mfem::Operator> TransformationOp_;

    /** \brief The solver for the transformed system */
    std::shared_ptr<mfem::Solver> Solver_;

    /** \brief Helper vector to store the transformed RHS */
    mutable mfem::Vector TransformedRHS_;

};// class BramblePasciakSolver

/** \class BramblePasciakSolver
 *  \todo This class really represents a "transforming solver" and
 *        further abstraction could be considered for this and perhaps
 *        Hybridization as well. Hybridization would require a
 *        solution transformation back to the "real solution", whereas
 *        this solver does not require that step as the transformed
 *        solution will have the same solution as the non-transformed
 *        problem.
 */

}// namespace parelag
#endif /* PARELAG_BRAMBLEPASCIAKSOLVER_HPP_ */
