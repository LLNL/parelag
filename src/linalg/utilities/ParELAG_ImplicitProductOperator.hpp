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


#ifndef PARELAG_IMPLICITPRODUCTOPERATOR_HPP_
#define PARELAG_IMPLICITPRODUCTOPERATOR_HPP_

#include <deque>

#include <mfem.hpp>

#include "utilities/elagError.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{

/** \class ImplicitProductOperator
 *  \brief Creates, stores, and applies the implicit product of an
 *         arbitrary collection of mfem::Operators.
 *
 *  This class facilitates the application of the action of the
 *  product of multiple operators, \f$y=ABCx\f$, where \f$x,y\f$ are
 *  vectors, when the explicit operator \f$D = ABC\f$ is never needed
 *  (and thus never computed).
 *
 *  \warning No checks are performed by this class as to the validity
 *           of the implicit operator products being computed. It is
 *           expected that if something goes wrong (with dimensions,
 *           vector types, etc), one of the individual operator
 *           classes will catch it. Note that this allows arbitrary
 *           projections to occur within the product so long as the
 *           aligned dimensions are valid.
 *
 *  \todo Test this class!
 */
class ImplicitProductOperator : public mfem::Operator
{
public:

    /** \name Constructors and destructor */
    ///@{

    /** \brief Construct an empty operator */
    ImplicitProductOperator() = default;

    /** \brief Construct from an arbitrary number of operators.
     *
     *  Operators should be passed in the reverse order from the order
     *  in which they should be applied. That is, they should be
     *  passed to match the visual equation \f$y=ABCx\f$. To create
     *  the ImplicitProductOperator for this operation, use this
     *  constuctor as
     *
     *      ImplicitProductOperator Op(A,B,C);
     *
     *  This is to match the mfem::RAPOperator, which takes (R,A,P).
     */
    template <typename... Ts>
    ImplicitProductOperator(
        const std::shared_ptr<mfem::Operator>& op, Ts&&... Args);

    /** \brief Destructor. */
    ~ImplicitProductOperator() {}

    ///@}
    /** \name mfem::Operator interface */
    ///@{

    /** \brief Apply the action of the current product.
     *
     *  \warning \c y will be overwritten rather than added to.
     *
     *  \warning It is assumed that \c mfem::Solvers that are used
     *           will have \c iter_mode appropriately set. If there
     *           are \c mfem::Solver operators with \c iter_mode=true,
     *           extra work will be performed to zero out a vector in
     *           their range space.
     */
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /** \brief Apply the action of the transpose of the current product.
     *
     *  \warning Each operator in the implicit product must also have
     *           MultTranspose() implemented for this to be
     *           successful!
     *
     *  \warning \c y will be overwritten rather than added to.
     *
     *  \warning It is assumed that mfem::Solvers that are used will
     *  have "iter_mode" appropriately set. If there are mfem::Solver
     *  operators with "iter_mode=true", extra work will be performed
     *  to zero out a vector in their range space.
     */
    void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

    ///@}
    /** \name Implicit matrix multiplies */
    ///@{

    /** \brief Implicitly multiply current operator on the left.
     *
     *  \param op The operator to add to the product.
     */
    ImplicitProductOperator& PreMultiply(
        const std::shared_ptr<mfem::Operator>& op);

    /** \brief Implicitly multiply on the left by an arbitrary number
     *         of operators.
     *
     *  Like the constructor, arguments should be passed as one
     *  would write them. If the current matrix is "A" and you want to
     *  pre-multiply by "CB", you would call
     *
     *      ImplicitProductOperator op(A); // A
     *      ...
     *      op->PreMultiply(C,B);          // CBA
     *
     *  \param op An operator to add to the product
     *  \param Args The other operators to add to the product
     */
    template <typename... Ts>
    ImplicitProductOperator& PreMultiply(
        const std::shared_ptr<mfem::Operator>& op, Ts&&... Args);

    /** \brief Implicitly multiply on the right. */
    ImplicitProductOperator& PostMultiply(
        const std::shared_ptr<mfem::Operator>& op);

    /** \brief Implicitly multiply on the right by an arbitrary number
     *         of operators.
     *
     *  Like the constructor, arguments should be passed as one would
     *  write them. If the current matrix is "A" and you want to
     *  post-multiply by "BC", you would call
     *
     *      ImplicitProductOperator op(A); // A
     *      ...
     *      op->PostMultiply(B,C);         // ABC
     *
     */
    template <typename... Ts>
    ImplicitProductOperator& PostMultiply(
        const std::shared_ptr<mfem::Operator>& op, Ts&&... Args);

    ///@}

private:

    /** \brief A list of operators. */
    std::deque<std::shared_ptr<mfem::Operator>> Ops_;

    /** \brief A list of helper vectors. */
    mutable std::deque<mfem::Vector> Vecs_;

};// class ImplicitProductOperator

template <typename... Ts>
std::unique_ptr<mfem::Operator>
IMultiply(Ts&&... Ops)
{
    return make_unique<ImplicitProductOperator>(std::forward<Ts>(Ops)...);
}

//
// Template function definitions
//

template <typename... Ts>
inline ImplicitProductOperator::ImplicitProductOperator(
    const std::shared_ptr<mfem::Operator>& op,
    Ts&&... Args)
    : mfem::Operator{op->Height(),op->Width()}
{
    this->PreMultiply(op,std::forward<Ts>(Args)...);
}

template <typename... Ts>
inline ImplicitProductOperator& ImplicitProductOperator::PreMultiply(
    const std::shared_ptr<mfem::Operator>& op, Ts&&... Args)
{
    PreMultiply(std::forward<Ts>(Args)...);
    return PreMultiply(op);
}

template <typename... Ts>
inline ImplicitProductOperator& ImplicitProductOperator::PostMultiply(
    const std::shared_ptr<mfem::Operator>& op, Ts&&... Args)
{
    Ops_.push_front(op);
    if (Ops_.size() > 1)
        Vecs_.emplace_front(op->Height());
    return PostMultiply(std::forward<Ts>(Args)...);
}

}// namespace parelag
#endif /* PARELAG_IMPLICITPRODUCTOPERATOR_HPP_ */
