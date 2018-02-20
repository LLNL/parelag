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

#ifndef MLDIVFREE_HPP_
#define MLDIVFREE_HPP_

#include <memory>
#include "amge/DeRhamSequence.hpp"

#include <mfem.hpp>

namespace parelag
{
class MLDivFree final : public mfem::Solver
{
public:
    enum CycleType {FULL_SPACE, DIVFREE_SPACE};

    MLDivFree(const std::shared_ptr<mfem::BlockMatrix>& A,
              std::vector<std::shared_ptr<DeRhamSequence> >& seqs,
              mfem::Array<int> & label_ess);

    void SetTolForPressureRank(double tol)
    {numerical_zero = tol;}

    void SetOperator(const mfem::Operator &op) override;

    void SetBlockMatrix(const std::shared_ptr<mfem::BlockMatrix> A);

    void Build(const mfem::Vector & ess_nullspace_p,
               const mfem::SparseMatrix & mass_p);

    void SetCycle(CycleType c);

    void Mult(const mfem::Vector & x, mfem::Vector & y) const override;
    void Mult(const MultiVector & x, MultiVector & y) const;

    ~MLDivFree() = default;

protected:

    enum { VELOCITY=0, PRESSURE=1};
    void MGVcycle(const mfem::Vector & x, mfem::Vector & y) const;
    void MGVcycleOnlyU(const mfem::Vector & x, mfem::Vector & y) const;

    //subdomainSmoother will always use the value of sol
    void subdomainSmoother(
        int i, const mfem::Vector & rhs, mfem::Vector & sol) const;

    //nullSpaceSmoother will always use the value of sol
    void nullSpaceSmoother(
        int i, const mfem::Vector & rhs, mfem::Vector & sol) const;

    //coarse solver
    void coarseSolver(const mfem::Vector & rhs, mfem::Vector & sol) const;

    void computeSharedDof(int ilevel, mfem::Array<int> & is_shared);

    int getLocalInternalDofs(
        int ilevel, int iAE, mfem::Array<int> & loc_dof) const;

    int getLocalInternalDofs(
        int comp, int ilevel, int iAE, mfem::Array<int> & loc_dof) const;

    int getLocalDofs(
        int comp, int ilevel, int iAE, mfem::Array<int> & loc_dof) const;

    bool isRankDeficient(
        const mfem::SparseMatrix & Bt, const mfem::Vector & x) const;

    std::unique_ptr<mfem::BlockMatrix> PtAP(
        const mfem::BlockMatrix & A,const mfem::BlockMatrix & P) const;

    // sequence
    // mfem::Array<DeRhamSequence *> sequence;
    std::vector<std::shared_ptr<DeRhamSequence> >& sequence_;

    int l2form_;
    int hdivform_;
    int hcurlform_;
    //Number of levels (set by P.Size()+1)
    int nLevels_;
    //Arithmetic TrueDof Complexitity
    int arithmeticTrueComplexity_;
    //Arithmetic Dof Complexitity
    int arithmeticComplexity_;

    std::vector<std::shared_ptr<mfem::BlockMatrix> > Al_;

    std::vector<std::unique_ptr<mfem::BlockOperator>> trueAl_;
    std::vector<std::unique_ptr<mfem::HypreParMatrix>> Cl_;

    std::vector<std::unique_ptr<mfem::Solver>> Maux_;

    // Interpolators
    std::vector<std::unique_ptr<mfem::BlockMatrix>> P_;
    std::vector<std::unique_ptr<mfem::BlockOperator>> trueP_;

    std::vector<std::unique_ptr<mfem::SparseMatrix>> P00_to_be_deleted_;

    // Cochain Projectors
    std::vector<std::unique_ptr<mfem::BlockMatrix>> Pi_;

    //For each level, excluded the coarsest
    std::vector<std::unique_ptr<mfem::BlockMatrix>> AE_dof;

    //We assume that in *_data the dofs for all levels are strided one after the other (from fine to coarse);
    // levelTrueStart is an array of length nLevels+1 such that
    // levelTrueStart[i] is such that *_data[levelTrueStart[i]] point to the first dofs at level i.
    mfem::Array<int> levelTrueStart;
    // Multiplier Stuff at level i goes from levelTrueStartMultiplier[i] to levelTrueStart[i+1].
    // Primal variable stuff at level i goes from levelTrueStart[i] to levelTrueStartMultiplier[i].
    mfem::Array<int> levelTrueStartMultiplier;
    //We assume that in *_data the dofs for all levels are strided one after the other (from fine to coarse);
    // levelStart is an array of length nLevels+1 such that
    // levelStart[i] is such that *_data[levelStart[i]] point to the first dofs at level i.
    mfem::Array<int> levelStart;
    // Multiplier Stuff at level i goes from levelStartMultiplier[i] to levelStart[i+1].
    // Primal variable stuff at level i goes from levelStart[i] to levelTrueMultiplier[i].
    mfem::Array<int> levelStartMultiplier;
    // 0 if the dof belongs to only one AE, 1 if it belongs to multiple arrays. Uses LevelStart
    mfem::Array<int> dof_is_shared_among_AE_data;
    // rhs (both u and p) for the multilevel reconstruction of \hat{u} (solve div equation) and p or just the auxiliary variable
    // Uses LevelTrueStart
    mutable mfem::Array<double> trueRhs_data;
    // sol (both u and p) for the multilevel reconstruction of \hat{u} (solve div equation) and p or just the auxiliary variable
    // Uses LevelTrueStart
    mutable mfem::Array<double> trueSol_data;
    // For the subdomain smoother
    mutable mfem::Array<double> rhs_data;
    mutable mfem::Array<double> sol_data;
    // Auxiliary workspace of local size
    // Uses levelStart and levelTrueStart
    mfem::Array<double> essnullspace_data;
    // Uses levelStart and levelTrueStart
    mfem::Array<double> t_data;
    //! If \| Bt_loc * ess_nullspace_loc \|_inf < numerical_zero then we consider A_loc singular
    double numerical_zero;

    CycleType my_cycle;

};

}//namespace parelag
#endif /* MLDIVFREE_HPP_ */
