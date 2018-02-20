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

#ifndef HYPREEXTENSION_HPP_
#define HYPREEXTENSION_HPP_

// hypre header files
#include "_hypre_utilities.h"

#include <mfem.hpp>

namespace parelag
{
class DeRhamSequence;

namespace HypreExtension
{

class HypreBoomerAMGData
{
public:
    HypreBoomerAMGData();
    int coarsen_type;
    int agg_levels;
    int relax_type;
    int relax_sweeps;
    double theta;
    int interp_type;
    int Pmax;
    int print_level;
    /** More robust options for systems, such as elastisity. Note that BoomerAMG
        assumes Ordering::byVDIM in the finite element space used to generate the
        matrix A. */
    int dim;
    int maxLevels;
};


/// The BoomerAMG solver in hypre
class HypreBoomerAMG : public mfem::HypreSolver
{
private:
    HYPRE_Solver amg_precond;

public:
    HypreBoomerAMG(mfem::HypreParMatrix &A, const HypreBoomerAMGData & data);
    HypreBoomerAMG(mfem::HypreParMatrix &A);

    HypreBoomerAMG(HypreBoomerAMG const&) = delete;
    HypreBoomerAMG(HypreBoomerAMG&&) = delete;

    HypreBoomerAMG& operator=(HypreBoomerAMG const&) = delete;
    HypreBoomerAMG& operator=(HypreBoomerAMG&&) = delete;

    void SetParameters(const HypreBoomerAMGData & data);

    /// The typecast to HYPRE_Solver returns the internal amg_precond
    virtual operator HYPRE_Solver() const { return amg_precond; }

    virtual HYPRE_PtrToParSolverFcn SetupFcn() const
    { return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup; }
    virtual HYPRE_PtrToParSolverFcn SolveFcn() const
    { return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve; }

    virtual ~HypreBoomerAMG();
};

class HypreAMSData
{
public:
    HypreAMSData();
    int cycle_type;
    int rlx_type;
    int rlx_sweeps;
    int rlx_weight;
    int rlx_omega;
    // Data for multigrid on Pi^T A Pi
    HypreBoomerAMGData dataAlpha;
    // Data for multigrid on G^T A G
    HypreBoomerAMGData dataBeta;
    int beta_is_zero;
};


/// The Auxiliary-space Maxwell Solver in hypre
class HypreAMS : public mfem::HypreSolver
{
public:

    int owns_G;
    int owns_Pi;
    int owns_Pixyz;

    HypreAMS(mfem::HypreParMatrix &A, mfem::HypreParMatrix &G, mfem::Array<mfem::HypreParMatrix *> & Pi);
    HypreAMS(mfem::HypreParMatrix &A, mfem::HypreParMatrix & G_, mfem::Array<mfem::HypreParMatrix *> &  Pi_, const HypreAMSData & data);
// Assume Pi is ordered by VDIM
    HypreAMS(mfem::HypreParMatrix &A, mfem::HypreParMatrix &G, mfem::HypreParMatrix &Pi);
    HypreAMS(mfem::HypreParMatrix &A, mfem::HypreParMatrix & G_, mfem::HypreParMatrix & Pi_, const HypreAMSData & data);
    HypreAMS(mfem::HypreParMatrix &A, DeRhamSequence * seq, const HypreAMSData & data);

    HypreAMS(HypreAMS const&) = delete;
    HypreAMS(HypreAMS&&) = delete;

    HypreAMS& operator=(HypreAMS const&) = delete;
    HypreAMS& operator=(HypreAMS&&) = delete;

    void SetParameters( const HypreAMSData & data );

    /// The typecast to HYPRE_Solver returns the internal ams object
    virtual operator HYPRE_Solver() const { return ams; }

    virtual HYPRE_PtrToParSolverFcn SetupFcn() const
    { return (HYPRE_PtrToParSolverFcn) HYPRE_AMSSetup; }
    virtual HYPRE_PtrToParSolverFcn SolveFcn() const
    { return (HYPRE_PtrToParSolverFcn) HYPRE_AMSSolve; }

    virtual ~HypreAMS();

private:

    HYPRE_Solver ams;

    /// Discrete gradient matrix
    mfem::HypreParMatrix *G;
    /// Nedelec interpolation matrix and its components
    mfem::HypreParMatrix *Pi, *Pix, *Piy, *Piz;

    int dim;
};


class HypreADSData
{
public:
    HypreADSData();

    int cycle_type;
    int rlx_type;
    int rlx_sweeps;
    double rlx_weight;
    double rlx_omega;

    //Data for Multigrid on Pi^T A Pi
    HypreBoomerAMGData dataAMG;
    //Data for Multigrid on C^T A C
    HypreAMSData dataAMS;


};

/// The Auxiliary-space Divergence Solver in hypre
class HypreADS : public mfem::HypreSolver
{
private:
    HYPRE_Solver ads;

    /// Discrete gradient matrix
    mfem::HypreParMatrix *G;
    /// Discrete curl matrix
    mfem::HypreParMatrix *C;
    /// Nedelec interpolation matrix and its components
    mfem::HypreParMatrix *ND_Pi, *ND_Pix, *ND_Piy, *ND_Piz;
    /// Raviart-Thomas interpolation matrix and its components
    mfem::HypreParMatrix *RT_Pi, *RT_Pix, *RT_Piy, *RT_Piz;

public:

    int owns_G;
    int owns_C;
    int owns_Pi;

    HypreADS(mfem::HypreParMatrix &A, mfem::ParFiniteElementSpace *face_fespace, const HypreADSData & data);
    HypreADS(mfem::HypreParMatrix &A, mfem::HypreParMatrix &G, mfem::HypreParMatrix &C, mfem::Array<mfem::HypreParMatrix *> & ND_Pi, mfem::Array<mfem::HypreParMatrix *> & RT_Pi);
    HypreADS(mfem::HypreParMatrix &A, mfem::HypreParMatrix &G, mfem::HypreParMatrix &C, mfem::Array<mfem::HypreParMatrix *> & ND_Pi, mfem::Array<mfem::HypreParMatrix *> & RT_Pi, const HypreADSData & data);
    HypreADS(mfem::HypreParMatrix &A, mfem::HypreParMatrix &G, mfem::HypreParMatrix &C, mfem::HypreParMatrix &ND_Pi, mfem::HypreParMatrix &RT_Pi);
    HypreADS(mfem::HypreParMatrix &A, mfem::HypreParMatrix &G_, mfem::HypreParMatrix &C_, mfem::HypreParMatrix &ND_Pi_, mfem::HypreParMatrix &RT_Pi_, const HypreADSData & data);
    HypreADS(mfem::HypreParMatrix &A, DeRhamSequence * seq, const HypreADSData & data);

    HypreADS(HypreADS const&) = delete;
    HypreADS(HypreADS&&) = delete;

    HypreADS& operator=(HypreADS const&) = delete;
    HypreADS& operator=(HypreADS&&) = delete;

    void SetParameters( const HypreADSData & data );

    /// The typecast to HYPRE_Solver returns the internal ads object
    virtual operator HYPRE_Solver() const { return ads; }

    virtual HYPRE_PtrToParSolverFcn SetupFcn() const
    { return (HYPRE_PtrToParSolverFcn) HYPRE_ADSSetup; }
    virtual HYPRE_PtrToParSolverFcn SolveFcn() const
    { return (HYPRE_PtrToParSolverFcn) HYPRE_ADSSolve; }

    virtual ~HypreADS();
};

mfem::HypreParMatrix * ParAdd(double a, mfem::HypreParMatrix *A, double b, mfem::HypreParMatrix *B);
mfem::HypreParMatrix * RAP(mfem::HypreParMatrix * Rt, mfem::HypreParMatrix * A, mfem::HypreParMatrix * P);
void PrintAsHypreParCSRMatrix(MPI_Comm comm, mfem::SparseMatrix & A, const std::string & fname);
hypre_ParCSRMatrix * DeepCopy(hypre_ParCSRMatrix * in);

// Recall: hypre_ParCSRMatrixFixZeroRows(hypre_ParCSRMatrix * A);
}//namespace HypreExtension
}//namespace parelag
#endif /* HYPREEXTESION_HPP_ */
