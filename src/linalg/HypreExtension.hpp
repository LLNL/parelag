/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
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
class HypreBoomerAMG : public HypreSolver
{
private:
   HYPRE_Solver amg_precond;

public:
   HypreBoomerAMG(HypreParMatrix &A, const HypreBoomerAMGData & data);
   HypreBoomerAMG(HypreParMatrix &A);
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
class HypreAMS : public HypreSolver
{
public:

   int owns_G;
   int owns_Pi;
   int owns_Pixyz;

   HypreAMS(HypreParMatrix &A, HypreParMatrix &G, Array<HypreParMatrix *> & Pi);
   HypreAMS(HypreParMatrix &A, HypreParMatrix & G_, Array<HypreParMatrix *> &  Pi_, const HypreAMSData & data);
   // Assume Pi is ordered by VDIM
   HypreAMS(HypreParMatrix &A, HypreParMatrix &G, HypreParMatrix &Pi);
   HypreAMS(HypreParMatrix &A, HypreParMatrix & G_, HypreParMatrix & Pi_, const HypreAMSData & data);
   HypreAMS(HypreParMatrix &A, DeRhamSequence * seq, const HypreAMSData & data);


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
   HypreParMatrix *G;
   /// Nedelec interpolation matrix and its components
   HypreParMatrix *Pi, *Pix, *Piy, *Piz;

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
class HypreADS : public HypreSolver
{
private:
   HYPRE_Solver ads;

   /// Discrete gradient matrix
   HypreParMatrix *G;
   /// Discrete curl matrix
   HypreParMatrix *C;
   /// Nedelec interpolation matrix and its components
   HypreParMatrix *ND_Pi, *ND_Pix, *ND_Piy, *ND_Piz;
   /// Raviart-Thomas interpolation matrix and its components
   HypreParMatrix *RT_Pi, *RT_Pix, *RT_Piy, *RT_Piz;

public:

   int owns_G;
   int owns_C;
   int owns_Pi;

   HypreADS(HypreParMatrix &A, ParFiniteElementSpace *face_fespace, const HypreADSData & data);
   HypreADS(HypreParMatrix &A, HypreParMatrix &G, HypreParMatrix &C, Array<HypreParMatrix *> & ND_Pi, Array<HypreParMatrix *> & RT_Pi);
   HypreADS(HypreParMatrix &A, HypreParMatrix &G, HypreParMatrix &C, Array<HypreParMatrix *> & ND_Pi, Array<HypreParMatrix *> & RT_Pi, const HypreADSData & data);
   HypreADS(HypreParMatrix &A, HypreParMatrix &G, HypreParMatrix &C, HypreParMatrix &ND_Pi, HypreParMatrix &RT_Pi);
   HypreADS(HypreParMatrix &A, HypreParMatrix &G_, HypreParMatrix &C_, HypreParMatrix &ND_Pi_, HypreParMatrix &RT_Pi_, const HypreADSData & data);
   HypreADS(HypreParMatrix &A, DeRhamSequence * seq, const HypreADSData & data);
   void SetParameters( const HypreADSData & data );

   /// The typecast to HYPRE_Solver returns the internal ads object
   virtual operator HYPRE_Solver() const { return ads; }

   virtual HYPRE_PtrToParSolverFcn SetupFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ADSSetup; }
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const
   { return (HYPRE_PtrToParSolverFcn) HYPRE_ADSSolve; }

   virtual ~HypreADS();
};

HypreParMatrix * ParAdd(double a, HypreParMatrix *A, double b, HypreParMatrix *B);
HypreParMatrix * RAP(HypreParMatrix * Rt, HypreParMatrix * A, HypreParMatrix * P);
void PrintAsHypreParCSRMatrix(MPI_Comm comm, SparseMatrix & A, const std::string & fname);

// Recall: hypre_ParCSRMatrixFixZeroRows(hypre_ParCSRMatrix * A);
}
#endif /* HYPREEXTESION_HPP_ */
