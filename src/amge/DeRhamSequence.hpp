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

#ifndef DERHAMSEQUENCE_HPP_
#define DERHAMSEQUENCE_HPP_

class DeRhamSequenceFE;

//! @class DeRhamSequence
/**
	@brief the key class for the whole project, keeps track of all the algebraic relationships at a given mesh level of the hierarchy

	this is normally created in two ways: from an MFEM ParMesh,
	or from calling Coarsen() from another DeRhamSequence
*/
class DeRhamSequence
{
public:
	DeRhamSequence(AgglomeratedTopology * topo, int nspaces);

	void SetjformStart(int start) { elag_assert(start >= topo->Dimensions() - topo->Codimensions() ); jformStart = start; }
	DeRhamSequence * Coarsen();
	void SetTargets(const Array<MultiVector *> & targets);
	void SetSVDTol( double tol) {tolSVD = tol;}

	void CheckInvariants();
	void CheckD();
	void CheckTrueD();
	void CheckDP();
	void CheckTrueDP();
	void CheckCoarseMassMatrix();
	void CheckTrueCoarseMassMatrix();
	void CheckCoarseDerivativeMatrix();
	void CheckTrueCoarseDerivativeMatrix();
	void CheckPi();

	void ShowProjector(int jform);
	void ShowDerProjector(int jform);
	virtual void showP(int jform, SparseMatrix & P_t, Array<int> & parts) = 0;
	virtual void show(int jform, MultiVector & v) = 0;
	void ComputeSpaceInterpolationError(int jform, const MultiVector & fineVector);
	virtual void ProjectCoefficient(int jform, Coefficient & c, Vector & v) = 0;
	virtual void ProjectVectorCoefficient(int jform, VectorCoefficient & c, Vector & v) = 0;

	HypreParMatrix * ComputeTrueDerivativeOperator(int jform);
	HypreParMatrix * ComputeTrueDerivativeOperator(int jform, Array<int> & ess_label);
	HypreParMatrix * ComputeTrueMassOperator(int jform);
	HypreParMatrix * ComputeTrueMassOperator(int jform, Vector & elemMatrixScaling);
	HypreParMatrix * ComputeTrueP(int jform);
	HypreParMatrix * ComputeTrueP(int jform, Array<int> & ess_label);
	HypreParMatrix * ComputeTruePi(int jform);
	HypreParMatrix * ComputeTrueProjectorFromH1ConformingSpace(int jform);
	void ComputeTrueProjectorFromH1ConformingSpace(int jform, HypreParMatrix *& Pix, HypreParMatrix *& Piy, HypreParMatrix *& Piz);

	SparseMatrix * GetDerivativeOperator(int jform) { return D[jform]; }
	SparseMatrix * ComputeDerivativeOperator(int jform, Array<int> & ess_label);
	SparseMatrix * ComputeMassOperator(int jform) { return Assemble(AgglomeratedTopology::ELEMENT, *M(AgglomeratedTopology::ELEMENT, jform), *dof[jform], *dof[jform]);}
	SparseMatrix * ComputeLumpedMassOperator(int jform);
	SparseMatrix * ComputeMassOperator(int jform, Vector & elemMatrixScaling);
	SparseMatrix * ComputeLumpedMassOperator(int jform, Vector & elemMatrixScaling);
	virtual SparseMatrix * ComputeProjectorFromH1ConformingSpace(int jform) = 0;
	SparseMatrix * GetP(int jform) { return P[jform]; }
	SparseMatrix * GetP(int jform, Array<int> & ess_label);
	SparseMatrix * GetAEntityDof(AgglomeratedTopology::EntityByCodim etype, int jform) { return dofAgg[jform]->GetAEntityDof(etype); }
	CochainProjector * GetPi(int jform) { return Pi[jform]; }
	int GetNumberOfForms() {return nforms;}
	int GetNumberOfDofs(int jform){ return dof[jform]->GetNDofs(); }
	int GetNumberOfTrueDofs(int jform){ return dof[jform]->GetDofTrueDof().GetTrueLocalSize(); }
	DofHandler * GetDofHandler(int jform){ return dof[jform]; }
	DofAgglomeration * GetDofAgg(int jform){ return dofAgg[jform]; }
	virtual DeRhamSequenceFE * FemSequence() = 0;

	void DumpD();
	void DumpP();

	virtual ~DeRhamSequence();

	friend class DeRhamSequenceAlg;

	static std::stringstream DeRhamSequence_os;

protected:

	MPI_Comm comm;

	int extimateUpperBoundNCoarseDof(int jform);
	void computeCoarseTraces(int jform);
	void computeCoarseTracesNoTargets(int jform);
	void computeCoarseTracesWithTargets(int jform);
	void computeCoarseTracesWithTargets_old(int jform);
	void compute0formCoarseTraces();
	void hFacetExtension(int jform);
	void hFacetExtension_new(int jform);
	void hRidgeExtension(int jform);
	void hPeakExtension(int jform);
	virtual void computePVTraces(AgglomeratedTopology::EntityByCodim icodim, Vector & PVinAgg) = 0;
	SparseMatrix * getUnextendedInterpolator(int jform);

	int jformStart;

	AgglomeratedTopology * topo;
	//! Number of Spaces Involved (4 in 3D, 3 in 2D)
	int nforms;
	//! DOF Handlers
	Array<DofHandler *> dof;
	//! DofAgglomeration handlers: dofAgg[i]==NULL  until Coarsen is called.
	Array<DofAgglomeration *> dofAgg;
	//! Derivative operators
	Array<SparseMatrix *> D;
	//! Local Mass Matrices (LOWER TRIANGULAR DATA STRUCTURE)
	Array2D<SparseMatrix *> M;

	Array< MultiVector *> targets;
	Array< Vector *> pvTraces;

	/*
	 * Lower triangular matrix P_t are the matrices that represent the new De-Rham sequence in term of the one on the previous level:
	 *
	 * In 3D we have:
	 *
	 *  0 --> H1 --> Hcurl --> Hdiv --> L2 --> 0
	 *
	 *              0-forms               1-forms                 2-forms                3-forms
	 * 0-codim   [  H1                    Hcurl                   Hdiv                   L2      ]
	 * 1-codim   [  H1_faces_traces       Hcurl_faces_traces      Hdiv_faces_traces              ]
	 * 2-codim   [  H1_edges_traces       Hcurl_edges_traces                                     ]
	 * 3-codim   [  H1_vertices_traces                                                           ]
	 *
	 * In 2D we have:
	 *
	 *  0 --> H1 --> Hcurl --> L2 --> 0
	 *
	 *              0-forms               1-forms                 2-forms
	 * 0-codim   [  H1                    Hcurl                   L2      ]
	 * 1-codim   [  H1_edges_traces       Hcurl_edges_traces              ]
	 * 2-codim   [  H1_vertices_traces                                    ]
	 *
	 */

	// P is the interpolation matrix from the coarser level to this level
	Array<SparseMatrix *> P;
	// Pi is the cochain projector from this level to the coarser one
	Array<CochainProjector *> Pi;

	DeRhamSequence * coarserSequence;
	DeRhamSequence * finerSequence;

    double tolSVD;         /*!< Relative tolerance to be used in the SVD */
    double smallEntry;     /*!< If an entry in the interpolation matrix is smaller then smallEntry it will be dropped. */


};

//! @class DeRhamSequenceAlg
/**
	@brief Subclass of DeRhamSequence to use if there is no mesh, if this is coarsened
*/
class DeRhamSequenceAlg : public DeRhamSequence
{
public:
	DeRhamSequenceAlg(AgglomeratedTopology * topo, int nspaces);
	virtual ~DeRhamSequenceAlg();
	virtual SparseMatrix * ComputeProjectorFromH1ConformingSpace(int jform);
	virtual void showP(int jform, SparseMatrix & P, Array<int> & parts);
	virtual void show(int jform, MultiVector & v);
	virtual DeRhamSequenceFE * FemSequence(){ return finerSequence->FemSequence(); }
	virtual void ProjectCoefficient(int jform, Coefficient & c, Vector & v);
	virtual void ProjectVectorCoefficient(int jform, VectorCoefficient & c, Vector & v);
protected:
	void computePVTraces(AgglomeratedTopology::EntityByCodim icodim, Vector & PVinAgg);

};

#endif /* DERHAMSEQUENCE2_HPP_ */
