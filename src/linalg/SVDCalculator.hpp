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

#ifndef SVDCALCULATOR_HPP_
#define SVDCALCULATOR_HPP_

class SVD_Calculator {
public:

	enum{ COMPUTE_U = 0x01, COMPUTE_VT = 0x02, SKINNY = 0x04 };

	SVD_Calculator();
	void setFlag(int flag);
	void setFlagOA();
	void setFlagON();
	void AllocateOptimalSize(int maxNRows_, int maxNCols_);

	// A = U S V^T
	void Compute(MultiVector & A, Vector & SingularValues, MultiVector & U, MultiVector * VT, int flag_);
	// A = U S V^T; U overrides A.
	void ComputeOA(MultiVector & A, Vector & singularValues, MultiVector & VT);
	// A = U S V^T; U overrides A. Now U is w orthogonal, i.e. U^T diag(w) U = I
	void ComputeOA(Vector & sqrt_w, MultiVector & A, Vector & singularValues, MultiVector & VT);
	// A = U S V^T; U overrides A.
	void ComputeON(MultiVector & A, Vector & singularValues);
	// A = U S V^T; U overrides A. Now U is w orthogonal, i.e. U^T diag(w) U = I
	void ComputeON(Vector & sqrt_w, MultiVector & A, Vector & singularValues);
	// A = U S V^T; U overrides A. Now U is W-orthogonal, i.e. U^T W U = I
	void ComputeON(DenseMatrix & W, MultiVector & A, Vector & singularValues);

	void ComputeON(DenseMatrix & A, Vector & singularValues);

	virtual ~SVD_Calculator();

private:

	int flag;

	char   jobu;
    char   jobvt;
    double *work;
    int    lwork;
    int    info;
    double qwork;

    int maxNRows;
    int maxNCols;

};

#endif /* SVDCALCULATOR_HPP_ */
