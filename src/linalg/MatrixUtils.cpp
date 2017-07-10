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

#include "elag_linalg.hpp"
#include "general/sort_pairs.hpp"

bool AreAlmostEqual(const SparseMatrix & A, const SparseMatrix & B, const std::string & Aname, const std::string & Bname, double tol, bool verbose, std::ostream & os)
{

	if(A.Size() != B.Size() || A.Width() != B.Width() )
	{
		std::stringstream err_msg;
		err_msg << __FILE__<<":"<< __LINE__ << " - AreAlmostEqual(const SparseMatrix & A, const SparseMatrix & B, const std::string & Aname, const std::string & Bname, double tol): \n";
		err_msg << "Size( " << Aname << " ) = [" << A.Size() << ", [" << A.Width() << "]\n";
		err_msg << "Size( " << Bname << " ) = [" << B.Size() << ", [" << B.Width() << "]\n";
		err_msg << "Sizes don't match"<<std::endl;
		mfem_error(err_msg.str().c_str());
	}

	int nrows = A.Size();
	int ncols = A.Width();

	int * i_A = A.GetI();
	int * j_A = A.GetJ();
	double * v_A = A.GetData();

	int * i_B = B.GetI();
	int * j_B = B.GetJ();
	double * v_B = B.GetData();

	Vector rowVals(ncols);
	rowVals = 0.0;

	int ndiff(0);
	double maxErr(0.0), mydiff(0);

	for(int irow(0); irow < nrows; ++irow)
	{
		for(int jpos = i_A[irow]; jpos < i_A[irow+1]; ++ jpos)
			rowVals[ j_A[jpos] ] = v_A[jpos];

		for(int jpos = i_B[irow]; jpos < i_B[irow+1]; ++ jpos)
			rowVals[ j_B[jpos] ] -= v_B[jpos];

		for(int jpos = i_A[irow]; jpos < i_A[irow+1]; ++ jpos)
		{
			mydiff = fabs( rowVals[ j_A[jpos] ] );

			if( mydiff > tol)
				++ndiff;
			if( mydiff > maxErr)
				maxErr = mydiff;

			rowVals[ j_A[jpos] ] = 0.0;
		}

		for(int jpos = i_B[irow]; jpos < i_B[irow+1]; ++ jpos)
		{
			mydiff = fabs( rowVals[ j_B[jpos] ] );
			if(mydiff  > tol)
				++ndiff;

			if( mydiff > maxErr)
				maxErr = mydiff;

			rowVals[ j_B[jpos] ] = 0.0;
		}
	}

	if(mydiff > tol || verbose)
	{
		os << "    nnz( " << Aname << " - " << Bname << " ) = " << ndiff << "\n";
		os << "normInf( " << Aname << " - " << Bname << " ) = " << maxErr << "\n";
	}

	return (mydiff <= tol);
}


bool AreAlmostEqual(const SparseMatrix & A, const SparseMatrix & B, const SparseMatrix & G, const std::string & Aname, const std::string & Bname,const std::string & Gname, double tol, bool verbose, std::ostream & os)
{

	if(A.Size() != B.Size() || A.Width() != B.Width() || G.Size() != A.Size() || G.Width() != A.Width() )
	{
		std::stringstream err_msg;
		err_msg << __FILE__<<":"<< __LINE__ << " - AreAlmostEqual(const SparseMatrix & A, const SparseMatrix & B, const SparseMatrix & G, const std::string & Aname, const std::string & Bname,const std::string & Gname, double tol): \n";
		err_msg << "Size( " << Aname << " ) = [" << A.Size() << ", [" << A.Width() << "]\n";
		err_msg << "Size( " << Bname << " ) = [" << B.Size() << ", [" << B.Width() << "]\n";
		err_msg << "Size( " << Gname << " ) = [" << G.Size() << ", [" << G.Width() << "]\n";
		err_msg << "Sizes don't match"<<std::endl;
		mfem_error(err_msg.str().c_str());
	}

	int nrows = A.Size();
	int ncols = A.Width();

	int * i_A = A.GetI();
	int * j_A = A.GetJ();
	double * v_A = A.GetData();

	int * i_B = B.GetI();
	int * j_B = B.GetJ();
	double * v_B = B.GetData();

	int * i_G = G.GetI();
	int * j_G = G.GetJ();

	Vector rowVals(ncols);
	rowVals = 0.0;

	int ndiff(0);
	double maxErr(0.0), mydiff(0);

	for(int irow(0); irow < nrows; ++irow)
	{
		for(int jpos = i_A[irow]; jpos < i_A[irow+1]; ++ jpos)
			rowVals[ j_A[jpos] ] = v_A[jpos];

		for(int jpos = i_B[irow]; jpos < i_B[irow+1]; ++ jpos)
			rowVals[ j_B[jpos] ] -= v_B[jpos];

		for(int jpos = i_G[irow]; jpos < i_G[irow+1]; ++ jpos)
		{
			mydiff = fabs( rowVals[ j_G[jpos] ] );

			if( mydiff > tol)
				++ndiff;
			if( mydiff > maxErr)
				maxErr = mydiff;
		}

		//Restore rowVals = 0
		for(int jpos = i_A[irow]; jpos < i_A[irow+1]; ++ jpos)
			rowVals[ j_A[jpos] ] = 0.0;


		for(int jpos = i_B[irow]; jpos < i_B[irow+1]; ++ jpos)
			rowVals[ j_B[jpos] ] = 0.0;
	}

	if(mydiff > tol || verbose)
	{
		os << " Restrict the comparison on the sparsisty pattern of " << Gname << "\n";
		os << "    nnz( " << Aname << " - " << Bname << " ) = " << ndiff << "\n";
		os << "normInf( " << Aname << " - " << Bname << " ) = " << maxErr << "\n";
	}

	return (mydiff <= tol);
}

bool IsAlmostIdentity(const SparseMatrix & A, double tol, bool verbose )
{
	int nrows = A.Size();
	int ncols = A.Width();

	int * i_A = A.GetI();
	int * j_A = A.GetJ();
	double * v_A = A.GetData();

	if(nrows != ncols)
		mfem_error("IsAlmostIdentity A should be a square matrix");

	int ndiff(0);
	double maxErr(0.0), mydiff(0);

	for(int irow(0); irow < nrows; ++irow)
		for(int jpos = i_A[irow]; jpos < i_A[irow+1]; ++ jpos)
		{
			if(j_A[jpos] == irow)
				mydiff = fabs(v_A[jpos] - 1.);
			else
				mydiff = fabs(v_A[jpos]);

			if( mydiff > tol)
				++ndiff;
			if( mydiff > maxErr)
				maxErr = mydiff;
		}

	if(mydiff > tol || verbose)
	{
		std::cout << "    nnz( A - I ) = " << ndiff << "\n";
		std::cout << "normInf( A - I  ) = " << maxErr << "\n";
	}

	return (mydiff <= tol);
}

bool IsDiagonal(const SparseMatrix & A)
{
    if( A.Size() != A.Width() || A.NumNonZeroElems() != A.Size() )
        return false;

    int size = A.Size();

    int * it = A.GetJ();
    for(int i = 0; i < size; ++i, ++it)
        if(*it != i)
            return false;

    return true;
}

void fillSparseIdentity(int * I, int * J, double * A, int size)
{
	for(int i(0); i <= size; ++i)
		I[i] = i;

	for(int i(0); i < size; ++i)
	{
		J[i] = i;
		A[i] = 1.;
	}
}

SparseMatrix * createSparseIdentityMatrix(int size)
{
	int * I = new int[size+1];
	int * J = new int[size];
	double * A = new double[size];

	fillSparseIdentity(I, J,  A, size);

	return new SparseMatrix(I,J,A,size, size);
}



// Returns a matrix 1 by width, with entries A(0,i) = data[i] for i \in 0 ... width-1
SparseMatrix * createSparseMatrixRepresentationOfScalarProduct(double * data, int width)
{
	int * I = new int[2];
	int * J = new int[width];

	I[0] = 0; I[1] = width;
	for(int i(0); i < width; ++i)
		J[i] = i;

	return new SparseMatrix(I,J,data,1,width);
}

void          destroySparseMatrixRepresentationOfScalarProduct(SparseMatrix *& A)
{
	delete[] A->GetI();
	delete[] A->GetJ();
	A->LoseData();
	delete A;
	A = static_cast<SparseMatrix *>(NULL);
}

SparseMatrix * diagonalMatrix(double * data, int size)
{
	int * i_A = new int[size+1];
	int * j_A = new int[size];

	for(int i(0); i < size; ++i)
	{
		i_A[i] = i;
		j_A[i] = i;
	}
	i_A[size] = size;

	return new SparseMatrix(i_A, j_A, data, size, size);
}

SparseMatrix * diagonalMatrix(int size)
{
	double * data = new double[size];
	std::fill(data, data+size, 0.);
	return diagonalMatrix(data, size);
}

SparseMatrix * spzeros(int nrows, int ncols)
{
	int * i = new int[nrows+1];
	std::fill(i, i+nrows+1, 0);
	int * j = NULL;
	double * a = NULL;
	return new SparseMatrix(i,j,a, nrows, ncols);
}

SparseMatrix * DeepCopy(SparseMatrix & A)
{
	if( !A.Finalized() )
		mfem_error("DeepCopy: Matrix A should be finalized \n");

	int nrows = A.Size();
	int ncols = A.Width();
	int nnz   = A.NumNonZeroElems();

	int * i_new = new int[nrows+1];
	int * j_new = new int[nnz];
	double * a_new = new double[nnz];

	memcpy(i_new, A.GetI(), (nrows+1)*sizeof(int) );
	memcpy(j_new, A.GetJ(), nnz*sizeof(int));
	memcpy(a_new, A.GetData(), nnz*sizeof(double));

	return new SparseMatrix(i_new, j_new, a_new, nrows, ncols);
}

void dropSmallEntry(SparseMatrix & A, double tol)
{
	int * i_it = A.GetI();
	int * j_it = A.GetJ();
	double * a_val = A.GetData();

	int * new_j = j_it;
	double * new_a = a_val;

	int nrows = A.Size();
	int currentCounter = 0;
	int end;
	int nnz = 0;

	for(int irow = 0; irow < nrows; ++irow)
	{
		for(end = i_it[irow+1]; currentCounter < end; ++currentCounter, ++j_it, ++a_val)
		{
			if( fabs(*a_val) > tol )
			{
				nnz++;
				*(new_a++) = *a_val;
				*(new_j++) = *j_it;
			}
		}
		i_it[irow+1] = nnz;
	}
}

void signumTransformation(SparseMatrix & A)
{
	double tol = 1e-10;
	int nnz = A.NumNonZeroElems();
	double * a = A.GetData();

	for(double * it = a; it != a+nnz; ++it)
		if(fabs(*it) < tol)
			*it = 0.;
		else
			*it = (*it > 0) ? 1.:-1.;
}

void CheckMatrix(SparseMatrix & A)
{
	int nCols = A.Width();

	int nnz = A.NumNonZeroElems();
	int * j = A.GetJ();
	int * end = j+nnz;

	for( ;j != end;++j)
		if(*j < 0 || *j >= nCols)
			mfem_error("CheckMatrix(SparseMatrix & A) #1");

	Vector v(A.GetData(), nnz);
	if(v.CheckFinite())
		mfem_error("CheckMatrix(SparseMatrix & A) #2");

}

SparseMatrix * Add(double a, const SparseMatrix & A, double b, const SparseMatrix & B, double c, const SparseMatrix & C)
{
	if(A.Size() != B.Size() || B.Size() != C.Size() )
		mfem_error("R = a*A+b*B+c*C: A,B,C have different number of rows \n");

	if(A.Width() != B.Width() || B.Width() != C.Width() )
		mfem_error("R = a*A+b*B+c*C: A,B,C have different number of cols \n");

	int nrows = A.Size();
	int ncols = A.Width();

	int * R_i = new int[nrows+1];
	int * R_j;
	double * R_data;

	int * A_i = A.GetI();
	int * A_j = A.GetJ();
	double * A_data = A.GetData();

	int * B_i = B.GetI();
	int * B_j = B.GetJ();
	double * B_data = B.GetData();

	int * C_i = C.GetI();
	int * C_j = C.GetJ();
	double * C_data = C.GetData();

	int * marker = new int[ncols];
	std::fill(marker, marker+ncols, -1);

	int num_nonzeros = 0, jcol;
	R_i[0] = 0;
	for (int ir = 0; ir < nrows; ir++)
	{
		for (int ia = A_i[ir]; ia < A_i[ir+1]; ia++)
		{
			jcol = A_j[ia];
			marker[jcol] = ir;
			num_nonzeros++;
		}
		for (int ib = B_i[ir]; ib < B_i[ir+1]; ib++)
		{
			jcol = B_j[ib];
			if (marker[jcol] != ir)
			{
				marker[jcol] = ir;
				num_nonzeros++;
			}
		}

		for (int ic = C_i[ir]; ic < C_i[ir+1]; ic++)
		{
			jcol = C_j[ic];
			if (marker[jcol] != ir)
			{
				marker[jcol] = ir;
				num_nonzeros++;
			}
		}

		R_i[ir+1] = num_nonzeros;
	}

	R_j = new int[num_nonzeros];
	R_data = new double[num_nonzeros];

	for (int ia = 0; ia < ncols; ia++)
		marker[ia] = -1;

	int pos = 0;
	for (int ir = 0; ir < nrows; ir++)
	{
		for (int ia = A_i[ir]; ia < A_i[ir+1]; ia++)
		{
			jcol = A_j[ia];
			R_j[pos] = jcol;
			R_data[pos] = a*A_data[ia];
			marker[jcol] = pos;
			pos++;
		}

		for (int ib = B_i[ir]; ib < B_i[ir+1]; ib++)
		{
			jcol = B_j[ib];
			if (marker[jcol] < R_i[ir])
			{
				R_j[pos] = jcol;
				R_data[pos] = b*B_data[ib];
				marker[jcol] = pos;
				pos++;
			}
			else
			{
				R_data[marker[jcol]] += b*B_data[ib];
			}
		}

		for (int ic = C_i[ir]; ic < C_i[ir+1]; ic++)
		{
			jcol = C_j[ic];
			if (marker[jcol] < R_i[ir])
			{
				R_j[pos] = jcol;
				R_data[pos] = c*C_data[ic];
				marker[jcol] = pos;
				pos++;
			}
			else
			{
				R_data[marker[jcol]] += c*C_data[ic];
			}
		}
	}

	delete[] marker;
	return new SparseMatrix(R_i, R_j, R_data, nrows, ncols);
}

void Full(const SparseMatrix & Asparse, DenseMatrix & Adense)
{
	int nrow = Asparse.Size();
	int ncol = Asparse.Width();

	Adense.SetSize(nrow, ncol);
	Adense = 0.;

	const int * i_A = Asparse.GetI();
	const int * j_A = Asparse.GetJ();
	const double * a_A = Asparse.GetData();

	int jcol = 0;
	int end;

	for(int irow(0); irow < nrow; ++irow)
		for(end = i_A[irow+1]; jcol != end; ++jcol)
			Adense(irow, j_A[jcol]) = a_A[jcol];
}

void AddMatrix(const DenseMatrix &A, const SparseMatrix & B, DenseMatrix & C)
{

	if( A.Height() != B.Size() || B.Size() != C.Height() )
		mfem_error("AddMatrix #1");

	if( A.Width() != B.Width() || B.Width() != C.Width() )
		mfem_error("AddMatrix #2");

	int nrows = B.Size();

	const int * i_B = B.GetI();
	const int * j_B = B.GetJ();
	const double * v_B = B.GetData();

	if( C.Data() != A.Data())
		C = A;

	for(int irow(0); irow < nrows; ++irow)
	{
		for(int jpos = i_B[irow]; jpos < i_B[irow+1]; ++ jpos)
			C( irow, j_B[jpos] ) += v_B[jpos];
	}

}

SparseMatrix * Kron(SparseMatrix & A, SparseMatrix & B)
{
	int nrowsA = A.Size();
	int ncolsA = A.Width();
	int nnzA   = A.NumNonZeroElems();

	int nrowsB = B.Size();
	int ncolsB = B.Width();
	int nnzB   = B.NumNonZeroElems();

	int nrowsR = nrowsA * nrowsB;
	int ncolsR = ncolsA * ncolsB;
	int nnzR   = nnzA   * nnzB;

	const int * i_A = A.GetI();
	const int * j_A = A.GetJ();
	const double * v_A = A.GetData();

	const int * i_B = B.GetI();
	const int * j_B = B.GetJ();
	const double * v_B = B.GetData();

	int    * ii_R = new int[nrowsR+2];
	ii_R[0] = 0;
	int    * i_R  = ii_R+1;
	int    * j_R = new int[nnzR];
	double * v_R = new double[nnzR];

	int irowA, jposA, jcolA, irownnzA;
	int irowB, jposB, jcolB, irownnzB;
	int irowR, jposR, jcolR;

	double vA, vB;

	//DO NOT ACCELERATE THIS LOOP
	i_R[0] = 0;
	for(irowA = 0; irowA < nrowsA; ++irowA)
	{
		irownnzA = i_A[irowA+1] - i_A[irowA];
		for(irowB = 0; irowB < nrowsB; ++irowB)
		{
			irownnzB = i_B[irowB+1] - i_B[irowB];
			irowR = irowA * nrowsB + irowB;
			i_R[irowR+1] = i_R[irowR]+(irownnzA*irownnzB);
		}
	}

#ifdef ELAG_DEBUG
	if(i_R[nrowsR] != nnzR)
		mfem_error("This routine is bugged!!");
#endif


	for(irowA = 0; irowA < nrowsA; ++irowA)
	{
		for(jposA = i_A[irowA]; jposA < i_A[irowA+1]; ++jposA)
		{
			jcolA = j_A[jposA];
			vA    = v_A[jposA];
			for(irowB = 0; irowB < nrowsB; ++irowB)
			{
				irowR = irowA * nrowsB + irowB;
				for(jposB = i_B[irowB]; jposB < i_B[irowB+1]; ++jposB)
				{
					jcolB = j_B[jposB];
					vB    = v_B[jposB];
					jcolR = jcolA * ncolsB + jcolB;
					jposR = i_R[irowR]++;
					j_R[jposR] = jcolR;
					v_R[jposR] = vA*vB;
				}
			}
		}
	}

	return new SparseMatrix(ii_R, j_R, v_R, nrowsR, ncolsR);
}

void AddMatrix(double sa, const DenseMatrix &A, double sb, const SparseMatrix & B, DenseMatrix & C)
{

	if( A.Height() != B.Size() || B.Size() != C.Height() )
	{
		mfem_error("AddMatrix #1");
	}

	if( A.Width() != B.Width() || B.Width() != C.Width() )
		mfem_error("AddMatrix #2");

	int nrows = B.Size();

	const int * i_B = B.GetI();
	const int * j_B = B.GetJ();
	const double * v_B = B.GetData();


	C = A;
	C *= sa;

	for(int irow(0); irow < nrows; ++irow)
	{
		for(int jpos = i_B[irow]; jpos < i_B[irow+1]; ++ jpos)
			C( irow, j_B[jpos] ) += sb * v_B[jpos];
	}

}

void AddOpenFormat(SparseMatrix & A, SparseMatrix & B)
{
	elag_assert( A.Size() == B.Size() );
	elag_assert( A.Width() >= B.Width() );

	int nrows = A.Size();
	Array<int> rows(1);
	Array<int> cols;
	DenseMatrix loc;

	int * i_B = B.GetI();
	int * j_B = B.GetJ();
	double * val = B.GetData();

	for(int i(0); i < nrows; ++i)
	{
		int start = i_B[i];
		int end = i_B[i+1];
		int len = end - start;

		rows[0] = i;
		cols.MakeRef(j_B+start, len);
		loc.UseExternalData(val+start, 1, len);

		A.AddSubMatrix(rows, cols, loc);

		loc.ClearExternalData();
	}
}

void Mult(const SparseMatrix & A, const DenseMatrix & B, DenseMatrix & out)
{
	if(A.Width() != B.Height() || A.Size() != out.Height() || B.Width() != out.Width() )
		mfem_error("Dimensions don't match");

	int size = A.Size();
	const int * I = A.GetI();
	const int * J = A.GetJ();
	const double * val = A.GetData();

	int i,j,end;
	double * bi_data(B.Data() ), * outi_data(out.Data() );

	for( int icol(0); icol < B.Width(); ++icol)
	{
		for (i = j = 0; i < size; i++)
		{
			double d = 0.0;
			for (end = I[i+1]; j < end; j++)
			{
				d += val[j] * bi_data[J[j]];
			}
			outi_data[i] = d;
		}
		bi_data += B.Height();
		outi_data += out.Height();
	}
}

SparseMatrix * PtAP(SparseMatrix & A, SparseMatrix & P)
{
	SparseMatrix * AP  = Mult(A,P);
	SparseMatrix * Pt  = Transpose(P);
	SparseMatrix * out = Mult(*Pt, *AP);

	delete AP;
	delete Pt;

	return out;
}

double RowNormL1(const SparseMatrix & A, int irow)
{
	Vector vals;
	Array<int> cols;

	A.GetRow(irow, cols, vals);

	return vals.Norml1();
}
