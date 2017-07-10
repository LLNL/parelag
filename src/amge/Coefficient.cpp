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

#include "elag_amge.hpp"

void fillCoefficientArray(int nDim, int polyorder, Array<Coefficient *> & coeffs)
{
	if(polyorder < 0)
	{
		coeffs.SetSize(0);
		return;
	}

	switch(nDim)
	{
	case 2:
		fill2DCoefficientArray(polyorder, coeffs);
		break;
	case 3:
		fill3DCoefficientArray(polyorder, coeffs);
		break;
	default:
		mfem_error("fillCoefficientArray");
	}
}


void fillVectorCoefficientArray(int nDim, int polyorder, Array<VectorCoefficient *> & coeffs)
{
	if(polyorder < 0)
	{
		coeffs.SetSize(0);
		return;
	}

	switch(nDim)
	{
	case 2:
		fill2DVectorCoefficientArray(polyorder, coeffs);
		break;
	case 3:
		fill3DVectorCoefficientArray(polyorder, coeffs);
		break;
	default:
		mfem_error("fillVectorCoefficientArray");
	}
}

void fillRTVectorCoefficientArray(int nDim, int polyorder, Array<VectorCoefficient *> & coeffs)
{
	if(polyorder < 0)
		mfem_error("fillRTVectorCoefficientArray #1");


	switch(nDim)
	{
	case 2:
		fillRT2DVectorCoefficientArray(polyorder, coeffs);
		break;
	case 3:
		fillRT3DVectorCoefficientArray(polyorder, coeffs);
		break;
	default:
		mfem_error("fillRTVectorCoefficientArray");
	}
}

void fill2DCoefficientArray(int polyorder, Array<Coefficient *> & coeffs)
{
	if(polyorder < 0)
		mfem_error("fill2DCoefficientArray #1");

	int size = (polyorder+1)*(polyorder+2);
	size /= 2;
	coeffs.SetSize(size);
	Coefficient ** c = coeffs.GetData();

	int order_x, order_y;

	for(int order_max(0); order_max <= polyorder; ++order_max)
		for(order_x = 0; order_x <= order_max; ++order_x)
		{
			order_y = order_max - order_x;
			*(c++) = new PolynomialCoefficient(order_x, order_y);
		}
}

void fill3DCoefficientArray(int polyorder, Array<Coefficient *> & coeffs)
{
	if(polyorder < 0)
		mfem_error("fill3DCoefficientArray #1");

	int size = 0;

	int order_x, order_y, order_z;
	for(int order_max(0); order_max <= polyorder; ++order_max)
		for(order_x = 0; order_x <= order_max; ++order_x)
			for(order_y = 0; order_y <= order_max-order_x; ++order_y)
				++size;

	coeffs.SetSize(size);
	Coefficient ** c = coeffs.GetData();

	for(int order_max(0); order_max <= polyorder; ++order_max)
		for(order_x = 0; order_x <= order_max; ++order_x)
			for(order_y = 0; order_y <= order_max-order_x; ++order_y)
		{
			order_z = order_max - order_x - order_y;
			*(c++) = new PolynomialCoefficient(order_x, order_y, order_z);
		}
}

void fill2DVectorCoefficientArray(int polyorder, Array<VectorCoefficient *> & coeffs)
{

	if(polyorder < 0)
		mfem_error("fill2DVectorCoefficientArray #1");

	int ncomp = 2;

	int size = ncomp*(polyorder+1)*(polyorder+2);
	size /= 2;
	coeffs.SetSize(size);
	VectorCoefficient ** c = coeffs.GetData();

	int order_x, order_y;

	for(int order_max(0); order_max <= polyorder; ++order_max)
		for(order_x = 0; order_x <= order_max; ++order_x)
			for(int icomp(0); icomp < ncomp; ++icomp)
		{
				order_y = order_max - order_x;
				*(c++) = new VectorPolynomialCoefficient(2, icomp, order_x, order_y);
		}
}

void fill3DVectorCoefficientArray(int polyorder, Array<VectorCoefficient *> & coeffs)
{

	if(polyorder < 0)
		mfem_error("fill3DVectorCoefficientArray #1");

	int ncomp = 3;

	int size = 0;

	int order_x, order_y, order_z;
	for(int order_max(0); order_max <= polyorder; ++order_max)
		for(order_x = 0; order_x <= order_max; ++order_x)
			for(order_y = 0; order_y <= order_max-order_x; ++order_y)
				++size;

	size *= ncomp;

	coeffs.SetSize(size);
	VectorCoefficient ** c = coeffs.GetData();

	for(int order_max(0); order_max <= polyorder; ++order_max)
		for(order_x = 0; order_x <= order_max; ++order_x)
			for(order_y = 0; order_y <= order_max-order_x; ++order_y)
				for(int icomp(0); icomp < ncomp; ++icomp)
				{
					order_z = order_max - order_x - order_y;
					*(c++) = new VectorPolynomialCoefficient(3, icomp, order_x, order_y, order_z);
				}
}

void fillRT2DVectorCoefficientArray(int polyorder, Array<VectorCoefficient *> & coeffs)
{

	if(polyorder < 0)
		mfem_error("fillRT2DVectorCoefficientArray #1");

	int ncomp = 2;

	int size = ncomp*(polyorder+1)*(polyorder+2);
	size /= 2;
	size += polyorder+1;
	coeffs.SetSize(size);
	VectorCoefficient ** c = coeffs.GetData();

	int order_x, order_y;

	for(int order_max(0); order_max <= polyorder; ++order_max)
		for(order_x = 0; order_x <= order_max; ++order_x)
			for(int icomp(0); icomp < ncomp; ++icomp)
		{
				order_y = order_max - order_x;
				*(c++) = new VectorPolynomialCoefficient(2, icomp, order_x, order_y);
		}

	for(order_x = 0; order_x <= polyorder; ++order_x)
	{
		order_y = polyorder - order_x;
		*(c++) = new RTPullbackCoefficient(2, order_x, order_y);
	}
}

void fillRT3DVectorCoefficientArray(int polyorder, Array<VectorCoefficient *> & coeffs)
{

	if(polyorder < 0)
		mfem_error("fillRT3DVectorCoefficientArray #1");


	int ncomp = 3;

	int size = 0;

	int order_x, order_y, order_z;
	for(int order_max(0); order_max <= polyorder; ++order_max)
		for(order_x = 0; order_x <= order_max; ++order_x)
			for(order_y = 0; order_y <= order_max-order_x; ++order_y)
				++size;
	size *= ncomp;

	for(order_x = 0; order_x <= polyorder; ++order_x)
		for(order_y = 0; order_y <= polyorder-order_x; ++order_y)
			++size;


	coeffs.SetSize(size);
	VectorCoefficient ** c = coeffs.GetData();

	for(int order_max(0); order_max <= polyorder; ++order_max)
		for(order_x = 0; order_x <= order_max; ++order_x)
			for(order_y = 0; order_y <= order_max-order_x; ++order_y)
				for(int icomp(0); icomp < ncomp; ++icomp)
				{
					order_z = order_max - order_x - order_y;
					*(c++) = new VectorPolynomialCoefficient(3, icomp, order_x, order_y, order_z);
				}

		for(order_x = 0; order_x <= polyorder; ++order_x)
			for(order_y = 0; order_y <= polyorder-order_x; ++order_y)
				{
					order_z = polyorder - order_x - order_y;
					*(c++) = new RTPullbackCoefficient(3, order_x, order_y, order_z);
				}
}
