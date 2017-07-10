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

#ifndef ELEMAGG_COEFFICIENT_HPP_
#define ELEMAGG_COEFFICIENT_HPP_

class PolynomialCoefficient : public Coefficient
{
public:
	PolynomialCoefficient(int orderx, int ordery, int orderz = 0):order_x(orderx), order_y(ordery), order_z(orderz){};
	virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
	{

		double x[3];
		Vector transip(x, 3);

		T.Transform(ip, transip);

		return std::pow(x[0], order_x)* std::pow(x[1], order_y)* std::pow(x[2], order_z);

	}
	virtual void Read(std::istream &in){ in >> order_x >> order_y >> order_z;}

private:
	int order_x;
	int order_y;
	int order_z;

};


class VectorPolynomialCoefficient : public VectorCoefficient
{

public:
	VectorPolynomialCoefficient(int VDim, int component_, int orderx, int ordery, int orderz=0):
		VectorCoefficient(VDim),
		component(component_),
		order_x(orderx), order_y(ordery), order_z(orderz)
		{
		if(component >= VDim)
			mfem_error("VectorPolynomialCoefficient");
		};


   using VectorCoefficient::Eval;
   virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
	   double x[3]; x[2] = 1.0; //Just to make sure that pow(x[2], ORDER_Z) is always ok
	   Vector transip(x, 3);

	   T.Transform (ip, transip);

	   V.SetSize (vdim);
	   V = 0.0;
	   V(component) = std::pow(x[0], order_x)* std::pow(x[1], order_y)*std::pow(x[2], order_z);
   }

   virtual ~VectorPolynomialCoefficient() { }

private:
   int component;
	int order_x;
	int order_y;
	int order_z;
};

class RTPullbackCoefficient : public VectorCoefficient
{
public:
	RTPullbackCoefficient(int VDim, int order_x_, int order_y_, int order_z_ = 0):
		VectorCoefficient(VDim),
		order_x(order_x_),
		order_y(order_y_),
		order_z(order_z_)
	{

	}

	using VectorCoefficient::Eval;

	virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
	{
		   double x[3]; x[2] = 1.0; //Just to make sure that pow(x[2], ORDER_Z) is always ok
		   Vector transip(x, 3);

		   T.Transform (ip, transip);

		   V.SetSize (vdim);

		   double scaling = std::pow(x[0], order_x)* std::pow(x[1], order_y)*std::pow(x[2], order_z);

		   for(int comp(0); comp < vdim; ++comp)
			   V(comp) = scaling * x[comp];
	}

private:
	int order_x;
	int order_y;
	int order_z;
};

void fillCoefficientArray(int nDim, int polyorder, Array<Coefficient *> & coeffs);
void fill2DCoefficientArray(int polyorder, Array<Coefficient *> & coeffs);
void fill3DCoefficientArray(int polyorder, Array<Coefficient *> & coeffs);

void fillVectorCoefficientArray(int nDim, int polyorder, Array<VectorCoefficient *> & coeffs);
void fill2DVectorCoefficientArray(int polyorder, Array<VectorCoefficient *> & coeffs);
void fill3DVectorCoefficientArray(int polyorder, Array<VectorCoefficient *> & coeffs);

void fillRTVectorCoefficientArray(int nDim, int polyorder, Array<VectorCoefficient *> & coeffs);
void fillRT2DVectorCoefficientArray(int polyorder, Array<VectorCoefficient *> & coeffs);
void fillRT3DVectorCoefficientArray(int polyorder, Array<VectorCoefficient *> & coeffs);

template<class T>
void freeCoeffArray(Array<T*> & coeffs)
{
	T** c = coeffs.GetData();
	T** end = c + coeffs.Size();

	for( ; c != end; ++c)
		delete *c;
}

#endif /* ELEMAGG_COEFFICIENT_HPP_ */
