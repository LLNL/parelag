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

#ifndef INVERSEPERMEABILITYFUNCTION_HPP_
#define INVERSEPERMEABILITYFUNCTION_HPP_

#include <mpi.h>

class InversePermeabilityFunction
{
public:

    enum SliceOrientation {NONE, XY, XZ, YZ};

    static void SetNumberCells(int Nx_, int Ny_, int Nz_);
    static void SetMeshSizes(double hx, double hy, double hz);
    static void Set2DSlice(SliceOrientation o, int npos );

    static void ReadPermeabilityFile(const std::string fileName);
    static void ReadPermeabilityFile(const std::string fileName, MPI_Comm comm);
    static void SetConstantInversePermeability(double ipx, double ipy, double ipz);

    template<class F>
    static void Transform(const F & f)
    {
        for (int i = 0; i < 3*Nx*Ny*Nz; ++i)
            inversePermeability[i] = f(inversePermeability[i]);
    }

    static void InversePermeability(const mfem::Vector & x, mfem::Vector & val);
    static void PermeabilityTensor(const mfem::Vector & x, mfem::DenseMatrix & val);
    static double PermeabilityXY(mfem::Vector &x);
    static void NegativeInversePermeability(const mfem::Vector & x, mfem::Vector & val);
    static void Permeability(const mfem::Vector & x, mfem::Vector & val);

    static double Norm2InversePermeability(const mfem::Vector & x);
    static double Norm1InversePermeability(const mfem::Vector & x);
    static double NormInfInversePermeability(const mfem::Vector & x);

    static double InvNorm2(const mfem::Vector & x);
    static double InvNorm1(const mfem::Vector & x);
    static double InvNormInf(const mfem::Vector & x);

    static void ClearMemory();

private:
    static int Nx;
    static int Ny;
    static int Nz;
    static double hx;
    static double hy;
    static double hz;
    static std::vector<double> inversePermeability;

    static SliceOrientation orientation;
    static int npos;
};

#endif /* INVERSEPERMEABILITYFUNCTION_HPP_ */
