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

#include <fstream>
#include <mfem.hpp>
#include "InversePermeabilityFunction.hpp"

using namespace mfem;

void InversePermeabilityFunction::SetNumberCells(int Nx_, int Ny_, int Nz_)
{
    Nx = Nx_;
    Ny = Ny_;
    Nz = Nz_;
}

void InversePermeabilityFunction::SetMeshSizes(double hx_, double hy_,
                                               double hz_)
{
    hx = hx_;
    hy = hy_;
    hz = hz_;
}

void InversePermeabilityFunction::Set2DSlice(SliceOrientation o, int npos_ )
{
    orientation = o;
    npos = npos_;
}

void InversePermeabilityFunction::SetConstantInversePermeability(double ipx,
                                                                 double ipy,
                                                                 double ipz)
{
    int compSize = Nx*Ny*Nz;
    int size = 3*compSize;
    inversePermeability.resize(size);

    for (int i(0); i < compSize; ++i)
    {
        inversePermeability[i] = ipx;
        inversePermeability[i+compSize] = ipy;
        inversePermeability[i+2*compSize] = ipz;
    }

}

void InversePermeabilityFunction::ReadPermeabilityFile(const std::string fileName)
{
    std::ifstream permfile(fileName);

    if (!permfile)
    {
        std::cerr << "Error in opening file " << fileName << std::endl;
        mfem_error("File does not exist");
    }

    inversePermeability.resize(3*Nx*Ny*Nz);
    double *ip = inversePermeability.data();
    double tmp;
    for(int l = 0; l < 3; l++)
    {
        for (int k = 0; k < Nz; k++)
        {
            for (int j = 0; j < Ny; j++)
            {
                for (int i = 0; i < Nx; i++)
                {
                    permfile >> *ip;
                    *ip = 1./(*ip);
                    ip++;
                }
                for (int i = 0; i < 60-Nx; i++)
                    permfile >> tmp; // skip unneeded part
            }
            for (int j = 0; j < 220-Ny; j++)
                for (int i = 0; i < 60; i++)
                    permfile >> tmp;  // skip unneeded part
        }

        if (l < 2) // if not processing Kz, skip unneeded part
            for (int k = 0; k < 85-Nz; k++)
                for (int j = 0; j < 220; j++)
                    for (int i = 0; i < 60; i++)
                        permfile >> tmp;
    }

}

void InversePermeabilityFunction::ReadPermeabilityFile(const std::string fileName,
                                                       MPI_Comm comm)
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    StopWatch chrono;

    chrono.Start();
    if (myid == 0)
        ReadPermeabilityFile(fileName);
    else
        inversePermeability.resize(3*Nx*Ny*Nz);
    chrono.Stop();

    if (myid==0)
        std::cout<<"Permeability file read in " << chrono.RealTime() << ".s \n";

    chrono.Clear();

    chrono.Start();
    MPI_Bcast(inversePermeability.data(), 3*Nx*Ny*Nz, MPI_DOUBLE, 0, comm);
    chrono.Stop();

    if (myid==0)
        std::cout<<"Permeability field distributed in " << chrono.RealTime() << ".s \n";

}

void InversePermeabilityFunction::InversePermeability(const Vector & x,
                                                      Vector & val)
{
    val.SetSize(x.Size());

    unsigned int i=0,j=0,k=0;

    switch (orientation)
    {
    case NONE:
        i = Nx-1-(int)floor(x[0]/hx/(1.+3e-16));
        j = (int)floor(x[1]/hy/(1.+3e-16));
        k = Nz-1-(int)floor(x[2]/hz/(1.+3e-16));
        break;
    case XY:
        i = Nx-1-(int)floor(x[0]/hx/(1.+3e-16));
        j = (int)floor(x[1]/hy/(1.+3e-16));
        k = npos;
        break;
    case XZ:
        i = Nx-1-(int)floor(x[0]/hx/(1.+3e-16));
        j = npos;
        k = Nz-1-(int)floor(x[2]/hz/(1.+3e-16));
        break;
    case YZ:
        i = npos;
        j = (int)floor(x[1]/hy/(1.+3e-16));
        k = Nz-1-(int)floor(x[2]/hz/(1.+3e-16));
        break;
    default:
        mfem_error("InversePermeabilityFunction::InversePermeability");
    }

    val[0] = inversePermeability[Ny*Nx*k + Nx*j + i];
    val[1] = inversePermeability[Ny*Nx*k + Nx*j + i + Nx*Ny*Nz];

    if (orientation == NONE)
        val[2] = inversePermeability[Ny*Nx*k + Nx*j + i + 2*Nx*Ny*Nz];

}

double InversePermeabilityFunction::PermeabilityXY(Vector &x)
{
    unsigned int i=0,j=0,k=0;

    i = Nx-1-(int)floor(x[0]/hx/(1.+3e-16));
    j = (int)floor(x[1]/hy/(1.+3e-16));
    k = npos;

    return 1.0/inversePermeability[Ny*Nx*k + Nx*j + i];
}

void InversePermeabilityFunction::NegativeInversePermeability(const Vector & x,
                                                              Vector & val)
{
    InversePermeability(x,val);
    val *= -1.;
}


void InversePermeabilityFunction::Permeability(const Vector & x, Vector & val)
{
    InversePermeability(x,val);
    for (double * it = val.GetData(), *end = val.GetData()+val.Size(); it != end; ++it )
        (*it) = 1./ (*it);
}

void InversePermeabilityFunction::PermeabilityTensor(const Vector & x, DenseMatrix & val)
{
    Vector tmp(val.Size());
    Permeability(x,tmp);
    val = 0.0;
    for (int i=0; i<val.Size(); ++i)
        val.Elem(i,i) = tmp(i);
}

double InversePermeabilityFunction::Norm2InversePermeability(const Vector & x)
{
    Vector val(3);
    InversePermeability(x,val);
    return val.Norml2();
}

double InversePermeabilityFunction::Norm1InversePermeability(const Vector & x)
{
    Vector val(3);
    InversePermeability(x,val);
    return val.Norml1();
}

double InversePermeabilityFunction::NormInfInversePermeability(const Vector & x)
{
    Vector val(3);
    InversePermeability(x,val);
    return val.Normlinf();
}

double InversePermeabilityFunction::InvNorm2(const Vector & x)
{
    Vector val(3);
    InversePermeability(x,val);
    return 1./val.Norml2();
}

double InversePermeabilityFunction::InvNorm1(const Vector & x)
{
    Vector val(3);
    InversePermeability(x,val);
    return 1./val.Norml1();
}

double InversePermeabilityFunction::InvNormInf(const Vector & x)
{
    Vector val(3);
    InversePermeability(x,val);
    return 1./val.Normlinf();
}


void InversePermeabilityFunction::ClearMemory()
{
    std::vector<double>().swap(inversePermeability);
}

int InversePermeabilityFunction::Nx(60);
int InversePermeabilityFunction::Ny(220);
int InversePermeabilityFunction::Nz(85);
double InversePermeabilityFunction::hx(20);
double InversePermeabilityFunction::hy(10);
double InversePermeabilityFunction::hz(2);
std::vector<double> InversePermeabilityFunction::inversePermeability;
InversePermeabilityFunction::SliceOrientation InversePermeabilityFunction::orientation(
    InversePermeabilityFunction::NONE );
int InversePermeabilityFunction::npos(-1);
