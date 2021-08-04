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
#include <sstream>
#include <ostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

#include <mpi.h>

#include "elag.hpp"
#include "utilities/MPIDataTypes.hpp"

using namespace mfem;
using namespace parelag;
using namespace std;

Vector *pOMx, *pOMy;
int is;
int *pxind, *pyind;
const double *plam;

double KLmode(const Vector &xc)
{
    return ((sin((*pOMx)(pxind[is])*xc(0)) + plam[0]*(*pOMx)(pxind[is])*cos((*pOMx)(pxind[is])*xc(0)))
            * (sin((*pOMy)(pyind[is])*xc(1)) + plam[1]*(*pOMy)(pyind[is])*cos((*pOMy)(pyind[is])*xc(1))));

}

void GetOmegas (Vector &OM, double a, int NOM)
{
    const double PI=3.14159265;
    const int maxit = 1000;
    const double tol = 0.00001;

    double asyx = 1.0/a;

    OM=0.0;

    // A vector of endpoints for intervals
    Vector xlvec(NOM+2);
    int ctr=0;

    xlvec = PI/2.0;
    if (asyx < PI/2.0)
    {
        xlvec[0] = asyx;
        xlvec[1] = PI/2.0;
        ctr++;
    }
    while (ctr < (NOM+1))
    {
        ctr++;
        xlvec[ctr] = xlvec[ctr-1] + PI;
        if ((asyx<xlvec[ctr])&&(asyx>xlvec[ctr-1]))
        {
            xlvec[ctr] = asyx;
            ctr++;
            xlvec[ctr] = xlvec[ctr-2] + PI;
        }
    }

    double xl, xr, xm, fl, fr, fm; 
    for (int ii=0; ii < NOM; ++ii)
    {
        xl = 1.001*xlvec[ii];
        xr = 0.999*xlvec[ii+1];
        xm = (xl+xr)/2.0;
        fl = tan(xl)-(2.0*a*xl)/(a*a*xl*xl-1.0);
        fr = tan(xr)-(2.0*a*xr)/(a*a*xr*xr-1.0);
        fm = tan(xm)-(2.0*a*xm)/(a*a*xm*xm-1.0);

        int it=0;
        while ((fabs(fm)>tol)&&(it<maxit))
        {
            xm = (xl+xr)/2.0;
            fm = tan(xm)-(2.0*a*xm)/(a*a*xm*xm-1.0);
            if ((fl*fm)<0)
                xr=xm;
            else
                xl=xm;
            fl = tan(xl)-(2.0*a*xl)/(a*a*xl*xl-1.0);
            fr = tan(xr)-(2.0*a*xr)/(a*a*xr*xr-1.0);
            it++;
        }
        OM(ii) = xm;
    }
}

void BuildAnalyticKLE(GridFunction *VECS, Vector &VALS, const double lam[2],
                      int NKL)
{
    // Number of 1D nodes necessary to form NKL 2D modes.
    int NOM = (int)((-1+sqrt(1+8*NKL))/2.0+1);
    Vector OMx(NOM), OMy(NOM); 
    pOMx = &OMx; pOMy = &OMy;
    plam = lam;

    // Get 1D evals.
    GetOmegas(OMx, lam[0], NOM);
    GetOmegas(OMy, lam[1], NOM);

    Vector Thx(NOM);
    Vector Thy(NOM);
    for (int ii=0; ii < NOM; ++ii)
    {
        Thx[ii] = 2*lam[0]/(lam[0]*lam[0]*OMx[ii]*OMx[ii]+1.0);
        Thy[ii] = 2*lam[1]/(lam[1]*lam[1]*OMy[ii]*OMy[ii]+1.0);
    }

    int NOM2D=(int)(NOM*(NOM+1)/2);
    int xind[NOM2D], yind[NOM2D];
    Vector tVALS(NOM2D);
    pxind = xind; pyind = yind;

    int ectr=0;
    for (int ii=0; ii < NOM; ++ii)
        for (int jj=0; jj <= ii; ++jj)
        {
            tVALS(ectr)= Thx(ii-jj)*Thy(jj);
            xind[ectr] = (int)(ii-jj);
            yind[ectr] = (int)(jj);
            ectr++;
        }

    // Sort 2D evals in ascending order.
    double dtmp;
    int xtmp, ytmp;
    for (int ii=0; ii < NOM2D; ++ii)
        for (int jj=NOM2D-1; jj > ii; --jj)
            if (tVALS(jj-1) > tVALS(jj))
            {
                // Swap values.
                dtmp = tVALS(jj);
                tVALS(jj) = tVALS(jj-1);
                tVALS(jj-1) = dtmp;
                // Swap ii indices
                xtmp = xind[jj];
                xind[jj] = xind[jj-1];
                xind[jj-1] = xtmp; 
                // Swap ii indices.
                ytmp = yind[jj];
                yind[jj] = yind[jj-1];
                yind[jj-1] = ytmp;
            }


    //  Build 2D eigenvectors.
    FunctionCoefficient fc(KLmode);
    for (int ii=0; ii < NKL; ++ii)
    {
        is=ii+(NOM2D-NKL);
        double L2N=0.0; 
        VECS[ii].ProjectCoefficient(fc);
        ConstantCoefficient zero(0.);
        Coefficient *zero_ptr = &zero;
        VECS[ii] /= VECS[ii].ComputeL2Error(&zero_ptr);
    }

    // Copy NKL evals into VALS.
    for (int ii=0; ii < NKL; ++ii)
    {
        int is=ii+(NOM2D-NKL);
        VALS[ii] = tVALS[is];
    }
}

GridFunction GetKLReal(FiniteElementSpace *f, int NKL,
                       GridFunction *VECS, Vector &VALS,
                       GridFunction &mean, double var,
                       Vector &Th)
{
    GridFunction K(f);
    K=mean;
    for (int ii=0; ii < NKL; ++ii)
        add(K, sqrt(var*VALS[ii])*Th[ii], VECS[ii], K);

    for (int ii=0; ii < K.Size(); ++ii)
        K[ii] = exp(K[ii]);

    return K;
}

/*
The format of the output is as follows.

All entities on a line a separated by white space. All matrices are
presented linearly by printing all their entries on a line column-by-column. The
coefficient is 

First line contains the information about the AE and the derivative matrix D
from H(div) to L2 on the fine scale:

<AE id> <# elements in AE> <height of D> <width of D> <space separated entries of D>

The rest of the lines show coefficient samples and local (on AE) prolongation, P,
matrices for H(div) and L2:

<space separated values of k on the elements of AE> <space separated values of k^{-1}>
    <H(div) form id> <H(div) P height> <H(div) P height> <H(div) P width> <H(div) P entries>
    <L2 form id> <L2 P height> <L2 P height> <L2 P width> <L2 P entries>
*/

int main(int argc, char *argv[])
{
    // Initialize MPI.
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    cout << "-- This generates H(div)-L2 samples with random permeabilities\n"
            "-- and outputs information about the permeability and hierarchy "
            "in a file.\n\n";

    // The file, from which to read the mesh.
    const string meshfile = "UnitSquare.mesh";

    // The file, where to write the output.
    const string outputfile = "output.txt";

    // The agglomerate element to observe.
    const int iAE = 53;

    // Number of samples.
    const int num_samples = 10000;

    // The number of times to refine in serial.
    int ser_ref_levels = 0;

    // The order of the finite elements on the finest level.
    const int feorder = 0;

    // The order of the polynomials to include in the coarse spaces
    // (after interpolating them onto the fine space).
    const int upscalingOrder = 0;

    // Number of levels to generate (including the finest one).
    const int nLevels = 2;

    // SVD tolerance.
    const double tolSVD = 1e-9;

    // Stochastic parameters:

    // The expected value for the Gauss field.
    const double cGauss_mean = 0.0;
    // The variance (i.e., squared) of the Gauss field.
    const double Gauss_var = 1.0;
    // Correlation lengths of the Gauss field.
    const double Gauss_cor[] = {0.3, 0.3};
    // Number of KL modes.
    const int KL_modes = 16;
    // Initial normal sample mean.
    const double normal_mean = 0.0;
    // Initial normal sample variance (i.e., squared).
    const double normal_var = 1.0;
    // Random walk mean.
    const double RW_mean = 0.0;
    // Random walk variance (i.e., squared).
    const double RW_var = 1.0;

    // Read the (serial) mesh from the given mesh file and uniformly refine it.
    shared_ptr<ParMesh> pmesh;
    {
        cout << "\nReading and refining serial mesh...\n";

        ifstream imesh(meshfile);
        if (!imesh)
        {
            cerr << "ERROR: Cannot open mesh file: " << meshfile << ".\n";
            return EXIT_FAILURE;
        }

        auto mesh = make_unique<Mesh>(imesh, true, true);
        imesh.close();

        for (int l = 0; l < ser_ref_levels; ++l)
            mesh->UniformRefinement();

        pmesh = make_shared<ParMesh>(comm, *mesh);
        pmesh->ReorientTetMesh();
    }

    // Refine the mesh in parallel.
    const int nDimensions = pmesh->Dimension();

    vector<int> level_nElements(nLevels);
    level_nElements[0] = pmesh->GetNE();
    for (int l = 1; l < nLevels; ++l)
        level_nElements[l] = (int)(level_nElements[l - 1] / 6);

    // Obtain the hierarchy of agglomerate topologies.
    cout << "Agglomerating topology for " << nLevels - 1
         << " coarse levels...\n";

    constexpr auto AT_elem = AgglomeratedTopology::ELEMENT;
    MetisGraphPartitioner partitioner;
    partitioner.setFlags(MetisGraphPartitioner::KWAY);
    partitioner.setOption(METIS_OPTION_SEED, 0);
    partitioner.setOption(METIS_OPTION_CONTIG, 1);
    vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

    topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);
    for(int l = 0; l < nLevels - 1; ++l)
    {
        Array<int> partitioning(topology[l]->GetNumberLocalEntities(AT_elem));
        partitioner.doPartition(*(topology[l]->LocalElementElementTable()),
                                level_nElements[l + 1], partitioning);
        topology[l + 1] = topology[l]->CoarsenLocalPartitioning(partitioning,
                                                                false, false);
    }

    ShowTopologyAgglomeratedElements(topology[1].get(), pmesh.get());

    // Construct the hierarchy of spaces, thus forming a hierarchy of (partial)
    // de Rham sequences.
    cout << "Building the fine-level de Rham sequence...\n";

    vector<shared_ptr<DeRhamSequence>> sequence(topology.size());

    const int jform = nDimensions - 1; // This is the H(div) form.
    if(nDimensions == 3)
        sequence[0] = make_shared<DeRhamSequence3D_FE>(topology[0], pmesh.get(),
                                                       feorder);
    else
        sequence[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(topology[0],
                                                            pmesh.get(),
                                                            feorder);

    // To build H(div) (form 1 in 2D), it is needed to obtain all forms and
    // spaces with larger indices.
    sequence[0]->SetjformStart(jform);

    cout << "Interpolating and setting polynomial targets...\n";

    DeRhamSequenceFE *DRSequence_FE = sequence[0]->FemSequence();
    MFEM_ASSERT(DRSequence_FE,
                "Failed to obtain the fine-level de Rham sequence.");
    DRSequence_FE->SetUpscalingTargets(nDimensions, upscalingOrder);

    ofstream ofile(outputfile);
    if (!ofile)
    {
        cerr << "ERROR: Cannot open output file: " << outputfile << ".\n";
        return EXIT_FAILURE;
    }
    ofile.precision(16);
    cout << "Sampling: ";

    FiniteElementSpace *fes = DRSequence_FE->GetFeSpace(nDimensions);
    GridFunction k_gf(fes);
    GridFunction Gauss_mean(fes);
    Gauss_mean = cGauss_mean;
    // Initialize storage for KL eigenpairs.
    Vector evals(KL_modes);
    GridFunction efuncs[KL_modes];
    for (int k=0; k < KL_modes; ++k)
        efuncs[k].SetSpace(fes);

    // Get KLE basis.
    BuildAnalyticKLE(efuncs, evals, Gauss_cor, KL_modes);

#if 0
    {
        for (int k=0; k < KL_modes; ++k)
        {
            MultiVector tmp(efuncs[k].GetData(), 1,
                            efuncs[k].Size());
            sequence[0]->show(nDimensions, tmp);
        }
    }
#endif

    // Get initial normal coefficients.
    default_random_engine generator(
            chrono::system_clock::now().time_since_epoch().count());
    Vector normal_coefs(KL_modes);
    {
        normal_distribution<double> distribution(normal_mean, sqrt(normal_var));
        for (int k=0; k < KL_modes; ++k)
            normal_coefs(k) = distribution(generator);
    }

    const int ne = topology[0]->AEntityEntity(AT_elem).RowSize(iAE);
    Array<int> elem_in_AE(const_cast<int *>(topology[0]->
                              AEntityEntity(AT_elem).GetRowColumns(iAE)),
                          ne);

    normal_distribution<double> distribution(RW_mean, sqrt(RW_var));
    Vector RW_coefs(KL_modes);
    for (int s=0; s < num_samples; ++s)
    {
        cout << s << " ";

        // Generate coefficient.
        if (0 != s)
        {
            for (int k=0; k < KL_modes; ++k)
                RW_coefs(k) = distribution(generator);
            //normal_coefs += RW_coefs;
            normal_coefs = RW_coefs;
        }

        k_gf = GetKLReal(fes, KL_modes, efuncs, evals, Gauss_mean, Gauss_var,
                         normal_coefs);

        for (int k=0; k < k_gf.Size(); ++k)
            k_gf(k) = 1.0/k_gf(k);

        GridFunctionCoefficient kinv(&k_gf);

        // Update H(div) coefficient and recompute.
        DRSequence_FE->ReplaceMassIntegrator(AT_elem, jform,
                           make_unique<VectorFEMassIntegrator>(kinv), true);

        for(int l=0; l < nLevels - 1; ++l)
        {
            sequence[l]->SetSVDTol(tolSVD);
            sequence[l + 1] = sequence[l]->Coarsen();
        }

        if (0 == s)
        {
            // Obtain the local div operator (as a submatrix of the global).
            const SparseMatrix& AE_HDIVdof =
                sequence[0]->GetDofAgg(jform)->GetAEntityDof(AT_elem);
            const SparseMatrix& AE_L2dof =
                sequence[0]->GetDofAgg(jform + 1)->GetAEntityDof(AT_elem);
            Array<int> HDIVdof_in_AE(const_cast<int *>(AE_HDIVdof.GetRowColumns(iAE)),
                                        AE_HDIVdof.RowSize(iAE));
            Array<int> L2dof_in_AE(const_cast<int *>(AE_L2dof.GetRowColumns(iAE)),
                                    AE_L2dof.RowSize(iAE));
            DenseMatrix Dloc(AE_L2dof.RowSize(iAE), AE_HDIVdof.RowSize(iAE));
            sequence[0]->GetD(jform)->GetSubMatrix(L2dof_in_AE, HDIVdof_in_AE, Dloc);

            ofile << iAE << " " << ne << " " << Dloc.Height() << " " << Dloc.Width()
                  << " ";
            for (int j = 0; j < Dloc.Width(); ++j)
                for (int i = 0; i < Dloc.Height(); ++i)
                    ofile << Dloc(i, j) << " ";
            ofile << endl;
        }

        for (int k=0; k < ne; ++k)
            ofile << 1.0/k_gf(elem_in_AE[k]) << " ";

        for (int k=0; k < ne; ++k)
            ofile << k_gf(elem_in_AE[k]) << " ";

        // Obtain the local (on AE) prolongation matrices (as submatrices of the
        // global).
        for (int form = jform; form <= nDimensions; ++form)
        {
            auto DofAgg = sequence[0]->GetDofAgg(form);
            const SparseMatrix &AE_dof = DofAgg->GetAEntityDof(AT_elem);
            auto cDofHdl = sequence[1]->GetDofHandler(form);
            cDofHdl->BuildEntityDofTable(AT_elem);
            const SparseMatrix &AE_cdof = cDofHdl->GetEntityDofTable(AT_elem);
            Array<int> dof_in_AE(const_cast<int *>(AE_dof.GetRowColumns(iAE)),
                                 AE_dof.RowSize(iAE));
            Array<int> cdof_in_AE(const_cast<int *>(AE_cdof.GetRowColumns(iAE)),
                                  AE_cdof.RowSize(iAE));
            DenseMatrix Ploc(AE_dof.RowSize(iAE), AE_cdof.RowSize(iAE));
            sequence[0]->GetP(form)->GetSubMatrix(dof_in_AE, cdof_in_AE, Ploc);

            ofile << form << " " << Ploc.Height() << " " << Ploc.Width() << " ";
            for (int j = 0; j < Ploc.Width(); ++j)
                for (int i = 0; i < Ploc.Height(); ++i)
                    ofile << Ploc(i, j) << " ";
        }
        ofile << endl;
    }
    ofile.close();

    cout << "\nFinished.\n";

    return EXIT_SUCCESS;
}
