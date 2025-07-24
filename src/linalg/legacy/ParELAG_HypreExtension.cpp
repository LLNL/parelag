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

#include "linalg/legacy/ParELAG_HypreExtension.hpp"

#include "amge/DeRhamSequence.hpp"
#include "hypreExtension/hypreExtension.hpp"
#include "utilities/elagError.hpp"
#include "utilities/mpiUtils.hpp"

namespace parelag
{
using namespace mfem;

namespace HypreExtension
{

HypreBoomerAMGData::HypreBoomerAMGData():
    coarsen_type(10),
    agg_levels(1),
    relax_type(8),
    relax_sweeps(1),
    theta(0.25),
    interp_type(6),
    Pmax(4),
    print_level(0),
    dim(1),
    maxLevels(25)
{

}

HypreBoomerAMG::HypreBoomerAMG(HypreParMatrix &A) : HypreSolver(&A)
{
    HypreBoomerAMGData data;
    SetParameters(data);
}

HypreBoomerAMG::HypreBoomerAMG(HypreParMatrix &A, const HypreBoomerAMGData & data) :
    HypreSolver(&A)
{
    SetParameters(data);
}

void HypreBoomerAMG::SetParameters(const HypreBoomerAMGData & data)
{
    HYPRE_BoomerAMGCreate(&amg_precond);

    HYPRE_BoomerAMGSetCoarsenType(amg_precond, data.coarsen_type);
    HYPRE_BoomerAMGSetAggNumLevels(amg_precond, data.agg_levels);
    HYPRE_BoomerAMGSetRelaxType(amg_precond, data.relax_type);
    HYPRE_BoomerAMGSetNumSweeps(amg_precond, data.relax_sweeps);
    HYPRE_BoomerAMGSetMaxLevels(amg_precond, data.maxLevels);
    HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
    HYPRE_BoomerAMGSetMaxIter(amg_precond, 1); // one V-cycle
    HYPRE_BoomerAMGSetStrongThreshold(amg_precond, data.theta);
    HYPRE_BoomerAMGSetInterpType(amg_precond, data.interp_type);
    HYPRE_BoomerAMGSetPMaxElmts(amg_precond, data.Pmax);
    HYPRE_BoomerAMGSetPrintLevel(amg_precond, data.print_level);

    if(data.dim > 1)
        HYPRE_BoomerAMGSetNumFunctions(amg_precond, data.dim);
}

HypreBoomerAMG::~HypreBoomerAMG()
{
    HYPRE_BoomerAMGDestroy(amg_precond);
}


HypreAMSData::HypreAMSData():
    cycle_type(13),
    rlx_type(2),
    rlx_sweeps(1),
    rlx_weight(1.0),
    rlx_omega(1.0),
    dataAlpha(),
    dataBeta(),
    beta_is_zero(0)
{

}

HypreAMS::HypreAMS(HypreParMatrix &A, HypreParMatrix &G_, Array<HypreParMatrix *> & Pi_)
    : HypreSolver(&A),
      owns_G(0),
      owns_Pi(0),
      owns_Pixyz(0),
      G(&G_),
      Pi(NULL),
      Pix(Pi_[0]),
      Piy(Pi_[1]),
      Piz( (Pi_.Size() == 3) ? Pi_[2]:NULL),
      dim( Pi_.Size() )
{
    HypreAMSData data;
    SetParameters(data);
}

HypreAMS::HypreAMS(HypreParMatrix &A, HypreParMatrix & G_, Array<HypreParMatrix *> &  Pi_, const HypreAMSData & data)
    : HypreSolver(&A),
      owns_G(0),
      owns_Pi(0),
      owns_Pixyz(0),
      G(&G_),
      Pi(NULL),
      Pix(Pi_[0]),
      Piy(Pi_[1]),
      Piz( (Pi_.Size() == 3) ? Pi_[2]:NULL),
      dim( Pi_.Size() )
{

    SetParameters(data);
}

HypreAMS::HypreAMS(HypreParMatrix &A, HypreParMatrix & G_, HypreParMatrix & Pi_)
    : HypreSolver(&A),
      owns_G(0),
      owns_Pi(0),
      owns_Pixyz(0),
      G(&G_),
      Pi(&Pi_),
      Pix(NULL),
      Piy(NULL),
      Piz(NULL),
      dim(0)
{
    HypreAMSData data;
    SetParameters(data);
}

HypreAMS::HypreAMS(HypreParMatrix &A, HypreParMatrix & G_, HypreParMatrix & Pi_, const HypreAMSData & data)
    : HypreSolver(&A),
      owns_G(0),
      owns_Pi(0),
      owns_Pixyz(0),
      G(&G_),
      Pi(&Pi_),
      Pix(NULL),
      Piy(NULL),
      Piz(NULL),
      dim(0)
{
    if( Pi->GetGlobalNumCols() % G->GetGlobalNumCols() )
        mfem_error( "Pi does not have a compatible number of columns respect to G #1");

    dim = Pi->GetGlobalNumCols() / G->GetGlobalNumCols();

    if( dim < 2 || dim > 3 )
        mfem_error( "Pi does not have a compatible number of columns respect to G #2");
    SetParameters(data);
}

HypreAMS::HypreAMS(HypreParMatrix &A, DeRhamSequence * seq, const HypreAMSData & data):
    HypreSolver(&A),
    owns_G(1),
    owns_Pi(1),
    owns_Pixyz(1),
    G(NULL),
    Pi(NULL),
    Pix(NULL),
    Piy(NULL),
    Piz(NULL),
    dim(seq->GetNumberOfForms() - 1)
{
    // FIXME (trb 12/10/15): Unclear on all ways in which G, Pi, etc
    // can enter into this picutre; must investigate to figure out its
    // appropriate type; guessing shared_ptr.
    G = seq->ComputeTrueD(0).release();
    std::unique_ptr<mfem::HypreParMatrix> tmpx,tmpy,tmpz;
    if(data.cycle_type > 10 )
    {
        seq->ComputeTrueProjectorFromH1ConformingSpace(1,tmpx,tmpy,tmpz);
        Pix = tmpx.release();
        Piy = tmpy.release();
        Piz = tmpz.release();
    }
    else
        Pi = seq->ComputeTrueProjectorFromH1ConformingSpace(1).release();

    SetParameters(data);
}

void HypreAMS::SetParameters( const HypreAMSData & data )
{

    HYPRE_AMSCreate(&ams);

    HYPRE_AMSSetDimension(ams, dim); // 2D H(div) and 3D H(curl) problems
    HYPRE_AMSSetTol(ams, 0.0);
    HYPRE_AMSSetMaxIter(ams, 1); // use as a preconditioner
    HYPRE_AMSSetCycleType(ams, data.cycle_type);
    HYPRE_AMSSetPrintLevel(ams, 1);

    HYPRE_AMSSetDiscreteGradient(ams, *G);

    // TODO what if the cycle_type == 10?
    if (data.cycle_type > 10 && Pix == NULL)
        mfem_error("HypreAMS::SetParameters: if AMS cycle > 10 then Pix, Piy, Piz should be provided \n");

    if (data.cycle_type < 10 && Pi == NULL)
        mfem_error("HypreAMS::SetParameters: if AMS cycle < 10 then Pi should be provided \n");

    if (data.cycle_type > 10)
    {
        if (dim == 2)
            HYPRE_AMSSetInterpolations(ams, nullptr, *Pix, *Piy, nullptr);
        else 
            HYPRE_AMSSetInterpolations(ams, nullptr, *Pix, *Piy, *Piz);
    }
    else
        HYPRE_AMSSetInterpolations(ams, *Pi, nullptr, nullptr, nullptr);

    if(data.beta_is_zero)
        HYPRE_AMSSetBetaPoissonMatrix(ams, NULL);

    // set additional AMS options
    HYPRE_AMSSetSmoothingOptions(ams, data.rlx_type, data.rlx_sweeps, data.rlx_weight, data.rlx_omega);
    HYPRE_AMSSetAlphaAMGOptions(ams, data.dataAlpha.coarsen_type, data.dataAlpha.agg_levels, data.dataAlpha.relax_type,
                                data.dataAlpha.theta, data.dataAlpha.interp_type, data.dataAlpha.Pmax);
    if(!data.beta_is_zero)
        HYPRE_AMSSetBetaAMGOptions(ams, data.dataBeta.coarsen_type, data.dataBeta.agg_levels, data.dataBeta.relax_type,
                                   data.dataBeta.theta, data.dataBeta.interp_type, data.dataBeta.Pmax);
}

HypreAMS::~HypreAMS()
{
    HYPRE_AMSDestroy(ams);

#if MFEM_HYPRE_VERSION <= 22200
    elag_assert( hypre_ParCSRMatrixOwnsRowStarts(static_cast<hypre_ParCSRMatrix *>(*G) ) == 0 );
    try
    {
        elag_assert( hypre_ParCSRMatrixOwnsColStarts(static_cast<hypre_ParCSRMatrix *>(*G)) == 0 );
    }
    catch(...)
    {
        hypre_ParCSRMatrixOwnsColStarts(static_cast<hypre_ParCSRMatrix *>(*G)) = 0;
    }
#endif

    if(owns_G)
        delete G;

    if(owns_Pi)
        delete Pi;

    if(owns_Pixyz)
    {
        delete Pix;
        delete Piy;
        delete Piz;
    }
}

HypreADSData::HypreADSData():
    cycle_type(11),
    rlx_type(2),
    rlx_sweeps(1),
    rlx_weight(1.0),
    rlx_omega(1.0),
    dataAMG(),
    dataAMS()
{
    dataAMS.dataAlpha.relax_type = 6;
    dataAMS.dataBeta.relax_type = 6;
    dataAMS.cycle_type   = 14;

    dataAMG.relax_type = 6;
}

HypreADS::HypreADS(HypreParMatrix &A, ParFiniteElementSpace *face_fespace, const HypreADSData & data):
    HypreSolver(&A),
    G(NULL),
    C(NULL),
    ND_Pi(NULL),
    ND_Pix(NULL),
    ND_Piy(NULL),
    ND_Piz(NULL),
    RT_Pi(NULL),
    RT_Pix(NULL),
    RT_Piy(NULL),
    RT_Piz(NULL),
    owns_G(1),
    owns_C(1),
    owns_Pi(1)
{
    int p = face_fespace->GetOrder(0);
    // define the nodal and edge finite element spaces associated with face_fespace
    ParMesh *pmesh = (ParMesh *) face_fespace->GetMesh();
    FiniteElementCollection *vert_fec   = new H1_FECollection(p, 3);
    ParFiniteElementSpace *vert_fespace = new ParFiniteElementSpace(pmesh, vert_fec);
    FiniteElementCollection *edge_fec   = new ND_FECollection(p, 3);
    ParFiniteElementSpace *edge_fespace = new ParFiniteElementSpace(pmesh, edge_fec);

    // generate and set the discrete curl
    ParDiscreteLinearOperator *curl;
    curl = new ParDiscreteLinearOperator(edge_fespace, face_fespace);
    curl->AddDomainInterpolator(new CurlInterpolator);
    curl->Assemble();
    curl->Finalize();
    C = curl->ParallelAssemble();
    delete curl;

    // generate and set the discrete gradient
    ParDiscreteLinearOperator *grad;
    grad = new ParDiscreteLinearOperator(vert_fespace, edge_fespace);
    grad->AddDomainInterpolator(new GradientInterpolator);
    grad->Assemble();
    grad->Finalize();
    G = grad->ParallelAssemble();
    delete grad;

    // generate and set the Nedelec and Raviart-Thomas interpolation matrices
    ParFiniteElementSpace *vert_fespace_d;

    if (data.dataAMS.cycle_type < 10)
        vert_fespace_d = new ParFiniteElementSpace(pmesh, vert_fec, 3, Ordering::byVDIM);
    else
        vert_fespace_d = new ParFiniteElementSpace(pmesh, vert_fec, 3, Ordering::byNODES);

    ParDiscreteLinearOperator *id_ND;
    id_ND = new ParDiscreteLinearOperator(vert_fespace_d, edge_fespace);
    id_ND->AddDomainInterpolator(new IdentityInterpolator);
    id_ND->Assemble();

    if (data.dataAMS.cycle_type < 10)
    {
        id_ND->Finalize();
        ND_Pi = id_ND->ParallelAssemble();
    }
    else
    {
        Array2D<HypreParMatrix *> ND_Pi_blocks;
        id_ND->GetParBlocks(ND_Pi_blocks);
        ND_Pix = ND_Pi_blocks(0,0);
        ND_Piy = ND_Pi_blocks(0,1);
        ND_Piz = ND_Pi_blocks(0,2);
    }

    if (data.cycle_type < 10 && data.dataAMS.cycle_type > 10)
    {
        delete vert_fespace_d;
        vert_fespace_d = new ParFiniteElementSpace(pmesh, vert_fec, 3,
                                                   Ordering::byVDIM);
    }
    else if (data.cycle_type > 10 && data.dataAMS.cycle_type < 10)
    {
        delete vert_fespace_d;
        vert_fespace_d = new ParFiniteElementSpace(pmesh, vert_fec, 3,
                                                   Ordering::byNODES);
    }

    ParDiscreteLinearOperator *id_RT;
    id_RT = new ParDiscreteLinearOperator(vert_fespace_d, face_fespace);
    id_RT->AddDomainInterpolator(new IdentityInterpolator);
    id_RT->Assemble();

    if (data.cycle_type < 10)
    {
        id_RT->Finalize();
        RT_Pi = id_RT->ParallelAssemble();
    }
    else
    {
        Array2D<HypreParMatrix *> RT_Pi_blocks;
        id_RT->GetParBlocks(RT_Pi_blocks);
        RT_Pix = RT_Pi_blocks(0,0);
        RT_Piy = RT_Pi_blocks(0,1);
        RT_Piz = RT_Pi_blocks(0,2);
    }

    delete vert_fespace_d;
    delete vert_fec;
    delete vert_fespace;
    delete edge_fec;
    delete edge_fespace;

    SetParameters( data );
}

HypreADS::HypreADS(HypreParMatrix &A, HypreParMatrix &G_, HypreParMatrix &C_, Array<HypreParMatrix *> & ND_Pi_, Array<HypreParMatrix *> & RT_Pi_):
    HypreSolver(&A),
    G(&G_),
    C(&C_),
    ND_Pi(NULL),
    ND_Pix(ND_Pi_[0]),
    ND_Piy(ND_Pi_[1]),
    ND_Piz(ND_Pi_[2]),
    RT_Pi(NULL),
    RT_Pix(RT_Pi_[0]),
    RT_Piy(RT_Pi_[1]),
    RT_Piz(RT_Pi_[2]),
    owns_G(0),
    owns_C(0),
    owns_Pi(0)
{
    HypreADSData data;
    SetParameters( data );
}

HypreADS::HypreADS(
    HypreParMatrix &A, HypreParMatrix &G_, HypreParMatrix &C_, Array<HypreParMatrix *> & ND_Pi_,
    Array<HypreParMatrix *> & RT_Pi_, const HypreADSData & data)
    :
    HypreSolver(&A),
    G(&G_),
    C(&C_),
    ND_Pi(NULL),
    ND_Pix(ND_Pi_[0]),
    ND_Piy(ND_Pi_[1]),
    ND_Piz(ND_Pi_[2]),
    RT_Pi(NULL),
    RT_Pix(RT_Pi_[0]),
    RT_Piy(RT_Pi_[1]),
    RT_Piz(RT_Pi_[2]),
    owns_G(0),
    owns_C(0),
    owns_Pi(0)
{
    SetParameters(data);
}

HypreADS::HypreADS(HypreParMatrix &A, HypreParMatrix &G_, HypreParMatrix &C_,
                   HypreParMatrix &ND_Pi_, HypreParMatrix &RT_Pi_):
    HypreSolver(&A),
    G(&G_),
    C(&C_),
    ND_Pi(&ND_Pi_),
    ND_Pix(NULL),
    ND_Piy(NULL),
    ND_Piz(NULL),
    RT_Pi(&RT_Pi_),
    RT_Pix(NULL),
    RT_Piy(NULL),
    RT_Piz(NULL),
    owns_G(0),
    owns_C(0),
    owns_Pi(0)
{
    HypreADSData data;
    SetParameters(data);
}

HypreADS::HypreADS(HypreParMatrix &A, HypreParMatrix &G_, HypreParMatrix &C_,
                   HypreParMatrix &ND_Pi_, HypreParMatrix &RT_Pi_,
                   const HypreADSData & data):
    HypreSolver(&A),
    G(&G_),
    C(&C_),
    ND_Pi(&ND_Pi_),
    ND_Pix(NULL),
    ND_Piy(NULL),
    ND_Piz(NULL),
    RT_Pi(&RT_Pi_),
    RT_Pix(NULL),
    RT_Piy(NULL),
    RT_Piz(NULL),
    owns_G(0),
    owns_C(0),
    owns_Pi(0)
{
    SetParameters(data);
}

HypreADS::HypreADS(HypreParMatrix &A, DeRhamSequence * seq,
                   const HypreADSData & data):
    HypreSolver(&A),
    G(NULL),
    C(NULL),
    ND_Pi(NULL),
    ND_Pix(NULL),
    ND_Piy(NULL),
    ND_Piz(NULL),
    RT_Pi(NULL),
    RT_Pix(NULL),
    RT_Piy(NULL),
    RT_Piz(NULL),
    owns_G(1),
    owns_C(1),
    owns_Pi(1)
{
    elag_assert( seq->GetNumberOfForms() == 4 );

    // FIXME (trb 12/10/15): where else can G, C, Pi come from?
    // std::shared_ptr?
    G = seq->ComputeTrueD(0).release();
    C = seq->ComputeTrueD(1).release();

    std::unique_ptr<mfem::HypreParMatrix> tmpx,tmpy,tmpz;
    if(data.dataAMS.cycle_type > 10 )
    {
        seq->ComputeTrueProjectorFromH1ConformingSpace(1,tmpx,tmpy,tmpz);
        ND_Pix = tmpx.release();
        ND_Piy = tmpy.release();
        ND_Piz = tmpz.release();
    }
    else
        ND_Pi = seq->ComputeTrueProjectorFromH1ConformingSpace(1).release();

    if(data.cycle_type > 10 )
    {
        seq->ComputeTrueProjectorFromH1ConformingSpace(2,tmpx,tmpy,tmpz);
        RT_Pix = tmpx.release();
        RT_Piy = tmpy.release();
        RT_Piz = tmpz.release();
    }
    else
        RT_Pi = seq->ComputeTrueProjectorFromH1ConformingSpace(2).release();

    SetParameters(data);
}

void HypreADS::SetParameters( const HypreADSData & data )
{
#if MFEM_HYPRE_VERSION <= 22200
    elag_assert( hypre_ParCSRMatrixOwnsRowStarts(static_cast<hypre_ParCSRMatrix *>(*C)) == 0 );
    elag_assert( hypre_ParCSRMatrixOwnsColStarts(static_cast<hypre_ParCSRMatrix *>(*C)) == 0 );
#endif

    HYPRE_ADSCreate(&ads);

    HYPRE_ADSSetTol(ads, 0.0);
    HYPRE_ADSSetMaxIter(ads, 1); // use as a preconditioner
    HYPRE_ADSSetCycleType(ads, data.cycle_type);
    HYPRE_ADSSetPrintLevel(ads, 1);

    HYPRE_ADSSetDiscreteCurl(ads, *C);
    HYPRE_ADSSetDiscreteGradient(ads, *G);

    if(data.cycle_type < 10 && RT_Pi == NULL)
        mfem_error("HypreADS::SetParameters: if ADS cycle type < 10 then RT_Pi should be provided \n");

    if(data.cycle_type >= 10 && RT_Pix == NULL)
        mfem_error("HypreADS::SetParameters: if ADS cycle type >= 10 then RT_Pix, RT_Piy, RT_Piz should be provided \n");

    if(data.cycle_type >= 10 && RT_Piy == NULL)
        mfem_error("HypreADS::SetParameters: if ADS cycle type >= 10 then RT_Pix, RT_Piy, RT_Piz should be provided \n");

    if(data.cycle_type >= 10 && RT_Piz == NULL)
        mfem_error("HypreADS::SetParameters: if ADS cycle type >= 10 then RT_Pix, RT_Piy, RT_Piz should be provided \n");

    if(data.dataAMS.cycle_type < 10 && ND_Pi == NULL)
        mfem_error("HypreADS::SetParameters: if AMS cycle type < 10 then ND_Pi should be provided \n");

    if(data.dataAMS.cycle_type >= 10 && ND_Pix == NULL)
        mfem_error("HypreADS::SetParameters: if AMS cycle type >= 10 then ND_Pi_x, ND_Pi_y, ND_Pi_z should be provided \n");

    if(data.dataAMS.cycle_type >= 10 && ND_Piy == NULL)
        mfem_error("HypreADS::SetParameters: if AMS cycle type >= 10 then ND_Pi_x, ND_Pi_y, ND_Pi_z should be provided \n");

    if(data.dataAMS.cycle_type >= 10 && ND_Piz == NULL)
        mfem_error("HypreADS::SetParameters: if AMS cycle type >= 10 then ND_Pi_x, ND_Pi_y, ND_Pi_z should be provided \n");

    hypre_ParCSRMatrix *RT_Pi_p, *RT_Pix_p, *RT_Piy_p, *RT_Piz_p,
        *ND_Pi_p, *ND_Pix_p, *ND_Piy_p, *ND_Piz_p;
    if (data.cycle_type < 10)
    {
        RT_Pi_p = *RT_Pi;
        RT_Pix_p = nullptr;
        RT_Piy_p = nullptr;
        RT_Piz_p = nullptr;
    }
    else
    {
        RT_Pi_p = nullptr;
        RT_Pix_p = *RT_Pix;
        RT_Piy_p = *RT_Piy;
        RT_Piz_p = *RT_Piz;
    }
    if (data.dataAMS.cycle_type < 10)
    {
        ND_Pi_p = *ND_Pi;
        ND_Pix_p = nullptr;
        ND_Piy_p = nullptr;
        ND_Piz_p = nullptr;
    }
    else
    {
        ND_Pi_p = nullptr;
        ND_Pix_p = *ND_Pix;
        ND_Piy_p = *ND_Piy;
        ND_Piz_p = *ND_Piz;
    }
    HYPRE_ADSSetInterpolations(ads,
                               RT_Pi_p, RT_Pix_p, RT_Piy_p, RT_Piz_p,
                               ND_Pi_p, ND_Pix_p, ND_Piy_p, ND_Piz_p);

    // set additional ADS options
    HYPRE_ADSSetSmoothingOptions(ads, data.rlx_type, data.rlx_sweeps, data.rlx_weight, data.rlx_omega);
    HYPRE_ADSSetAMGOptions(ads, data.dataAMG.coarsen_type, data.dataAMG.agg_levels, data.dataAMG.relax_type,
                           data.dataAMG.theta, data.dataAMG.interp_type, data.dataAMG.Pmax);
    HYPRE_ADSSetAMSOptions(
        ads, data.dataAMS.cycle_type, data.dataAMS.dataAlpha.coarsen_type, data.dataAMS.dataAlpha.agg_levels,
        data.dataAMS.dataAlpha.relax_type, data.dataAMS.dataAlpha.theta, data.dataAMS.dataAlpha.interp_type,
        data.dataAMS.dataAlpha.Pmax);
}

HypreADS::~HypreADS()
{
    HYPRE_ADSDestroy(ads);

#if MFEM_HYPRE_VERSION <= 22200
    elag_assert( hypre_ParCSRMatrixOwnsRowStarts(static_cast<hypre_ParCSRMatrix *>(*G) ) == 0 );
    elag_assert( hypre_ParCSRMatrixOwnsColStarts(static_cast<hypre_ParCSRMatrix *>(*G)) == 0 );
    elag_assert( hypre_ParCSRMatrixOwnsRowStarts(static_cast<hypre_ParCSRMatrix *>(*C)) == 0 );
    try
    {
        elag_assert( hypre_ParCSRMatrixOwnsColStarts(static_cast<hypre_ParCSRMatrix *>(*C)) == 0 );
    }
    catch(...)
    {
        hypre_ParCSRMatrixOwnsColStarts(static_cast<hypre_ParCSRMatrix *>(*C)) = 0;
    }
#endif

    if(owns_G)
        delete G;
    if(owns_C)
        delete C;
    if(owns_Pi)
    {
        delete RT_Pi;
        delete RT_Pix;
        delete RT_Piy;
        delete RT_Piz;

        delete ND_Pi;
        delete ND_Pix;
        delete ND_Piy;
        delete ND_Piz;

    }
}

HypreParMatrix * ParAdd(double a, HypreParMatrix *A, double b, HypreParMatrix *B)
{
    hypre_ParCSRMatrix *C;
    hypre_ParCSRMatrixAdd2(a, *A, b, *B,&C);
    hypre_MatvecCommPkgCreate(C);

    return new HypreParMatrix(C);
}

HypreParMatrix * RAP(HypreParMatrix * Rt, HypreParMatrix * A, HypreParMatrix * P)
{
#if MFEM_HYPRE_VERSION <= 22200
    int P_owns_its_col_starts =
        hypre_ParCSRMatrixOwnsColStarts((hypre_ParCSRMatrix*)(*P));
    int Rt_owns_its_col_starts =
        hypre_ParCSRMatrixOwnsColStarts((hypre_ParCSRMatrix*)(*P));
#endif

    hypre_ParCSRMatrix * rap;
    hypre_BoomerAMGBuildCoarseOperator(*Rt,*A,*P,&rap);
    hypre_ParCSRMatrixSetNumNonzeros(rap);

#if MFEM_HYPRE_VERSION <= 22200
    /* Warning: hypre_BoomerAMGBuildCoarseOperator steals the col_starts
       from P (even if it does not own them)! */
    if (!P_owns_its_col_starts)
        hypre_ParCSRMatrixSetColStartsOwner(rap,0);

    if (!Rt_owns_its_col_starts)
        hypre_ParCSRMatrixSetRowStartsOwner(rap,0);
#endif

    return new HypreParMatrix(rap);
}

void PrintAsHypreParCSRMatrix(MPI_Comm comm, SparseMatrix & A, const std::string & fname)
{
    int localSize = A.Size();
    int localWidth = A.Width();

    Array<int> row_starts, col_starts;
    ParPartialSums_AssumedPartitionCheck(comm, localSize, row_starts);
    ParPartialSums_AssumedPartitionCheck(comm, localWidth, col_starts);

    HypreParMatrix Ad(comm, row_starts.Last(), col_starts.Last(), row_starts, col_starts, &A);
    Ad.Print(fname.c_str(), 0, 0);
}

hypre_ParCSRMatrix * DeepCopy(hypre_ParCSRMatrix * in)
{
    hypre_ParCSRMatrix * out = hypre_ParCSRMatrixCompleteClone(in);
    hypre_ParCSRMatrixCopy(in,out,1);
    return out;
}
}//namespace HypreExtension
}//namespace parelag
