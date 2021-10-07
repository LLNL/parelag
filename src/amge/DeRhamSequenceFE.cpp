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
#include <array>
#include <memory>
#include <numeric>
#include <vector>

#include "DeRhamSequenceFE.hpp"
#include "amge/bilinIntegrators.hpp"
#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "utilities/MemoryUtils.hpp"
#include "utilities/mpiUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

DeRhamSequenceFE::DeRhamSequenceFE(
    const std::shared_ptr<AgglomeratedTopology>& topo, int nforms)
    : DeRhamSequence(topo, nforms),
      di_{}, FEColl_{}, FESpace_{}, mi_{},
      Mesh_(nullptr)
{}

void DeRhamSequenceFE::ReplaceMassIntegrator(
    AgglomeratedTopology::Entity ientity,
    int jform,
    unique_ptr<BilinearFormIntegrator> m,
    bool recompute )
{
    //TODO This method is NOT very smart :(
    elag_assert(jform >= jformStart_);
    elag_assert(jform+ientity<nForms_);

    const int idx = (nForms_-1-jform)*(nForms_-jform)/2 + ientity;
    mi_[idx] = std::move(m);

    if(recompute)
        assembleLocalMass();
}

unique_ptr<SparseMatrix>
DeRhamSequenceFE::ComputeProjectorFromH1ConformingSpace(int jform) const
{
    FiniteElementSpace * conformingH1space;

    if(jform == nForms_-1)
        conformingH1space = FESpace_[0].get();
    else
        conformingH1space = new FiniteElementSpace(Mesh_,
                                                   FESpace_[0]->FEColl(),
                                                   Mesh_->Dimension(),
                                                   Ordering::byVDIM);

    unique_ptr<DiscreteLinearOperator> id{
        new DiscreteLinearOperator(conformingH1space, FESpace_[jform].get())};
    id->AddDomainInterpolator(new IdentityInterpolator());
    id->Assemble();
    id->Finalize();

    // FIXME (trb 12/14/15)
    if(jform != nForms_-1)
        delete conformingH1space;

    return unique_ptr<SparseMatrix>{id->LoseMat()};
}

DeRhamSequenceFE::~DeRhamSequenceFE()
{
}

void DeRhamSequenceFE::buildDof()
{
    int codim_base;
    for(int jform(jformStart_); jform < nForms_; ++jform)
    {
        codim_base = nForms_ - jform -1;
        //              std::cout<<"Build FE dof for space " << FESpace_[jform]->FEColl()->Name() << "(codim_base = "<< codim_base << ")" << std::endl;
        Dof_[jform] = make_unique<DofHandlerFE>(Mesh_->GetComm(),
                                                FESpace_[jform].get(),
                                                codim_base);
        Dof_[jform]->BuildEntityDofTables();
    }
}

void DeRhamSequenceFE::assembleLocalMass()
{
    //#ifdef ParELAG_ENABLE_OPENMP
    //	assembleLocalMass_omp();
    //#else
    assembleLocalMass_ser();
    //#endif
}

void DeRhamSequenceFE::assembleLocalMass_old()
{

    ElementTransformation *eltrans;
    const FiniteElement * fe;
    DenseMatrix Mloc;
    int * I, *J, *J_it;
    double *A, * A_it;
    int nnz, nLocDof;
    const int nDim = Topo_->Dimensions();

    for(int icodim(0); icodim < Topo_->Codimensions()+1; ++icodim)
        for(int jform(jformStart_); jform < nForms_ - icodim; ++jform)
        {
            //                  std::cout << " Assemble mass matrix on entity_type " << icodim << " for space " << FESpace_[jform]->FEColl()->Name() << std::endl;
            const SparseMatrix & entity_dof = Dof_[jform]->GetEntityDofTable( static_cast<AgglomeratedTopology::Entity>(icodim) );
            const int nEntities  = entity_dof.Size();
            const int nRdof      = entity_dof.NumNonZeroElems();

            I = new int[nRdof+1];
            nnz = 0;

            I[0] = nnz;
            Array<int> rdofs;
            for(int ientity(0); ientity < nEntities; ++ientity)
            {
                Dof_[jform]->GetrDof(static_cast<AgglomeratedTopology::Entity>(icodim), ientity, rdofs);
                nLocDof = rdofs.Size();
                nnz += nLocDof*nLocDof;
                for(int * idof(rdofs.GetData()); idof != rdofs.GetData()+nLocDof; ++idof)
                    I[*idof+1] = nLocDof;
            }

            std::partial_sum(I, I+nRdof+1, I);

            J = new int[nnz];
            A = new double[nnz];

            for(int ientity(0); ientity < nEntities; ++ientity)
            {
                fe = GetFE(jform, icodim, ientity);
                eltrans = GetTransformation(icodim, ientity);
                nLocDof = fe->GetDof();
                Dof_[jform]->GetrDof(static_cast<AgglomeratedTopology::Entity>(icodim), ientity, rdofs);

                const int idx = (nForms_-1-jform)*(nForms_-jform)/2 + icodim;
                mi_[idx]->AssembleElementMatrix(*fe, *eltrans, Mloc);
                for(int iloc(0); iloc < nLocDof; ++iloc)
                {
                    int idof = rdofs[iloc];
                    J_it = J+I[ idof ];
                    A_it = A+I[ idof ];
                    for(int jloc(0); jloc < nLocDof; ++jloc)
                    {
                        *(J_it++) = rdofs[jloc];
                        *(A_it++) = Mloc(iloc, jloc);
                    }
                }
            }

            const int idx = (nDim-jform)*(nForms_-jform)/2 + icodim;

            M_[idx] = make_unique<SparseMatrix>(I,J,A,nRdof,nRdof);
        }

#if 0
    {
        std::ofstream fid("M.dat");
        for(int icodim(0); icodim < nForms_; ++icodim)
            for(int jform(0); jform < nForms_-icodim; ++jform)
            {
                fid << "Mass matrix on entity_type " << icodim << " for space " << FESpace_[jform]->FEColl()->Name() << std::endl;
                M_(icodim, jform)->PrintMatlab(fid);
            }
    }
#endif
}

void DeRhamSequenceFE::assembleLocalMass_ser()
{
    ElementTransformation *eltrans;
    const FiniteElement * fe;
    int nLocDof;
    const int nDim = Topo_->Dimensions();

    for(int icodim(0); icodim < Topo_->Codimensions()+1; ++icodim)
        for(int jform(jformStart_); jform < nForms_ - icodim; ++jform)
        {
            const int idx = (nDim-jform)*(nForms_-jform)/2 + icodim;

            const SparseMatrix & entity_dof =
                Dof_[jform]->GetEntityDofTable(
                    static_cast<AgglomeratedTopology::Entity>(icodim) );
            const int nEntities  = entity_dof.Size();
            ElementalMatricesContainer mass(nEntities);
            for(int ientity(0); ientity < nEntities; ++ientity)
            {
                fe = GetFE(jform, icodim, ientity);
                eltrans = GetTransformation(icodim, ientity);
                nLocDof = fe->GetDof();
                auto Mloc = make_unique<DenseMatrix>(nLocDof);

                mi_[idx]->AssembleElementMatrix(*fe, *eltrans, *Mloc);
                mass.SetElementalMatrix(ientity,std::move(Mloc));
            }

            M_[idx] = mass.GetAsSparseMatrix();
        }

#if 0
    {
        std::ofstream fid("M.dat");
        for(int icodim(0); icodim < nForms_; ++icodim)
            for(int jform(0); jform < nForms_-icodim; ++jform)
            {
                fid << "Mass matrix on entity_type " << icodim << " for space " << FESpace_[jform]->FEColl()->Name() << std::endl;
                M_(icodim, jform)->PrintMatlab(fid);
            }
    }
#endif
}


void DeRhamSequenceFE::assembleLocalMass_omp()
{
    std::cout << "assembleLocalMass_opm\n";
    int * I, *J;
    double *A;
    int nnz, nLocDof;
    const int nDim = Topo_->Dimensions();

    for(int icodim(0); icodim < Topo_->Codimensions()+1; ++icodim)
        for(int jform(jformStart_); jform < nForms_ - icodim; ++jform)
        {
            //                  std::cout << " Assemble mass matrix on entity_type " << icodim << " for space " << FESpace_[jform]->FEColl()->Name() << std::endl;
            const SparseMatrix & entity_dof = Dof_[jform]->GetEntityDofTable( static_cast<AgglomeratedTopology::Entity>(icodim) );
            const int nEntities  = entity_dof.Size();
            const int nRdof      = entity_dof.NumNonZeroElems();

            I = new int[nRdof+1];
            nnz = 0;

            I[0] = nnz;
            {
                Array<int> rdofs;
                for(int ientity(0); ientity < nEntities; ++ientity)
                {
                    Dof_[jform]->GetrDof(static_cast<AgglomeratedTopology::Entity>(icodim), ientity, rdofs);
                    nLocDof = rdofs.Size();
                    nnz += nLocDof*nLocDof;
                    for(int * idof(rdofs.GetData()); idof != rdofs.GetData()+nLocDof; ++idof)
                        I[*idof+1] = nLocDof;
                }
            }

            std::partial_sum(I, I+nRdof+1, I);

            J = new int[nnz];
            A = new double[nnz];
#ifdef ParELAG_ENABLE_OPENMP
#pragma omp parallel default(shared) private(nLocDof)
#endif
            {
                IsoparametricTransformation *eltrans = new IsoparametricTransformation;
                const FiniteElement * fe;
                DenseMatrix Mloc;
                Array<int> rdofs;

                int *J_it;
                double *A_it;

                int idof, iloc, jloc;

#ifdef ParELAG_ENABLE_OPENMP
#pragma omp for schedule(guided)
#endif
                for(int ientity = 0; ientity < nEntities; ++ientity)
                {
                    fe = GetFE(jform, icodim, ientity);
                    GetTransformation(icodim, ientity, *eltrans);
                    nLocDof = fe->GetDof();
                    Dof_[jform]->GetrDof(static_cast<AgglomeratedTopology::Entity>(icodim), ientity, rdofs);

                    const int idx = (nForms_-1-jform)*(nForms_-jform)/2 + icodim;
                    mi_[idx]->AssembleElementMatrix(*fe, *eltrans, Mloc);
                    for(iloc = 0; iloc < nLocDof; ++iloc)
                    {
                        idof = rdofs[iloc];
                        J_it = J+I[ idof ];
                        A_it = A+I[ idof ];
                        for(jloc = 0; jloc < nLocDof; ++jloc)
                        {
                            *(J_it++) = rdofs[jloc];
                            *(A_it++) = Mloc(iloc, jloc);
                        }
                    }
                }
            }

            const int idx = (nDim-jform)*(nForms_-jform)/2 + icodim;
            M_[idx] = make_unique<SparseMatrix>(I,J,A,nRdof,nRdof);
        }
}


void DeRhamSequenceFE::assembleDerivative()
{
    for(int jform(jformStart_); jform < nForms_-1; ++jform)
    {
        //              std::cout << " Assemble Differential Operator from "<< FESpace_[jform]->FEColl()->Name() << " to " << FESpace_[jform+1]->FEColl()->Name() << std::endl;
        DiscreteLinearOperator interp(FESpace_[jform].get(),
                                      FESpace_[jform+1].get());
        interp.AddDomainInterpolator(di_[jform].get());
        interp.Assemble();
        interp.Finalize();
        LoseInterpolators(interp);
        D_[jform].reset(interp.LoseMat());
    }

#if 0
    {
        std::ofstream fid("D.dat");
        for(int jform(0); jform < nForms_-1; ++jform)
        {
            fid << "Differential Operator from "<< FESpace_[jform]->FEColl()->Name() << " to " << FESpace_[jform]->FEColl()->Name() << std::endl;
            D_[jform]->PrintMatlab(fid);
        }
    }
#endif
}

void DeRhamSequenceFE::showP(int jform, SparseMatrix & P, Array<int> & parts)
{
    elag_assert(jform < nForms_);
    elag_assert(jform >= jformStart_ );
    elag_assert(FESpace_[jform]->GetNDofs() == P.Size());

    int num_procs, myid;
    MPI_Comm comm = Mesh_->GetComm();

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    GridFunction u(FESpace_[jform].get());
    Array<int> cols;
    Vector srow;
    unique_ptr<SparseMatrix> P_t(Transpose(P));
    char vishost[] = "localhost";
    int  visport   = 19916;
    osockstream sock (visport, vishost);
    for (int i(0); i < P.Width(); ++i)
    {
        u = 0.0;
        P_t->GetRow(i, cols, srow);
        u.SetSubVector(cols, srow);
        sock << "parallel " << num_procs << " " << myid << "\n";
        sock << "solution\n";
        this->Mesh_->PrintWithPartitioning(parts.GetData(), sock);
        u.Save(sock);
        sock.send();
        sock << "pause \n";
    }
}

void DeRhamSequenceFE::show(int jform, MultiVector & v)
{
    elag_assert(jform < nForms_);
    elag_assert(jform >= jformStart_);
    elag_assert(FESpace_[jform]->GetNDofs() == v.Size());

    int num_procs, myid;
    MPI_Comm comm = Mesh_->GetComm();

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    GridFunction u;
    Vector v_view;
    char vishost[] = "localhost";
    int  visport   = 19916;
    osockstream sock (visport, vishost);

    for (int i(0); i < v.NumberOfVectors(); ++i)
    {
        v.GetVectorView(i, v_view);
        u.MakeRef(FESpace_[jform].get(), v_view, 0);
        sock << "parallel " << num_procs << " " << myid << "\n";
        sock << "solution\n";
        this->Mesh_->Print(sock);
        u.Save(sock);
        sock.send();
        sock << "pause \n";
    }
    sock.close();
}

void DeRhamSequenceFE::ShowTrueData(int jform, MultiVector & true_v)
{
   const int nv = true_v.NumberOfVectors();
   MultiVector v(nv, GetNumDofs(jform));
   Vector v_view, true_v_view;
   for(int i(0); i < nv; ++i)
   {
       true_v.GetVectorView(i, true_v_view);
       v.GetVectorView(i,v_view);
       Dof_[jform]->GetDofTrueDof().Distribute(true_v_view, v_view);
   }

   show(jform, v);
}

void DeRhamSequenceFE::ExportGLVis(int jform, Vector & v, std::ostream & os)
{
    elag_assert(v.Size() == this->GetNumberOfDofs(jform));
    GridFunction u;
    u.MakeRef(FESpace_[jform].get(), v, 0);
    u.Save(os);
}

unique_ptr<MultiVector> DeRhamSequenceFE::InterpolateScalarTargets(
    int jform, const Array<Coefficient *> & targets)
{
    elag_assert( jform == 0 || jform == nForms_ - 1);
    // jform is a VectorFiniteElement!

    const int size = FESpace_[jform]->GetNDofs();
    const int nTargets = targets.Size();

    unique_ptr<MultiVector> outMV(new MultiVector(nTargets, size));
    Vector MVview;
    GridFunction gf;

    for(int itarget(0); itarget < nTargets; ++itarget)
    {
        outMV->GetVectorView(itarget, MVview);
        gf.MakeRef(FESpace_[jform].get(), MVview, 0);
        gf.ProjectCoefficient(*targets[itarget]);
    }

    return outMV;
}

unique_ptr<MultiVector> DeRhamSequenceFE::InterpolateVectorTargets(
    int jform, const Array<VectorCoefficient *> & targets)
{
    elag_assert( jform != 0 && jform != nForms_ - 1);
    // jform is a ScalarFiniteElement!

    const int size = FESpace_[jform]->GetNDofs();
    const int nTargets = targets.Size();

    unique_ptr<MultiVector> outMV(new MultiVector(nTargets, size));
    Vector MVview;
    GridFunction gf;

    for(int itarget(0); itarget < nTargets; ++itarget)
    {
        outMV->GetVectorView(itarget, MVview);
        gf.MakeRef(FESpace_[jform].get(), MVview, 0);
        gf.ProjectCoefficient(*targets[itarget]);
    }

    return outMV;
}

void DeRhamSequenceFE::ProjectCoefficient(int jform,Coefficient & c,Vector & v)
{
    v.SetSize(FESpace_[jform]->GetNDofs());
    GridFunction gf;
    gf.MakeRef(FESpace_[jform].get(), v, 0);
    gf.ProjectCoefficient(c);
}

void DeRhamSequenceFE::ProjectVectorCoefficient(
    int jform,VectorCoefficient & c,Vector & v)
{
    v.SetSize(FESpace_[jform]->GetNDofs());
    GridFunction gf;
    gf.MakeRef(FESpace_[jform].get(), v, 0);
    gf.ProjectCoefficient(c);
}

const FiniteElement * DeRhamSequenceFE::GetFE(
    int jform, int ientity_type, int ientity) const
{
    if(jform + ientity_type > nForms_)
        mfem_error("Wrong combination of jform and ientity_type");

    if(Mesh_->Dimension() == 3)
    {
        switch(ientity_type)
        {
        case AgglomeratedTopology::ELEMENT:
            return FESpace_[jform]->GetFE(ientity);
            break;
        case AgglomeratedTopology::FACET:
            return FESpace_[jform]->FEColl()->FiniteElementForGeometry(Mesh_->GetFaceBaseGeometry(ientity));
            break;
        case AgglomeratedTopology::RIDGE:
            return FESpace_[jform]->FEColl()->FiniteElementForGeometry(Geometry::SEGMENT);
        case AgglomeratedTopology::PEAK:
            return FESpace_[jform]->FEColl()->FiniteElementForGeometry(Geometry::POINT);
            break;
        default:
            mfem_error("Wrong ientity_type");
            return nullptr;
        }
    }
    else
    {
        switch(ientity_type)
        {
        case AgglomeratedTopology::ELEMENT:
            return FESpace_[jform]->GetFE(ientity);
            break;
        case AgglomeratedTopology::FACET:
            return FESpace_[jform]->FEColl()->FiniteElementForGeometry(Geometry::SEGMENT);
        case AgglomeratedTopology::RIDGE:
            return FESpace_[jform]->FEColl()->FiniteElementForGeometry(Geometry::POINT);
            break;
        default:
            mfem_error("Wrong ientity_type");
            return nullptr;
        }
    }
}

ElementTransformation * DeRhamSequenceFE::GetTransformation(
    int ientity_type, int ientity) const
{
    if(Mesh_->Dimension() == 3)
    {
        switch(ientity_type)
        {
        case AgglomeratedTopology::ELEMENT:
            return Mesh_->GetElementTransformation(ientity);
            break;
        case AgglomeratedTopology::FACET:
            return Mesh_->GetFaceTransformation(ientity);
            break;
        case AgglomeratedTopology::RIDGE:
            return Mesh_->GetEdgeTransformation(ientity);
            break;
        case AgglomeratedTopology::PEAK:
            return nullptr;
            break;
        default:
            mfem_error("Wrong ientity_type");
            return nullptr;
        }
    }
    else
    {
        switch(ientity_type)
        {
        case AgglomeratedTopology::ELEMENT:
            return Mesh_->GetElementTransformation(ientity);
            break;
        case AgglomeratedTopology::FACET:
            return Mesh_->GetEdgeTransformation(ientity);
        case AgglomeratedTopology::RIDGE:
            return nullptr;
            break;
        default:
            mfem_error("Wrong ientity_type");
            return nullptr;
        }
    }
}

void DeRhamSequenceFE::GetTransformation(
    int ientity_type, int ientity, IsoparametricTransformation & tr) const
{
    if(Mesh_->Dimension() == 3)
    {
        switch(ientity_type)
        {
        case AgglomeratedTopology::ELEMENT:
            Mesh_->GetElementTransformation(ientity, &tr);
            break;
        case AgglomeratedTopology::FACET:
            Mesh_->GetFaceTransformation(ientity, &tr);
            break;
        case AgglomeratedTopology::RIDGE:
            Mesh_->GetEdgeTransformation(ientity, &tr);
            break;
        case AgglomeratedTopology::PEAK:
            tr.GetPointMat().SetSize(Mesh_->Dimension(), 0);
            tr.SetFE(nullptr);
            break;
        default:
            mfem_error("Wrong ientity_type");
        }
    }
    else
    {
        switch(ientity_type)
        {
        case AgglomeratedTopology::ELEMENT:
            Mesh_->GetElementTransformation(ientity, &tr);
            break;
        case AgglomeratedTopology::FACET:
            Mesh_->GetEdgeTransformation(ientity, &tr);
            break;
        case AgglomeratedTopology::RIDGE:
            tr.GetPointMat().SetSize(Mesh_->Dimension(), 0);
            tr.SetFE(nullptr);
            break;
        default:
            mfem_error("Wrong ientity_type");
        }
    }
}

void DeRhamSequenceFE::Update()
{
    for (int i=0; i < nForms_; ++i)
        FESpace_[i]->Update();

    buildDof();
    assembleLocalMass();
    assembleDerivative();
}

//-----------------------------------------------------------------------------

/// FIXME (trb 12/15/15): Shouldn't the topo have the mesh already?
DeRhamSequence3D_FE::DeRhamSequence3D_FE(
    const std::shared_ptr<AgglomeratedTopology>& topo,ParMesh * mesh,int order,bool assemble,bool assemble_mass)
    : DeRhamSequenceFE(topo, 4)
{
    Mesh_ = mesh;
    constexpr int nDimensions{3};

    elag_assert(nDimensions == Mesh_->Dimension());

    // Unfortunately, no cleaner way to do this.
    FEColl_[0] = make_unique<H1_FECollection>(order+1,nDimensions);
    FEColl_[1] = make_unique<ND_FECollection>(order+1,nDimensions);
    FEColl_[2] = make_unique<RT_FECollection>(order,nDimensions);
    FEColl_[3] = make_unique<L2_FECollection>(order,nDimensions);

    for(int i(0); i < nDimensions+1; ++i)
        this->FESpace_[i] = make_unique<FiniteElementSpace>(Mesh_,FEColl_[i].get());

    this->di_[0] = make_unique<GradientInterpolator>();
    this->di_[1] = make_unique<CurlInterpolator>();
    this->di_[2] = make_unique<DivergenceInterpolator2>();


    // Integrators for L2
    mi_[0] = make_unique<MassIntegrator>();//volumes

    // Integrators for HDiv
    mi_[1] = make_unique<VectorFEMassIntegrator>();//volumes
    mi_[2] = make_unique<VolumetricFEMassIntegrator>();//faces

    // Integrators for HCurl
    mi_[3] = make_unique<VectorFEMassIntegrator>();//volumes
    mi_[4] = make_unique<ND_3D_FacetMassIntegrator>();//faces
    mi_[5] = make_unique<VolumetricFEMassIntegrator>();//edges

    //Integrators on volumes
    mi_[6] = make_unique<MassIntegrator>();//volumes
    mi_[7] = make_unique<MassIntegrator>();//faces
    mi_[8] = make_unique<MassIntegrator>();//edges
    mi_[9] = make_unique<PointFEMassIntegrator>();//points

    if (assemble)
    {
        buildDof();
        if (assemble_mass)
            assembleLocalMass();
        assembleDerivative();
    }

    mfem::ConstantCoefficient one(1.0);
    ProjectCoefficient(nDimensions, one, L2_const_rep_);
}

DeRhamSequence3D_FE::~DeRhamSequence3D_FE()
{
}

void DeRhamSequence3D_FE::computePVTraces(
    AgglomeratedTopology::Entity icodim, Vector & pv)
{
    const int jform = nForms_ - 1 - icodim;

    switch(icodim)
    {
    case AgglomeratedTopology::ELEMENT:
        InterpolatePV_L2(FESpace_[jform].get(),
                         Topo_->AEntityEntity(AgglomeratedTopology::ELEMENT),
                         pv);
        break;
    case AgglomeratedTopology::FACET:
        InterpolatePV_HdivTraces(
            FESpace_[jform].get(),
            Topo_->AEntityEntity(AgglomeratedTopology::FACET),
            pv);
        break;
    case AgglomeratedTopology::RIDGE:
        InterpolatePV_HcurlTraces(
            FESpace_[jform].get(),
            Topo_->AEntityEntity(AgglomeratedTopology::RIDGE),
            pv );
        break;
    case AgglomeratedTopology::PEAK:
        InterpolatePV_H1Traces(FESpace_[jform].get(),
                               Topo_->AEntityEntity(AgglomeratedTopology::PEAK),
                               pv);
        break;
    default:
        mfem_error("DeRhamSequence3D_FE::computePVTraces(Topology::entity icodim, Vector & pv)");
    }
}

DeRhamSequence2D_Hdiv_FE::DeRhamSequence2D_Hdiv_FE(
    const std::shared_ptr<AgglomeratedTopology>& topo, ParMesh * mesh, int order, bool assemble, bool assemble_mass)
    : DeRhamSequenceFE(topo, 3)
{
    Mesh_ = mesh;
    constexpr int nDimensions{2};

    elag_assert(nDimensions == Mesh_->Dimension());

    FEColl_[0] = make_unique<H1_FECollection>(order+1, nDimensions);
    FEColl_[1] = make_unique<RT_FECollection>(order, nDimensions);
    FEColl_[2] = make_unique<L2_FECollection>(order, nDimensions);

    for(int i(0); i < nDimensions+1; ++ i)
        this->FESpace_[i] = make_unique<FiniteElementSpace>(Mesh_,FEColl_[i].get());

    this->di_[0] = make_unique<GradientInterpolator>();
    this->di_[1] = make_unique<DivergenceInterpolator2>();

    // Integrators on L2
    mi_[0] = make_unique<MassIntegrator>();//faces

    // Integrators on HDIV
    mi_[1] = make_unique<VectorFEMassIntegrator>();//faces
    mi_[2] = make_unique<VolumetricFEMassIntegrator>();//edges

    //Integrators on faces
    mi_[3] = make_unique<MassIntegrator>();//faces
    mi_[4] = make_unique<MassIntegrator>();//edges
    mi_[5] = make_unique<PointFEMassIntegrator>();//points

    if (assemble)
    {
        buildDof();
        if (assemble_mass)
            assembleLocalMass();
        assembleDerivative();
    }

    mfem::ConstantCoefficient one(1.0);
    ProjectCoefficient(nDimensions, one, L2_const_rep_);
}

void DeRhamSequence2D_Hdiv_FE::computePVTraces(
    AgglomeratedTopology::Entity icodim, Vector & pv)
{
    const int jform = nForms_ - 1 - icodim;
    switch(icodim)
    {
    case AgglomeratedTopology::ELEMENT:
        InterpolatePV_L2(FESpace_[jform].get(),
                         Topo_->AEntityEntity(AgglomeratedTopology::ELEMENT),
                         pv);
        break;
    case AgglomeratedTopology::FACET:
        InterpolatePV_HdivTraces(
            FESpace_[jform].get(),
            Topo_->AEntityEntity(AgglomeratedTopology::FACET),
            pv );
        break;
    case AgglomeratedTopology::RIDGE:
        InterpolatePV_H1Traces(
            FESpace_[jform].get(),
            Topo_->AEntityEntity(AgglomeratedTopology::RIDGE),
            pv );
        break;
    default:
        mfem_error("DeRhamSequence2D_Hdiv_FE::computePVTraces(Topology::entity icodim, Vector & pv)");
    }
}

DeRhamSequence2D_Hdiv_FE::~DeRhamSequence2D_Hdiv_FE()
{
}

void InterpolatePV_L2(const FiniteElementSpace * fespace,
                      SparseMatrix const&,
                      Vector & AE_Interpolant)
{
    GridFunction gf;
    gf.MakeRef(const_cast<FiniteElementSpace *>(fespace), AE_Interpolant, 0);
    ConstantCoefficient ones(1);

    gf.ProjectCoefficient(ones);
}


void InterpolatePV_HdivTraces(const FiniteElementSpace * fespace,
                              const SparseMatrix & AF_facet,
                              Vector & AF_Interpolant)
{
    const int nAF = AF_facet.Size();
    const int ndofs = fespace->GetNDofs();
    Mesh * mesh = fespace->GetMesh();
    int nDim = mesh->Dimension();

    PARELAG_ASSERT(AF_Interpolant.Size() == ndofs);

    const int    * i_AF_facet = AF_facet.GetI();
    const int    * j_AF_facet = AF_facet.GetJ();
    const double * a_AF_facet = AF_facet.GetData();

    Array<int> vdofs(0);

    const FiniteElement * fe;
    const IntegrationRule * nodes;

    Vector localVector;
    for (int iAF(0); iAF < nAF; ++iAF)
    {
        for (const int * ifc = j_AF_facet + i_AF_facet[iAF];
             ifc != j_AF_facet + i_AF_facet[iAF+1];
             ++ifc, ++a_AF_facet)
        {
            fe = fespace->FEColl()->FiniteElementForGeometry(
                mesh->GetFaceBaseGeometry(*ifc));
            int ndofs = fe->GetDof();
            vdofs.SetSize(ndofs);

            if(nDim == 3)
                fespace->GetFaceVDofs(*ifc, vdofs);
            else /* nDim == 2 */
                fespace->GetEdgeVDofs(*ifc, vdofs);

            ElementTransformation * tr = mesh->GetFaceTransformation(*ifc);
            nodes = &fe->GetNodes();
            localVector.SetSize(ndofs);
            for (int i = 0; i < ndofs; i++)
            {
                const IntegrationPoint &ip = nodes->IntPoint(i);
                tr->SetIntPoint(&ip);
                localVector(i) = *(a_AF_facet)*tr->Weight();
            }
            AF_Interpolant.SetSubVector(vdofs,localVector);
        }
    }
}

void InterpolatePV_HcurlTraces(const FiniteElementSpace * fespace,
                               const SparseMatrix & AR_ridge,
                               Vector & AR_Interpolant)
{
    const int nAR = AR_ridge.Size();
    const int ndofs = fespace->GetNDofs();
    Mesh * mesh = fespace->GetMesh();

    PARELAG_ASSERT(AR_Interpolant.Size() == ndofs);

    const int    * i_AR_ridge = AR_ridge.GetI();
    const int    * j_AR_ridge = AR_ridge.GetJ();
    const double * a_AR_ridge = AR_ridge.GetData();

    const FiniteElement * fe =
        fespace->FEColl()->FiniteElementForGeometry(Geometry::SEGMENT);
    const IntegrationRule * nodes = &fe->GetNodes();
    const int ndof_ridge = fe->GetDof();

    AR_Interpolant.SetSize( ndofs );
    AR_Interpolant = 0.0;

    Array<int> vdofs(ndof_ridge);
    Vector localVector;

    for (int iAR(0); iAR < nAR; ++iAR)
    {
        for (const int * irg = j_AR_ridge + i_AR_ridge[iAR];
             irg != j_AR_ridge + i_AR_ridge[iAR+1];
             ++irg, ++a_AR_ridge)
        {
            fespace->GetEdgeVDofs(*irg, vdofs);
            ElementTransformation * tr = mesh->GetEdgeTransformation(*irg);
            localVector.SetSize(ndof_ridge);
            for (int i = 0; i < ndof_ridge; i++)
            {
                const IntegrationPoint &ip = nodes->IntPoint(i);
                tr->SetIntPoint(&ip);
                localVector(i) = *(a_AR_ridge)*tr->Weight();
            }
            AR_Interpolant.SetSubVector(vdofs, localVector);
        }
    }
}

void InterpolatePV_H1Traces(const FiniteElementSpace * fespace,
                            const SparseMatrix & AP_peak,
                            Vector & AP_Interpolant)
{
    const int ndofs = fespace->GetNDofs();
    const int nnz = AP_peak.NumNonZeroElems();

    PARELAG_ASSERT(AP_Interpolant.Size() == ndofs);

    AP_Interpolant.SetSize(ndofs);
    AP_Interpolant = 0.0;

    const int * j_AP_peak = AP_peak.GetJ();
    const int * end = j_AP_peak+nnz;

    double * val = AP_Interpolant.GetData();
    for(; j_AP_peak != end; ++j_AP_peak)
        val[*j_AP_peak] =1.0;
}

void DeRhamSequenceFE::SetUpscalingTargets(int nDimensions, int upscalingOrder,
                                           int form_start)
{
    if (form_start < 0)
        form_start = jformStart_;

    Array<Coefficient *> L2coeff;
    Array<VectorCoefficient *> Hdivcoeff;
    Array<VectorCoefficient *> Hcurlcoeff; // only needed for form < 2
    Array<Coefficient *> H1coeff; // only needed for form < 1

    fillCoefficientArray(nDimensions, upscalingOrder, L2coeff);
    fillVectorCoefficientArray(nDimensions, upscalingOrder, Hdivcoeff);
    if (form_start < 2)
        fillVectorCoefficientArray(nDimensions, upscalingOrder, Hcurlcoeff);
    if (form_start < 1)
        fillCoefficientArray(nDimensions, upscalingOrder+1, H1coeff);

    std::vector<unique_ptr<MultiVector>> targets(GetNumberOfForms());

    for (int jform = 0; jform < GetNumberOfForms(); ++jform)
        targets[jform] = nullptr;
    int jform(0);
    if (form_start < 1)
    {
        targets[jform] = InterpolateScalarTargets(jform, H1coeff);
    }
    ++jform;
    if (form_start < 2)
    {
        if (nDimensions == 3)
        {
            targets[jform] = InterpolateVectorTargets(jform, Hcurlcoeff);
            ++jform;
        }
    }
    else
    {
        // targets[jform]= static_cast<MultiVector *>(NULL);
        ++jform;
    }
    targets[jform] = InterpolateVectorTargets(jform, Hdivcoeff);
    ++jform;
    targets[jform] = InterpolateScalarTargets(jform, L2coeff);
    ++jform;

    freeCoeffArray(L2coeff);
    freeCoeffArray(Hdivcoeff);
    freeCoeffArray(Hcurlcoeff);
    freeCoeffArray(H1coeff);

    Array<MultiVector *> targets_in(targets.size());
    for (int ii = 0; ii < targets_in.Size(); ++ii)
        targets_in[ii] = targets[ii].get();
    SetTargets(targets_in);
}

void DeRhamSequenceFE::DEBUG_CheckNewLocalMassAssembly()
{
    MPI_Comm comm;
    std::stringstream os;

    std::array<unique_ptr<SparseMatrix>,10> M_ser;
    std::array<unique_ptr<SparseMatrix>,10> M_old;

    //Array2D<SparseMatrix *> M_ser( M_.NumRows(), M_.NumCols() );
    //Array2D<SparseMatrix *> M_old( M_.NumRows(), M_.NumCols() );
    //M_old = nullptr;
    M_.swap(M_old);

    StopWatch chrono;

    chrono.Start();
    assembleLocalMass_ser();
    chrono.Stop();
    os << "assembleLocalMass_ser took " << chrono.RealTime() << "\n";
    M_.swap(M_ser);

    chrono.Clear();
    chrono.Start();
    assembleLocalMass_old();
    chrono.Stop();
    os << "assembleLocalMass_old took " << chrono.RealTime() << "\n";
    M_.swap(M_old);

    const int nDim = Topo_->Dimensions();
    for (int icodim(0); icodim < nForms_; ++icodim)
    {
        for (int jform(0); jform < nForms_ - icodim; ++jform)
        {
            std::stringstream name_ser;
            std::stringstream name_new;
            name_ser << "M_ser(" << icodim <<" , " << jform << ")";
            name_ser << "M_old(" << icodim <<" , " << jform << ")";

            const int idx = (nDim-jform)*(nForms_-jform)/2 + icodim;
            AreAlmostEqual(*(M_ser[idx]), *(M_old[idx]), name_ser.str(),
                           name_new.str(), 1e-9, true, os);
        }
    }
    comm = Topo_->GetComm();
    SerializedOutput(comm, std::cout, os.str());
}

}//namespace parelag
