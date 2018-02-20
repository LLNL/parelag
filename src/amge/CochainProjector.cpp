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
#include <memory>

#include "CochainProjector.hpp"

#include "linalg/dense/ParELAG_InnerProduct.hpp"
#include "linalg/dense/ParELAG_LAPACK.hpp"
#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

CochainProjector::CochainProjector(AgglomeratedTopology *,
                                   DofHandler * cDof,
                                   DofAgglomeration * dofAgg,
                                   SparseMatrix * P)
    : cDof_(dynamic_cast<DofHandlerALG*>(cDof) ),
      dofAgg_(dofAgg),
      P_(P),
      dofLinearFunctional_(cDof->GetMaxCodimensionBaseForDof() + 1),
      Pi_{nullptr}
{
    // FIXME (trb 12/23/15): cTopo is unused. There was a (private)
    // data member called cTopo_, but it was unused, so it was
    // deleted. This argument should either be used or removed.

    PARELAG_TEST_FOR_EXCEPTION(
        cDof_ == nullptr,
        std::runtime_error,
        "CochainProjector Ctor: cDof must be non-null DofHandlerALG");

    auto dlf_size = dofLinearFunctional_.size();
    for(auto i = decltype(dlf_size){0}; i < dlf_size; ++i)
        dofLinearFunctional_[i] = make_unique<ElementalMatricesContainer>(
                cDof_->GetNumberEntities(
                    static_cast<AgglomeratedTopology::Entity>(i)));
}

void CochainProjector::CreateDofFunctional(
    AgglomeratedTopology::Entity entity_type,
    int entity_id,
    const MultiVector & localProjector,
    const SparseMatrix & M_ii)
{
    PARELAG_TEST_FOR_EXCEPTION(
        entity_type > cDof_->GetMaxCodimensionBaseForDof(),
        std::runtime_error,
        "CochainProjector::CreateDofFunctional(): entity_type too big." );

    PARELAG_TEST_FOR_EXCEPTION(
        entity_id >= cDof_->GetNumberEntities(entity_type),
        std::runtime_error,
        "CochainProjector::CreateDofFunctional(): entity_id too big.");

    PARELAG_TEST_FOR_EXCEPTION(
        M_ii.Width() != localProjector.Size(),
        std::runtime_error,
        "CochainProjector::CreateDofFunctional(): "
        "M_ii width not equal to localProjector's size.");

    PARELAG_TEST_FOR_EXCEPTION(
        M_ii.Size() != localProjector.Size(),
        std::runtime_error,
        "CochainProjector::CreateDofFunctional(): "
        "M_ii size not equal to localProjector's size.");


    const int nFineDof = localProjector.Size();
    const int nCoarseDof = localProjector.NumberOfVectors();
    auto dof_lin_func = make_unique<DenseMatrix>(nCoarseDof, nFineDof);

    if(nCoarseDof != 0)
    {
        MultiVector MlP(nCoarseDof, nFineDof);
        MatrixTimesMultiVector(M_ii, localProjector, MlP);
        MlP.CopyToDenseMatrixT(0, nCoarseDof, 0, nFineDof, *dof_lin_func);

        StdInnerProduct dot( nFineDof );
        DenseMatrix cLocMass_ii(nCoarseDof, nCoarseDof);
        dot(localProjector, MlP, cLocMass_ii);

        PARELAG_TEST_FOR_EXCEPTION(
            cLocMass_ii.CheckFinite(),
            std::runtime_error,
            "CochainProjector::CreateDofFunctional(): "
            "Something bad with cLocMass_ii");

        double * data_cLocMass = cLocMass_ii.Data();
        double * data_MlP = dof_lin_func->Data();

        // Compute the LU P factorization of xLocMass and Solve cLocMass^-1 MlP'
        {
            char trans = 'N';
            int  n     = nCoarseDof;
            int  nrhs  = nFineDof;
            auto ipiv = unique_ptr<int[]>(new int[nCoarseDof]);

            Lapack<double,ExceptionOnNonzeroError> lapack;
            lapack.GETRF(n, n, data_cLocMass, n, ipiv.get());
            lapack.GETRS(trans, n, nrhs, data_cLocMass, n,
                         ipiv.get(), data_MlP, n);
        }

        if (dof_lin_func->CheckFinite())
        {
            std::cout << "Topology::entity " << entity_type
                      << " entity_id " << entity_id << "\n";
            MatrixTimesMultiVector(M_ii, localProjector, MlP);
            dot(localProjector, MlP, cLocMass_ii);

            cLocMass_ii.PrintMatlab(std::cout);

            PARELAG_TEST_FOR_EXCEPTION(
                true,
                std::runtime_error,
                "CochainProjector::CreateDofFunctional #5");
        }
    }

    dofLinearFunctional_[entity_type]->SetElementalMatrix(
        entity_id,
        std::move(dof_lin_func));
}

void CochainProjector::SetDofFunctional(
    AgglomeratedTopology::Entity entity_type, int entity_id,
    unique_ptr<DenseMatrix> dof_lin_func)
{
    dofLinearFunctional_[entity_type]->SetElementalMatrix(
        entity_id, std::move(dof_lin_func));
}

void CochainProjector::Project(const MultiVector & vFine, MultiVector & vCoarse)
{
    if (Pi_)
        MatrixTimesMultiVector(*Pi_, vFine, vCoarse);
    else
    {
        MultiVector res(vFine.NumberOfVectors(), vFine.Size());
        Project(vFine, vCoarse, res);
    }
}

void CochainProjector::Project(const MultiVector & vFine, MultiVector & vCoarse, MultiVector & res)
{
    PARELAG_TEST_FOR_EXCEPTION(
        !Finalized(),
        std::runtime_error,
        "CochainProjector::Project(): Projector has not been finalized.");

    const int nv = vFine.NumberOfVectors();
    const int codim_max = cDof_->GetMaxCodimensionBaseForDof();

    vCoarse = 0.;
    res = vFine;

    MultiVector vCoarse_d;

    MultiVector res_local, vCoarse_local;
    Array<int> internalAggDofs, internalCDofs;

    for(int codim(codim_max); codim >= 0; --codim)
    {
        for (int ientity(0);
             ientity < cDof_->GetNumberEntities(
                 static_cast<AgglomeratedTopology::Entity>(codim));
             ++ientity)
        {
            dofAgg_->GetViewAgglomerateInternalDofGlobalNumering(
                static_cast<AgglomeratedTopology::Entity>(codim),
                ientity,
                internalAggDofs );
            cDof_->GetInteriorDofs(
                static_cast<AgglomeratedTopology::Entity>(codim),
                ientity,
                internalCDofs );

            res_local.SetSizeAndNumberOfVectors(internalAggDofs.Size(), nv);
            vCoarse_local.SetSizeAndNumberOfVectors(internalCDofs.Size(), nv);

#if 1
            vCoarse.GetSubMultiVector(internalCDofs, vCoarse_local);
            PARELAG_TEST_FOR_EXCEPTION(
                vCoarse_local.Norml2() > 1e-14,
                std::runtime_error,
                "CochainProjector::Project(): "
                "vCoarse_local.Norml2() is too big (>1e-14).");

#endif

            res.GetSubMultiVector(internalAggDofs, res_local);
            Mult(dofLinearFunctional_[codim]->GetElementalMatrix(ientity),
                 res_local,
                 vCoarse_local);

            vCoarse.SetSubMultiVector(internalCDofs, vCoarse_local);
        }

        res = vFine;
        MatrixTimesMultiVector(-1, *P_, vCoarse, res);
    }
}

void CochainProjector::ComputeProjector()
{
    /*
     * ALGORITHM:
     * Pi_{codim_base} = \hat{ Pi_{codim_base} }
     * Pi_{codim - 1} = Pi_{codim} + hat{ Pi_{codim-1} } (I - P Pi_codim )
     *
     * where \hat{ Pi_codim } is assembled from dofLinearFunctional[codim]
     */

    if(Pi_)
        return;

    PARELAG_TEST_FOR_EXCEPTION(
        !Finalized(),
        std::runtime_error,
        "Please finish the setup of the projector "
        "before assembling it as a SparseMatrix");

    PARELAG_TEST_FOR_EXCEPTION(
        P_->Width() != cDof_->GetNDofs(),
        std::runtime_error,
        "P->Width() != cDof_->GetNDofs()" );// very descriptive...

    PARELAG_TEST_FOR_EXCEPTION(
        P_->Size() != dofAgg_->GetDofHandler()->GetNDofs(),
        std::runtime_error,
        "P_->Size() != dofAgg_->GetDofHandler()->GetNDofs()");

//      int nFineDofs = P->Size();
//      int nCoarseDofs = P->Width();
    const int codimBase = cDof_->GetMaxCodimensionBaseForDof();

    auto Pi_this = assembleInternalProjector(codimBase);

    for(int codim(codimBase-1); codim >= 0; --codim)
    {
        auto hatPi = assembleInternalProjector(codim);
        unique_ptr<SparseMatrix> PPi_this{Mult(*P_, *Pi_this)};
        unique_ptr<SparseMatrix> hatPiPPi_this{Mult(*hatPi, *PPi_this)};
        Pi_this = Add(1., *Pi_this, 1., *hatPi, -1., *hatPiPPi_this);
    }
    Pi_ = std::move(Pi_this);
}

SparseMatrix & CochainProjector::GetProjectorMatrix()
{
    PARELAG_TEST_FOR_EXCEPTION(
        !Pi_,
        std::runtime_error,
        "CochainProjector::GetProjectorMatrix(): "
        "Need to Call Compute Projector first.");

    return *Pi_;
}

const SparseMatrix & CochainProjector::GetProjectorMatrix() const
{
    PARELAG_TEST_FOR_EXCEPTION(
        !Pi_,
        std::runtime_error,
        "CochainProjector::GetProjectorMatrix(): "
        "Need to Call Compute Projector first.");

    return *Pi_;
}

unique_ptr<SparseMatrix>
CochainProjector::GetIncompleteProjector()
{
    const int nFineDofs = P_->Size();
    const int nCoarseDofs = P_->Width();
    auto out = make_unique<SparseMatrix>(nCoarseDofs, nFineDofs);

    Array<int> internalAggDofs, internalCDofs;

    const int baseCodim = cDof_->GetMaxCodimensionBaseForDof();

    for(int codim(baseCodim); codim >= 0; --codim)
    {
        ElementalMatricesContainer & data = *(dofLinearFunctional_[codim]);

        for(int ientity(0); ientity < cDof_->GetNumberEntities(static_cast<AgglomeratedTopology::Entity>(codim)); ++ientity)
        {
            dofAgg_->GetViewAgglomerateInternalDofGlobalNumering(
                static_cast<AgglomeratedTopology::Entity>(codim),
                ientity,
                internalAggDofs );
            cDof_->GetInteriorDofs(
                static_cast<AgglomeratedTopology::Entity>(codim),
                ientity,
                internalCDofs );
            out->AddSubMatrix(internalCDofs,internalAggDofs,
                              data.GetElementalMatrix(ientity));
        }
    }
    out->Finalize();
    return out;
}

void CochainProjector::CheckInverseProperty()
{
    PARELAG_TEST_FOR_EXCEPTION(
        !Finalized(),
        std::runtime_error,
        "CochainProjector::Check(): Called Check() before Finalized.");

    const int nFineDofs = P_->Size();
    const int nCoarseDofs = P_->Width();
    constexpr int nv = 5;
    constexpr int seed = 1;

    MultiVector vc(nv, nCoarseDofs), vf(nv, nFineDofs), Piv(nv, nCoarseDofs);

    vc.Randomize(seed);
    vf = 0.0;

    MatrixTimesMultiVector(*P_, vc, vf);

    PARELAG_TEST_FOR_EXCEPTION(
        vc.CheckFinite(),
        std::runtime_error,
        "CochainProjector::Check(): Something bad about vc.");

    PARELAG_TEST_FOR_EXCEPTION(
        vf.CheckFinite(),
        std::runtime_error,
        "CochainProjector::Check(): Something bad about vf.");

    Project(vf, Piv);

    PARELAG_TEST_FOR_EXCEPTION(
        Piv.CheckFinite(),
        std::runtime_error,
        "CochainProjector::Check(): Something bad about Piv.");

    Piv -= vc;

    if (Piv.Normlinf() > 1e-9)
    {
        std::cout << "|| v_c - Pi P v_c ||_2 / || v_c ||_2 = "
                  << Piv.Norml2() / vc.Norml2() << std::endl;
        std::cout << "|| v_c - Pi P v_c ||_inf / || v_c ||_inf = "
                  << Piv.Normlinf() / vc.Normlinf() << std::endl;
    }
}

void CochainProjector::CheckInvariants()
{
    CheckInverseProperty();

    if (!Pi_)
        ComputeProjector();

    unique_ptr<SparseMatrix> PiP{Mult(*Pi_, *P_)};

    bool passed = IsAlmostIdentity(*PiP, 1e-9);
    PARELAG_TEST_FOR_EXCEPTION(
        !passed,
        std::runtime_error,
        "Pi*P is not the identity operator :(" );

    PiP.reset();

    const int nFineDofs = P_->Size();
    const int nCoarseDofs = P_->Width();
    constexpr int nv = 5;
    constexpr int seed = 1;

    MultiVector vf(nv, nFineDofs);
    vf.Randomize(seed);

    MultiVector vc1(nv, nCoarseDofs),
        vc2(nv, nCoarseDofs),
        vdiff(nv, nCoarseDofs);
    Project(vf, vc1);
    MatrixTimesMultiVector(*Pi_, vf, vc2);
    subtract(vc1, vc2, vdiff);
    if(vdiff.Normlinf() > 1e-9)
    {
        std::cout << " MatrixFree Projector and Matrix Projector "
                  << "are not the same." << std::endl;
        std::cout << vdiff.Normlinf() << std::setw(14)
                  << vdiff.Norml1() << std::endl;
    }
}


bool CochainProjector::Finalized()
{
    for (const auto& dof :dofLinearFunctional_)
        if ( !dof->Finalized() )
            return false;

    return P_->Finalized() && cDof_->Finalized();
}

unique_ptr<SparseMatrix>
CochainProjector::assembleInternalProjector(int codim)
{
    const int nFineDofs = P_->Size();
    const int nCoarseDofs = P_->Width();
    auto out = make_unique<SparseMatrix>(nCoarseDofs, nFineDofs);
    Array<int> internalAggDofs, internalCDofs;
    ElementalMatricesContainer & data = *(dofLinearFunctional_[codim]);

    for(int ientity(0); ientity < cDof_->GetNumberEntities(static_cast<AgglomeratedTopology::Entity>(codim)); ++ientity)
    {
        dofAgg_->GetViewAgglomerateInternalDofGlobalNumering(
            static_cast<AgglomeratedTopology::Entity>(codim),
            ientity,
            internalAggDofs );
        cDof_->GetInteriorDofs(
            static_cast<AgglomeratedTopology::Entity>(codim),
            ientity,
            internalCDofs );
        out->AddSubMatrix(internalCDofs,
                          internalAggDofs,
                          data.GetElementalMatrix(ientity));
    }
    out->Finalize();
    return out;
}
}//namespace parelag
