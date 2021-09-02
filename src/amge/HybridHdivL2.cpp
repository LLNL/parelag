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

#include "HybridHdivL2.hpp"

#include "linalg/dense/ParELAG_LDLCalculator.hpp"
#include "linalg/dense/ParELAG_QDQCalculator.hpp"
#include "linalg/utilities/ParELAG_MatrixUtils.hpp"
#include "linalg/utilities/ParELAG_SubMatrixExtraction.hpp"
#include "utilities/MemoryUtils.hpp"

namespace parelag
{
using namespace mfem;
using std::unique_ptr;

HybridHdivL2::HybridHdivL2(const std::shared_ptr<DeRhamSequence>& sequence,
                           bool IsSameOrient_, double W_weight_,
                           Array<int>& ess_HdivDofs_,
                           std::shared_ptr<mfem::Vector> elemMatrixScaling_):
    IsSameOrient(IsSameOrient_),
    W_weight(W_weight_),
    HybridSystem(nullptr),
    topo(sequence->GetTopology()),
    nDimension(topo->Dimensions()),
    facet_elem(nullptr),
    dofHdiv(sequence->GetDofHandler(nDimension-1)),
    dofL2(sequence->GetDofHandler(nDimension)),
    dofMultiplier(nullptr),
    M_el(sequence->GetM(AgglomeratedTopology::ELEMENT, nDimension-1)),
    W(nullptr),
    B(nullptr),
    elemMatrixScaling(elemMatrixScaling_),
    Multiplier_Hdiv_(nullptr),
    Hybrid_el(0),
    AinvCT(0),
    Ainv_f(0),
    Ainv(0),
    A_el(0),
    L2_const_rep_(sequence->GetL2ConstRepresentation()),
    CCT_inv_CBT1(0)
{
    // Mass matrix for L2 space
    W = sequence->ComputeMassOperator(nDimension);

    //This is Mass*Divergence operator
    B.reset(Mult(*(sequence->ComputeMassOperator(nDimension)),
                 *(sequence->GetDerivativeOperator(nDimension-1))));

    // copy essential BC for H(div) space
    ess_HdivDofs.SetSize(ess_HdivDofs_.Size());
    ess_HdivDofs_.Copy(ess_HdivDofs);

    // Construct the hybridized system and the transform matrices from the
    // given saddle point problem
    AssembleHybridSystem();
}


HybridHdivL2::~HybridHdivL2()
{
}

void HybridHdivL2::AssembleHybridSystem()
{
    // Given M_el, W, and B, where M_el is the the DG-like weighted mass matrix
    // for the Hdiv space, the following hybrid system is formed
    // Hybrid = [C 0][M B^T;B W]^-1[C 0]^T
    // The C matrix is the constraint matrix for enforcing the continuity of
    // the "broken" Hdiv space as well as the boundary conditions. This is
    // created inside this function
    // Each constraint in turn creates a dual variable (Lagrange multiplier) dof.
    // The construction is done locally in each element

    // Extracting the relation tables needed from DofHandlers
    const SparseMatrix & elem_HdivDof
        = dofHdiv->GetEntityDofTable(AgglomeratedTopology::ELEMENT);
    const SparseMatrix & facet_HdivDof
        = dofHdiv->GetEntityDofTable(AgglomeratedTopology::FACET);
    const SparseMatrix & elem_rHdivDof
        = dofHdiv->GetEntityRDofTable(AgglomeratedTopology::ELEMENT);
    const SparseMatrix & rHdivDof_HdivDof
        = dofHdiv->GetrDofDofTable(AgglomeratedTopology::ELEMENT);
    const SparseMatrix & elem_L2Dof
        = dofL2->GetEntityDofTable(AgglomeratedTopology::ELEMENT);
    const SparseMatrix & B_const = *B;

    const int nElem = elem_L2Dof.Size();
    const int nFacet = facet_HdivDof.Size();
    const int nHdivDof = dofHdiv->GetNDofs();

    // Set the size of the Hybrid_el, AinvCT, Ainv_f, these are all local
    // matrices and vector for each element
    Hybrid_el.resize(nElem);
    AinvCT.resize(nElem);
    Ainv_f.resize(nElem);
    Ainv.resize(nElem);
    A_el.resize(nElem);

    // The vector of element matrix scaling should have size nElem
    if (elemMatrixScaling)
    {
        PARELAG_ASSERT(elemMatrixScaling->Size() == nElem);
    }

    // L2DofLoc, rHdivDofLoc, HdivDofLoc, and MultiplierDofLoc are array of
    // golbal numbering of the local Dofs (or rdof) in each element
    // MultiplierToGlobalHdivMap is an array of size number of local Multiplier
    // dofs. Every multiplier dof comes from an interior facet Hdiv dof. So
    // here, the i-th entry of MultiplierToGlobalHdivMap contains the global
    // numbering of the Hdiv dof associated with the i-th local multiplier dof.
    // Orientation is an array of size number of local Multiplier dofs. Every
    // multiplier dof is supported in an interior facet. The i-th entry of
    // Orientation contains the orientation (either 1 or -1) of the facet
    // (where the i-th local multiplier dof is supported in) relative to the
    // element
    Array<int> L2DofLoc, rHdivDofLoc, HdivDofLoc;
    Array<int> MultiplierDofLoc, MultiplierToGlobalHdivMap, Orientation;

    // Auxiliary array for submatrix extraction of Q
    Array<int> WMarker( W->Width() );
    WMarker = -1;

    unique_ptr<SparseMatrix> Mloc;
    unique_ptr<SparseMatrix> Wloc;
    DenseMatrix ClocT;
    unique_ptr<DenseMatrix> tmpAinvCT;
    unique_ptr<DenseMatrix> tmpHybrid_el;
    unique_ptr<Vector> Ainv_f_loc;
    unique_ptr<DenseInverseCalculator> solver;
    unique_ptr<DenseMatrix> tmpAloc;

    SharingMap & HdivTrueHdiv = dofHdiv->GetDofTrueDof();
    SharingMap & FacetTrueFacet = topo->EntityTrueEntity(
                AgglomeratedTopology::FACET);
    TopologyTable & elem_facet = topo->GetB(AgglomeratedTopology::ELEMENT);
    facet_elem.reset(Transpose(elem_facet));
    int * i_elem_facet = elem_facet.GetI();
    int * j_elem_facet = elem_facet.GetJ();
    const int * i_facet_HdivDof = facet_HdivDof.GetI();
    const int * j_facet_HdivDof = facet_HdivDof.GetJ();

    const int * i_elem_L2Dof = elem_L2Dof.GetI();
    const int * j_elem_L2Dof = elem_L2Dof.GetJ();
    const int * i_elem_HdivDof = elem_HdivDof.GetI();
    const int * j_elem_HdivDof = elem_HdivDof.GetJ();
    const int * i_elem_rHdivDof = elem_rHdivDof.GetI();
    const int * j_elem_rHdivDof = elem_rHdivDof.GetJ();

    // facet_FullHdivDof is almost the same as facet_HdivDof table, except that
    // the size of facet_FullHdivDof is always equal to number of HdivDof, but
    // the size of facet_HdivDof may equal to number of HdivDof associated with
    // the facets only (for coarse levels)
    int * i_tmp = new int[nFacet+1];
    int * j_tmp = new int[facet_HdivDof.NumNonZeroElems()];
    double * data_tmp = new double[facet_HdivDof.NumNonZeroElems()];
    std::copy(facet_HdivDof.GetI(),facet_HdivDof.GetI()+nFacet+1, i_tmp);
    std::copy(facet_HdivDof.GetJ(),facet_HdivDof.GetJ()+
              facet_HdivDof.NumNonZeroElems(), j_tmp);
    std::copy(facet_HdivDof.GetData(),facet_HdivDof.GetData()+
              facet_HdivDof.NumNonZeroElems(), data_tmp);
    auto facet_FullHdivDof = make_unique<SparseMatrix>(
        i_tmp,j_tmp,data_tmp,nFacet,nHdivDof );

    SparseMatrix& facet_bdrAttribute = topo->FacetBdrAttribute();
    auto FullHdivDof_facet = ToUnique(Transpose(*facet_FullHdivDof));
    auto FullHdivDof_bdrAttribute = ToUnique(Mult(*FullHdivDof_facet,facet_bdrAttribute));

    // Constructing the relation table (in SparseMatrix format) between Hdiv
    // dof and multiplier dof. For every Hdiv dof, if it is associated with an
    // facet then a Lagrange multiplier dof is created in connection with the
    // Hdiv dof
    unique_ptr<SparseMatrix> HdivDof_elem{Transpose(elem_HdivDof)};
    int * i_HdivDof_Multiplier = new int[nHdivDof+1];
    i_HdivDof_Multiplier[0] = 0;
    for (int i = 0; i < nHdivDof; ++i)
    {
        i_HdivDof_Multiplier[i+1] = i_HdivDof_Multiplier[i];
        if ( (HdivDof_elem->RowSize(i) == 2) | HdivTrueHdiv.IsShared(i) |
             FullHdivDof_bdrAttribute->RowSize(i))
            i_HdivDof_Multiplier[i+1] ++;
    }
    HdivDof_elem.reset();

    int nnz_HdivDof_Multiplier = i_HdivDof_Multiplier[nHdivDof];
    int * j_HdivDof_Multiplier = new int[nnz_HdivDof_Multiplier];
    for (int i = 0; i < nnz_HdivDof_Multiplier; ++i)
        j_HdivDof_Multiplier[i] = i;
    double * data_HdivDof_Multiplier = new double[nnz_HdivDof_Multiplier];
    std::fill(data_HdivDof_Multiplier, data_HdivDof_Multiplier +
              nnz_HdivDof_Multiplier, 1.);
    auto HdivDof_Multiplier = make_unique<SparseMatrix>(
        i_HdivDof_Multiplier,
        j_HdivDof_Multiplier,
        data_HdivDof_Multiplier,
        nHdivDof,
        nnz_HdivDof_Multiplier );
    Multiplier_Hdiv_ = ToUnique(Transpose(*HdivDof_Multiplier));

    // Construct facet_MultiplierDof table from facet_HdivDof table and
    // HdivDof_MultiplierDof table. The elem_Multiplier_tmp table is a
    // temperary elem to multiplier dof relation table, we use it to count the
    // number of local dofs, the actual local to global numbering (the "J"
    // array) is constructed during the computation of the element matrices
    std::vector<std::unique_ptr<const SparseMatrix>> Entity_Multiplier(4);
    Entity_Multiplier[AgglomeratedTopology::FACET] = ToUnique(
        Mult(*facet_FullHdivDof, *HdivDof_Multiplier));
    unique_ptr<SparseMatrix> elem_Multiplier_temp{
        Mult(elem_facet,*(Entity_Multiplier[AgglomeratedTopology::FACET])) };

    facet_FullHdivDof.reset();

    const int * i_facet_Multiplier = Entity_Multiplier[AgglomeratedTopology::FACET]->
            GetI();
    int * i_elem_Multiplier = new int[nElem+1];
    std::copy(elem_Multiplier_temp->GetI(),
              elem_Multiplier_temp->GetI()+nElem+1, i_elem_Multiplier);
    int * j_elem_Multiplier = new int[elem_Multiplier_temp->NumNonZeroElems()];

    HybridSystem = make_unique<SparseMatrix>(nnz_HdivDof_Multiplier);

    Array<int> HdivGlobalToLocalMap(elem_HdivDof.Width());
    HdivGlobalToLocalMap = -1;

    // Determine whether to construct rescaling vector (CC^T)^{-1}CB^T1
    // (rescaling works in fine level or if number of HdivDof on each facet = 1)
    const bool do_rescale = !IsSameOrient || (nFacet == facet_HdivDof.NumCols());
    mfem::Vector CCT_diag(do_rescale ? HybridSystem->NumRows() : 0);
    mfem::Vector CBT1(CCT_diag.Size());
    CCT_diag = 0.0;
    CBT1 = 0.0;

    for (int elem = 0; elem < nElem; ++elem)
    {
        // Extracting the size and global numbering of local dof
        int nL2DofLoc = elem_L2Dof.RowSize(elem);
        int nHdivDofLoc = elem_HdivDof.RowSize(elem);

        L2DofLoc.MakeRef(const_cast<int *>(j_elem_L2Dof)+i_elem_L2Dof[elem],
                         nL2DofLoc);
        rHdivDofLoc.MakeRef(const_cast<int *>(j_elem_rHdivDof)+i_elem_rHdivDof[elem],
                            nHdivDofLoc);
        HdivDofLoc.MakeRef(const_cast<int *>(j_elem_HdivDof)+i_elem_HdivDof[elem],
                           nHdivDofLoc);

        // Build the Hdiv dof global to local map which will be used
        // later for mapping local multiplier dof to local Hdiv dof
        for (int i=0; i < nHdivDofLoc; ++i)
            HdivGlobalToLocalMap[HdivDofLoc[i]]=i;

        // Extracting local (weighted) mass matrix Mloc for the Hdiv space
        Mloc = ExtractSubMatrix(*M_el,
                                i_elem_rHdivDof[elem],
                                i_elem_rHdivDof[elem+1],
                                i_elem_rHdivDof[elem],
                                i_elem_rHdivDof[elem+1]);

        // Scale the local mass matrix if elemMatrixScaling not a nullptr
        if (elemMatrixScaling)
            (*Mloc) *= elemMatrixScaling->Elem(elem);

        // Constructing the local saddle point system [M B^T; B
        // 0]. Notice that here the local matrices are based on
        // repeated Hdiv dof, so the entries of the local matrix B_el
        // need to flip the sign (by multiplying rHdivDof_HdivDof) if
        // the corresponding Hdiv dof and repeated Hdiv dof have
        // opposite orientation. The essential boundary condition (for
        // the Hdiv space) is eliminated during the construction of
        // Aloc
        tmpAloc = make_unique<DenseMatrix>(nHdivDofLoc+nL2DofLoc, nHdivDofLoc+nL2DofLoc);
        (*tmpAloc) = 0.;
        Full(*Mloc, *tmpAloc, 0, 0);

        for (int j=0; j < nHdivDofLoc; ++j)
        	for (int i=0; i < nL2DofLoc; ++i)
        	{
        		(*tmpAloc)(nHdivDofLoc+i,j) = B_const(L2DofLoc[i],HdivDofLoc[j])
                            		* rHdivDof_HdivDof(rHdivDofLoc[j],HdivDofLoc[j]);
        		(*tmpAloc)(j,nHdivDofLoc+i) = (*tmpAloc)(nHdivDofLoc+i,j);
        	}

        if (W_weight != 0)
        {
            Wloc = ExtractRowAndColumns(*W, L2DofLoc, L2DofLoc, WMarker);
            (*Wloc) *= -1.*W_weight;
            Full(*Wloc, *tmpAloc, nHdivDofLoc, nHdivDofLoc);
        }

        // Extracting local multiplier dof. The extraction loops over every
        // facet and then loops over every Hdiv dof on the facet. This is
        // because the orientation of the facet will be used to construct the
        // local matrix Cloc (see below). MultiplierToGlobalHdivMap maps local
        // multiplier dof to the associated global Hdiv dof. Combined with
        // HdivGlobalToLocalMap, these two maps connect local multiplier dofs
        // to the corresponding local Hdiv dofs
        int nMultiplierDofLoc = i_elem_Multiplier[elem+1] -
                i_elem_Multiplier[elem];
        MultiplierToGlobalHdivMap.SetSize(nMultiplierDofLoc);
        MultiplierToGlobalHdivMap = 0;
        MultiplierDofLoc.SetSize(nMultiplierDofLoc);
        MultiplierDofLoc = 0;
        Orientation.SetSize(nMultiplierDofLoc);
        Orientation = 0;
        int counter(0), nDofLoc;
        for (int it = i_elem_facet[elem]; it != i_elem_facet[elem+1]; ++it)
        {
            int facet = j_elem_facet[it];
            if ( ( facet_elem->RowSize(facet) ==2 ) |
            	 FacetTrueFacet.IsShared(facet) |
                 facet_bdrAttribute.RowSize(facet) )
            {
                nDofLoc = i_facet_Multiplier[facet+1] -
                        i_facet_Multiplier[facet];
#ifdef ELAG_DEBUG
                elag_assert(nDofLoc == i_facet_HdivDof[facet+1] -
                            i_facet_HdivDof[facet]);
#endif
                for (int i = 0; i < nDofLoc; ++i)
                {
                    Orientation[counter] = elem_facet(elem,facet);
                    MultiplierDofLoc[counter] = i_HdivDof_Multiplier[
                            j_facet_HdivDof[i_facet_HdivDof[facet]+i] ];
                    j_elem_Multiplier[i_elem_Multiplier[elem]+counter] =
                    		MultiplierDofLoc[counter];
                    MultiplierToGlobalHdivMap[counter++] =
                    		j_facet_HdivDof[i_facet_HdivDof[facet]+i];
                }
            }
        }
#ifdef ELAG_DEBUG
        elag_assert(counter == nMultiplierDofLoc);
#endif

        /* Cloc is a constraint matrix enforcing the normal continuity of the
           solution in the broken Hdiv space
           XXX: The construction now requires an extra information about the
           orientation of the Hdiv repeated dof. If they have the same
           orientation, then Cloc forces the difference of them to be 0. If
           they have the opposite orientation, then Cloc forces the sum of them
           to be 0. This extra information may be not needed if the SharingMap
           of the Multiplier carries the orientation information in the
           SharingMap of the Hdiv dof. Note that for upscaled systems, the
           orientation of Hdiv repeated dof are always the same, but for the
           fine level problem (come from finite element), the situation is
           different. In 2D, the orientation information are carried in local
           relation tables (rdof_dof). In 3D, the orientation information are
           carried in the SharingMap of the Hdiv dof. Therefore, before we
           assemble the "True" global system, we cannot correct the
           orientation. But we use the SharingMap of the mutliplier dof (which
           do not inherit the orientation of Hdiv dof in the current
           construction) to assemble the global system, so the orientation
           cannot be corrected if we do not provide the extra information about
           the orientation

           XXX: The code right now cannot correctly compute the case when
           upscalingorder = -1, i.e., when there are only PV vectors in the
           coarse space. This is because in this case the orientation of the
           coarse Hdiv repeated dofs may be in opposite direction after
           splitting IF THE HDIV REPEATED DOFS ARE SHARED. More precisely, in
           3D, they are always in opposite direction, while in 2D, some of them
           are in opposite direction, but some are in the same direction. This
           problem seems can only be solved if we can do something to the
           SharingMap or we change the way we define the repeated dofs
        */
        ClocT.SetSize(nHdivDofLoc+nL2DofLoc,nMultiplierDofLoc);
        ClocT = 0.;

        int LocalHdiv, GlobalHdiv, IsShared;
        if (IsSameOrient)
            for (int i = 0; i < nMultiplierDofLoc; ++i)
            {
            	GlobalHdiv = MultiplierToGlobalHdivMap[i];
            	LocalHdiv = HdivGlobalToLocalMap[GlobalHdiv];
            	IsShared = HdivTrueHdiv.IsShared(GlobalHdiv);
            	if (FullHdivDof_bdrAttribute->RowSize(GlobalHdiv))
            		ClocT(LocalHdiv,i) = rHdivDof_HdivDof(
            				rHdivDofLoc[LocalHdiv],HdivDofLoc[LocalHdiv]);
            	else
            		ClocT(LocalHdiv,i) = (IsShared?IsShared:Orientation[i]);
            }
        else
            for (int i=0; i < nMultiplierDofLoc; ++i)
            {
                GlobalHdiv = MultiplierToGlobalHdivMap[i];
                LocalHdiv = HdivGlobalToLocalMap[GlobalHdiv];
                if (FullHdivDof_bdrAttribute->RowSize(GlobalHdiv))
                	ClocT(LocalHdiv,i) = rHdivDof_HdivDof(
                			rHdivDofLoc[LocalHdiv],HdivDofLoc[LocalHdiv]);
                else
                	ClocT(LocalHdiv,i) = 1.;
            }

        // Solve local problem to form element matrix [C 0][M B^T;B
        // 0]^-1[C 0]^T (it is a dense matrix which tmpHybrid_el
        // points to)
        // Note that here [C 0]^T is the matrix Cloc, the latter will
        // be transposed to form [C 0] later
        solver = make_unique<LDLCalculator>();
        solver->Compute(*tmpAloc);

        tmpAinvCT = make_unique<DenseMatrix>(nHdivDofLoc+nL2DofLoc,
                                             nMultiplierDofLoc);
        solver->Mult(ClocT, *tmpAinvCT);
        ClocT.Transpose();
        tmpHybrid_el = make_unique<DenseMatrix>(nMultiplierDofLoc,
                                                nMultiplierDofLoc);
        Mult(ClocT,*tmpAinvCT,*tmpHybrid_el);
        tmpHybrid_el->Symmetrize();

        // Add contribution of the element matrix to the golbal system
        // (not the "True global system")
        HybridSystem->AddSubMatrix(MultiplierDofLoc,
                                   MultiplierDofLoc,
                                   *tmpHybrid_el,
                                   1);

        // Save CCT and CBT1
        if (do_rescale)
        {
            mfem::DenseMatrix CCT(nMultiplierDofLoc);
            mfem::MultAAt(ClocT, CCT);
            mfem::Vector CCT_diag_local;
            CCT.GetDiag(CCT_diag_local);

            mfem::Vector const_one;
            L2_const_rep_.GetSubVector(L2DofLoc, const_one);

            mfem::Vector zero_one(nHdivDofLoc+nL2DofLoc);
            for (int i = 0; i < nHdivDofLoc; ++i)
            {
                zero_one[i] = 0.0;
            }
            for (int i = 0; i < nL2DofLoc; ++i)
            {
                zero_one[nHdivDofLoc+i] = const_one[i];
            }

            mfem::Vector BTone(nHdivDofLoc+nL2DofLoc);
            tmpAloc->Mult(zero_one, BTone);

            mfem::Vector CBT1_local(nMultiplierDofLoc);
            ClocT.Mult(BTone, CBT1_local);

            for (int i = 0; i < nMultiplierDofLoc; ++i)
            {
                CCT_diag[MultiplierDofLoc[i]] += CCT_diag_local[i];
                CBT1[MultiplierDofLoc[i]] += CBT1_local[i];
            }
        }

        // Save the factorization of [M B^T;B 0] for solution recovery purpose
        A_el[elem] = std::move(tmpAloc);
        Ainv[elem] = std::move(solver);

        // Save the element matrix [M B^T;B 0]^-1[C 0]^T for solution
        // recovery purpose
        AinvCT[elem] = std::move(tmpAinvCT);

        // Save the element matrix [C 0][M B^T;B 0]^-1[C 0]^T (this is needed
        // only if one wants to construct H1 spectral AMGe preconditioner)
        Hybrid_el[elem] = std::move(tmpHybrid_el);
    }

    HybridSystem->Finalize();

    // Construct elem_Multiplier table which matchs the local numbering of the
    // local matrices Hybrid_el
    double * data_elem_Multiplier = new double[elem_Multiplier_temp->
											   NumNonZeroElems()];
    std::fill(data_elem_Multiplier,
    		data_elem_Multiplier+elem_Multiplier_temp->NumNonZeroElems(), 1.);
    Entity_Multiplier[AgglomeratedTopology::ELEMENT]
        = make_unique<SparseMatrix>(
            i_elem_Multiplier, j_elem_Multiplier,data_elem_Multiplier, nElem,
            elem_Multiplier_temp->Width());

    // Construct the DofHandler for the Lagrange multiplier from scratch since
    // we have the entity_dof tables already The multiplier dof SharingMap is
    // built from the SharingMap of the Hdiv dof and the HdivDof_MultiplierDof
    // relation table Notice that in general the Hdiv dof SharingMap contains
    // orientation information of the Hdiv dof, but the multiplier dof
    // SharingMap constructed this way does not carry orientation
    dofMultiplier = make_unique<DofHandlerSCRATCH>(HdivTrueHdiv.GetComm(),
        AgglomeratedTopology::FACET,nDimension,std::move(Entity_Multiplier));
    SharingMap & MultiplierTrueMultiplier(dofMultiplier->GetDofTrueDof());
    MultiplierTrueMultiplier.SetUp(HdivTrueHdiv,*HdivDof_Multiplier);

    // Mark the multiplier dof with essential BC
    ess_MultiplierDofs.SetSize(nnz_HdivDof_Multiplier);
    ess_MultiplierDofs = 0;//ess_HdivDofs.Print();
    for (int i = 0; i < nHdivDof; i++)
    	// natural BC for H(div) dof => essential BC for multiplier dof
        if (FullHdivDof_bdrAttribute->RowSize(i) & !ess_HdivDofs[i])
    	{
    		int mult_dof = j_HdivDof_Multiplier[i_HdivDof_Multiplier[i]];
    		ess_MultiplierDofs[mult_dof] = 1;
    	}

    // Assemble global rescaling vector (CC^T)^{-1}CB^T 1
    if (do_rescale)
    {
        mfem::Vector CCT_diag_global(MultiplierTrueMultiplier.GetTrueLocalSize());
        MultiplierTrueMultiplier.Assemble(CCT_diag, CCT_diag_global);

        mfem::Vector CBT1_global(MultiplierTrueMultiplier.GetTrueLocalSize());
        MultiplierTrueMultiplier.Assemble(CBT1, CBT1_global);

        CCT_inv_CBT1.SetSize(MultiplierTrueMultiplier.GetTrueLocalSize());
        for (int i = 0; i < CCT_inv_CBT1.Size(); ++i)
        {
            CCT_inv_CBT1[i] = CBT1_global[i] / CCT_diag_global[i];
        }
    }
}

// TODO: impose nonzero boundary condition for u.n
void HybridHdivL2::RHSTransform(const BlockVector& OriginalRHS,
		Vector& HybridRHS, Vector& essentialData)
{
    // The notation here are the same as AssembleMuHybridSystem()
    const SparseMatrix & elem_HdivDof =
    		dofHdiv->GetEntityDofTable(AgglomeratedTopology::ELEMENT);
    const SparseMatrix & elem_rHdivDof =
    		dofHdiv->GetEntityRDofTable(AgglomeratedTopology::ELEMENT);
    const SparseMatrix & rHdivDof_HdivDof =
    		dofHdiv->GetrDofDofTable(AgglomeratedTopology::ELEMENT);
    const SparseMatrix & elem_L2Dof =
    		dofL2->GetEntityDofTable(AgglomeratedTopology::ELEMENT);
    const SparseMatrix & elem_Multiplier =
    		dofMultiplier->GetEntityDofTable(AgglomeratedTopology::ELEMENT);

    const int nElem = elem_L2Dof.Size();
    const int nHdivDof = dofHdiv->GetNDofs();
    const int nMultiplierDof = dofMultiplier->GetNDofs();

	HybridRHS.SetSize(nMultiplierDof);
	HybridRHS = 0.;

    Array<int> L2DofLoc, HdivDofLoc, rHdivDofLoc, MultiplierDofLoc;

    int * j_Multiplier_Hdiv = Multiplier_Hdiv_->GetJ();
    Array<int> HdivGlobalToLocalMap(elem_HdivDof.Width());
    HdivGlobalToLocalMap = -1;
	for (int elem = 0; elem < nElem; ++elem)
	{
		// Extracting the size and global numbering of local dof
		int nL2DofLoc = elem_L2Dof.RowSize(elem);
		int nHdivDofLoc = elem_HdivDof.RowSize(elem);
		int nMultiplierDofLoc = elem_Multiplier.RowSize(elem);
		L2DofLoc.MakeRef(const_cast<int *>(elem_L2Dof.GetJ())+elem_L2Dof.GetI()[elem],
				nL2DofLoc);
		HdivDofLoc.MakeRef(const_cast<int *>(elem_HdivDof.GetJ())+elem_HdivDof.GetI()[elem],
				nHdivDofLoc);
		rHdivDofLoc.MakeRef(const_cast<int *>(elem_rHdivDof.GetJ())+elem_rHdivDof.GetI()[elem],
				nHdivDofLoc);
		MultiplierDofLoc.MakeRef(
				const_cast<int *>(elem_Multiplier.GetJ()) +elem_Multiplier.GetI()[elem],
				nMultiplierDofLoc);

        // Build the Hdiv dof global to local map which will be used
        // later for mapping local multiplier dof to local Hdiv dof
        for (int i=0; i < nHdivDofLoc; ++i)
            HdivGlobalToLocalMap[HdivDofLoc[i]]=i;

		// Compute local contribution to the right hand side of the hybrid
		// system. The orientation and essential boundary condition of the Hdiv
		// repeated dof  accounted for when forming the right hand side
		Vector f_loc(nHdivDofLoc+nL2DofLoc);
		f_loc = 0.;
		// In the new hybridization method this is always zero
//		for (int i = 0; i < nHdivDofLoc; ++i)
//				f_loc(i) = OriginalRHS(HdivDofLoc[i]) *
//				              rHdivDof_HdivDof(rHdivDofLoc[i],HdivDofLoc[i]);
		for (int i = 0; i < nL2DofLoc; ++i)
			f_loc(nHdivDofLoc+i) = OriginalRHS(nHdivDof+L2DofLoc[i]);

		Vector CAinv_f_loc(nMultiplierDofLoc);
		CAinv_f_loc = 0.;
		AinvCT[elem]->MultTranspose(f_loc,CAinv_f_loc);

		for (int i = 0; i < nMultiplierDofLoc; ++i)
			HybridRHS(MultiplierDofLoc[i]) += CAinv_f_loc(i);

		// Save the element rhs [M B^T;B 0]^-1[f;g] to recover solution later
		unique_ptr<Vector> Ainv_f_loc =
				make_unique<Vector>(nHdivDofLoc+nL2DofLoc);
		(*Ainv_f_loc) = 0.;
		Ainv[elem]->Mult(f_loc,*Ainv_f_loc);
		Ainv_f[elem] = std::move(Ainv_f_loc);

		int LocalHdiv, GlobalHdiv;
		for (int i = 0; i < nMultiplierDofLoc; ++i)
		{
			GlobalHdiv = j_Multiplier_Hdiv[MultiplierDofLoc[i]];
			LocalHdiv = HdivGlobalToLocalMap[GlobalHdiv];
			essentialData(MultiplierDofLoc[i]) = -1 * OriginalRHS(GlobalHdiv) *
                    rHdivDof_HdivDof(rHdivDofLoc[LocalHdiv],
                                     HdivDofLoc[LocalHdiv]);
		}

	}
}

void HybridHdivL2::RecoverOriginalSolution(const Vector& HybridSol,
		BlockVector& RecoveredSol)
{
    // After solving the hybridized system, we will have the solution for the
	// Lagrange multiplier mu. The Hdiv and L2 solution (solution of the
    // original problem) can then be recovered from mu, i.e.,
    // [u;p] = [f;g] - [M B^T;B 0]^-1[C 0]^T * mu
    // This procedure is done locally in each element

    // The notation here are the same as AssembleMuHybridSystem()
    const SparseMatrix & elem_HdivDof =
    		dofHdiv->GetEntityDofTable(AgglomeratedTopology::ELEMENT);
    const SparseMatrix & elem_rHdivDof =
    		dofHdiv->GetEntityRDofTable(AgglomeratedTopology::ELEMENT);
    const SparseMatrix & rHdivDof_HdivDof =
    		dofHdiv->GetrDofDofTable(AgglomeratedTopology::ELEMENT);
    const SparseMatrix & elem_L2Dof =
    		dofL2->GetEntityDofTable(AgglomeratedTopology::ELEMENT);
    const SparseMatrix & elem_Multiplier =
    		dofMultiplier->GetEntityDofTable(AgglomeratedTopology::ELEMENT);

    const int nElem = elem_L2Dof.Size();
    const int nHdivDof = rHdivDof_HdivDof.Width();

    RecoveredSol = 0.;

    Array<int> L2DofLoc, HdivDofLoc, rHdivDofLoc, MultiplierDofLoc;

    for (int elem = 0; elem < nElem; ++elem)
    {
        // Extract the global numbering of different variables
        int nL2DofLoc = elem_L2Dof.RowSize(elem);
        int nHdivDofLoc = elem_HdivDof.RowSize(elem);
        int nMultiplierDofLoc = elem_Multiplier.RowSize(elem);
        L2DofLoc.MakeRef(const_cast<int *>(elem_L2Dof.GetJ())+elem_L2Dof.GetI()[elem],
                         nL2DofLoc);
        HdivDofLoc.MakeRef(const_cast<int *>(elem_HdivDof.GetJ())+elem_HdivDof.GetI()[elem],
                           nHdivDofLoc);
        rHdivDofLoc.MakeRef(const_cast<int *>(elem_rHdivDof.GetJ())+elem_rHdivDof.GetI()[elem],
                            nHdivDofLoc);
        MultiplierDofLoc.MakeRef(
        		const_cast<int *>(elem_Multiplier.GetJ()) +elem_Multiplier.GetI()[elem],
				nMultiplierDofLoc);

        // Initialize a vector which will store the local contribution of Hdiv
        // and L2 space
        Vector VelocityPressureLoc(nHdivDofLoc+nL2DofLoc);
        VelocityPressureLoc = 0.;

        // This check is just for the case when there is only one element for
        // the global problem, then there will be no Lagrange multipliers
        if (nMultiplierDofLoc > 0)
        {
            // Extract the local portion of the Lagrange multiplier solution
            Vector MuLoc(nMultiplierDofLoc);
            HybridSol.GetSubVector(MultiplierDofLoc,MuLoc);

            // Compute [M B^T;B 0]^-1[C 0]^T*mu
            AinvCT[elem]->Mult(MuLoc,VelocityPressureLoc);
        }

        // Compute [M B^T;B 0]^-1([C 0]^T*mu-[f;g])
        VelocityPressureLoc -= *(Ainv_f[elem]);

        // Here we set RecoveredSol = -VelocityPressureLoc
        //                          = [M B^T;B 0]^-1([f;g]-[C 0]^T*mu)
        for (int i = 0; i < nL2DofLoc; ++i)
            RecoveredSol(nHdivDof+L2DofLoc[i]) =
            		-VelocityPressureLoc(nHdivDofLoc+i);

        // For the Hdiv part we need to take care of the orientation of
        // repeated dof
        for (int i = 0; i < nHdivDofLoc; ++i)
            RecoveredSol(HdivDofLoc[i]) = -VelocityPressureLoc(i) *
			    (rHdivDof_HdivDof(rHdivDofLoc[i],HdivDofLoc[i]));
    }
}

}//namespace parelag
