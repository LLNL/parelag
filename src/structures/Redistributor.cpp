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

#include <numeric>

#include "Redistributor.hpp"
#include "utilities/MemoryUtils.hpp"
#include "linalg/utilities/ParELAG_MatrixUtils.hpp"


namespace parelag
{
using namespace mfem;
using std::shared_ptr;
using std::make_shared;
using std::unique_ptr;

unique_ptr<ParallelCSRMatrix> Move(matred::ParMatrix& A)
{
   auto out = make_unique<ParallelCSRMatrix>(A);
   A.SetOwnerShip(false);
   out->CopyRowStarts();
   out->CopyColStarts();
   return out;
}

Redistributor::Redistributor(
      const AgglomeratedTopology& topo, const std::vector<int>& elem_redist_procs)
   : redTrueEntity_trueEntity(topo.Codimensions()+1),
     redEntity_trueEntity(topo.Codimensions()+1),
     redTrueDof_TrueDof(topo.Codimensions()+1)
{
   // TODO: entities in codimension > 1 (need to adjust B)
   auto elem_redProc = matred::EntityToProcessor(topo.GetComm(), elem_redist_procs);
   auto redProc_elem = matred::Transpose(elem_redProc);
   auto redElem_elem = matred::BuildRedistributedEntityToTrueEntity(redProc_elem);
   redTrueEntity_trueEntity[0] = Move(redElem_elem);

   auto elem_trueEntity = const_cast<AgglomeratedTopology&>(topo).TrueB(0);
   redEntity_trueEntity[1] = BuildRedEntToTrueEnt(elem_trueEntity);
}

unique_ptr<ParallelCSRMatrix>
Redistributor::BuildRedEntToTrueEnt(const ParallelCSRMatrix& elem_tE) const
{
    auto redElem_tE = Mult(*redTrueEntity_trueEntity[0], elem_tE);
    matred::ParMatrix redElem_trueEntity(*redElem_tE, false);
    auto out = matred::BuildRedistributedEntityToTrueEntity(redElem_trueEntity);
    return Move(out);
}

unique_ptr<ParallelCSRMatrix>
Redistributor::BuildRedEntToRedTrueEnt(const ParallelCSRMatrix& redE_tE) const
{
    matred::ParMatrix redE_tE_ref(redE_tE, false);
    auto tE_redE = matred::Transpose(redE_tE_ref);
    auto redE_tE_redE = matred::Mult(redE_tE_ref, tE_redE);
    auto out = matred::BuildEntityToTrueEntity(redE_tE_redE);
    return Move(out);
}

unique_ptr<ParallelCSRMatrix>
Redistributor::BuildRedTrueEntToTrueEnt(const ParallelCSRMatrix& redE_redTE,
                                        const ParallelCSRMatrix& redE_tE) const
{
    unique_ptr<ParallelCSRMatrix> redTE_redE(redE_redTE.Transpose());
    unique_ptr<ParallelCSRMatrix> out = Mult(*redTE_redE, redE_tE);
    *out = 1.0;
    out->CopyRowStarts();
    out->CopyColStarts();
    return out;
}

std::shared_ptr<AgglomeratedTopology>
Redistributor::Redistribute(const AgglomeratedTopology& topo)
{
   // Redistribute the current topology to another set of processors
   // TODO: currently only elements and faces are redistributed, need to
   // generalize the code to cover edges and vertices as well.
   auto out = make_shared<AgglomeratedTopology>(topo.Comm_, topo.nCodim_);
   out->nDim_ = topo.nDim_;

   // Redistribute core topological data
   const int num_redElem = TrueEntityRedistribution(0).NumRows();
   out->EntityTrueEntity(0).SetUp(num_redElem);

   auto redE_redTE = BuildRedEntToRedTrueEnt(*redEntity_trueEntity[1]);
   redTrueEntity_trueEntity[1] =
         BuildRedTrueEntToTrueEnt(*redE_redTE, *redEntity_trueEntity[1]);
   out->entityTrueEntity[1]->SetUp(std::move(redE_redTE));

   // Redistribute other remaining data
   auto& trueB = const_cast<AgglomeratedTopology&>(topo).TrueB(0);
   unique_ptr<ParallelCSRMatrix> TE_redE(redEntity_trueEntity[1]->Transpose());
   auto redB = RAP(*redTrueEntity_trueEntity[0], trueB, *TE_redE);

   SerialCSRMatrix redB_diag;
   redB->GetDiag(redB_diag);
   SerialCSRMatrix redB_diag_copy(redB_diag);
   out->B_[0] = make_unique<TopologyTable>(redB_diag_copy);

   auto e_tE = topo.EntityTrueEntity(1).get_entity_trueEntity();
   unique_ptr<ParallelCSRMatrix> tE_E(e_tE->Transpose());

   int myid;
   MPI_Comm_rank(topo.Comm_, &myid);

   Array<int> e_starts(3), attr_starts(3), attr_map;
   e_starts[0] = e_tE->RowPart()[0];
   e_starts[1] = e_tE->RowPart()[1];
   e_starts[2] = e_tE->M();
   attr_starts[0] = myid > 0 ? topo.facet_bdrAttribute->NumCols() : 0;
   attr_starts[1] = topo.facet_bdrAttribute->NumCols();
   attr_starts[2] = topo.facet_bdrAttribute->NumCols();

   if (myid > 0)
   {
      attr_map.SetSize(topo.facet_bdrAttribute->NumCols());
      iota(attr_map.GetData(), attr_map.GetData()+attr_starts[2], 0);
   }

   SerialCSRMatrix empty(topo.facet_bdrAttribute->NumRows(), 0);
   empty.Finalize();

   ParallelCSRMatrix e_bdrattr(topo.Comm_, e_tE->M(), attr_starts[2], e_starts,
         attr_starts, myid ? &empty : topo.facet_bdrAttribute.get(),
         myid ? topo.facet_bdrAttribute.get() : &empty, attr_map);

   auto tE_bdrattr = Mult(*tE_E, e_bdrattr);
   auto redE_bdrattr = Mult(*redEntity_trueEntity[1], *tE_bdrattr);

   HYPRE_Int* trash_map;
   mfem::SparseMatrix redE_bdrattr_local_ref;
   if (myid == 0)
   {
      redE_bdrattr->GetDiag(redE_bdrattr_local_ref);
   }
   else
   {
      redE_bdrattr->GetOffd(redE_bdrattr_local_ref, trash_map);
   }
   mfem::SparseMatrix redE_bdrattr_local(redE_bdrattr_local_ref);
   out->facet_bdrAttribute = make_unique<TopologyTable>(redE_bdrattr_local);

   out->element_attribute.SetSize(num_redElem);
   Mult(*redTrueEntity_trueEntity[0], topo.element_attribute, out->element_attribute);

   out->Weights_[0]->SetSize(num_redElem);
   redTrueEntity_trueEntity[0]->Mult(topo.Weight(0), out->Weight(0));

   auto trueFacetWeight = topo.TrueWeight(1);
   out->Weights_[1]->SetSize(out->B_[0]->NumCols());
   redEntity_trueEntity[1]->Mult(*trueFacetWeight, *(out->Weights_[1]));

   return out;
}

std::unique_ptr<DofHandler> Redistributor::Redistribute(
        const DofHandler& dof,
        const std::shared_ptr<AgglomeratedTopology>& redist_topo)
{
    auto& nonconst_dof = const_cast<DofHandler&>(dof);
    auto dof_alg = dynamic_cast<DofHandlerALG*>(&nonconst_dof);
    PARELAG_ASSERT(dof_alg); // only DofHandlerALG can be redistributed currently

    const int max_codim_base = dof.GetMaxCodimensionBaseForDof();
    auto out = make_unique<DofHandlerALG>(max_codim_base, redist_topo);

    auto at_elem = AgglomeratedTopology::ELEMENT;
    auto& elem_dof = const_cast<SerialCSRMatrix&>(dof.GetEntityDofTable(at_elem));
    auto& dof_trueDof = dof.GetDofTrueDof();
    auto elem_trueDof = Assemble(dof.GetEntityTrueEntity(0), elem_dof, dof_trueDof);
    auto redElem_trueDof = Mult(*redTrueEntity_trueEntity[0], *elem_trueDof);

    auto redDof_trueDof = BuildRedEntToTrueEnt(*redElem_trueDof);
    auto redDof_redTrueDof = BuildRedEntToRedTrueEnt(*redDof_trueDof);
    redTrueDof_TrueDof[max_codim_base] = // TODO: prabably need to take case of rdof ordering
          BuildRedTrueEntToTrueEnt(*redDof_redTrueDof, *redDof_trueDof);

    std::unique_ptr<ParallelCSRMatrix> tD_redD(redDof_trueDof->Transpose());
    out->dofTrueDof.SetUp(std::move(redDof_redTrueDof));

    for (int i = 1; i < max_codim_base; ++i)
    {
        auto codim = static_cast<AgglomeratedTopology::Entity>(i);
        auto& ent_dof = const_cast<SerialCSRMatrix&>(dof.GetEntityDofTable(codim));
        auto tE_tD = Assemble(dof.GetEntityTrueEntity(i), ent_dof, dof_trueDof);
        auto redEnt_redDof = RAP(*redEntity_trueEntity[i], *tE_tD, *tD_redD);

        SerialCSRMatrix redEnt_redDof_diag;
        redEnt_redDof->GetDiag(redEnt_redDof_diag);
        out->entity_dof[i].reset(new SerialCSRMatrix(redEnt_redDof_diag));
        out->finalized[i] = true;

//        Mult(*redTE_TE_helper[i], dof_alg->entity_NumberOfInteriorDofsNullSpace[i],
//             out->entity_NumberOfInteriorDofsNullSpace[i]);
//        Mult(*redTE_TE_helper[i], dof_alg->entity_NumberOfInteriorDofsRangeTSpace[i],
//             out->entity_NumberOfInteriorDofsRangeTSpace[i]);
//        Mult(*redTE_TE_helper[i], dof_alg->entity_InteriorDofOffsets[i],
//             out->entity_InteriorDofOffsets[i]);
    }

    return out;
}

void Mult(const ParallelCSRMatrix& A, const mfem::Array<int>& x, mfem::Array<int>& Ax)
{
   PARELAG_ASSERT(A.NumRows() == Ax.Size() && A.NumCols() == x.Size());

   mfem::Vector x_vec(x.Size());
   for (int i = 0; i < x.Size(); ++i)
   {
      x_vec[i] = x[i];
   }

   mfem::Vector Ax_vec(Ax.Size());
   A.Mult(x_vec, Ax_vec);

   for (int i = 0; i < x.Size(); ++i)
   {
      Ax[i] = Ax_vec[i];
   }
}

} // namespace parelag
