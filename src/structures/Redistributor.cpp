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

namespace parelag
{
using namespace mfem;
using std::shared_ptr;
using std::make_shared;
using std::unique_ptr;
using matred::ParMatrix;

Redistributor::Redistributor(
      const AgglomeratedTopology& topo, const std::vector<int>& elem_redist_procs)
   : redTrueEntity_trueEntity(topo.Codimensions()+1),
     redTE_TE_helper(topo.Codimensions()+1),
     redEntity_redTrueEntity(topo.Codimensions()+1),
     redE_redTE_helper(topo.Codimensions()+1)
{
   // TODO: entities in codimension > 1 (need to adjust B)
   auto elem_redProc = matred::EntityToProcessor(topo.GetComm(), elem_redist_procs);
   auto redProc_elem = matred::Transpose(elem_redProc);
   redTrueEntity_trueEntity[0] = matred::BuildRedistributedEntityToTrueEntity(redProc_elem);

   auto& B0 = const_cast<TopologyTable&>(topo.GetB(0));
   auto elem_tEnt = Assemble(topo.EntityTrueEntity(0), B0, topo.EntityTrueEntity(1));

   auto redEnt_trueEnt = BuildRedEntToTrueEnt(*elem_tEnt);
   redEntity_redTrueEntity[1] = BuildNewEntTrueEnt(redEnt_trueEnt);
   redTrueEntity_trueEntity[1] = BuildRedTrueEntTrueEnt(
               redEntity_redTrueEntity[1], redEnt_trueEnt);

   for (int codim = 0; codim < topo.Codimensions()+1; ++ codim)
   {
      redTE_TE_helper[codim].reset(
               new ParallelCSRMatrix(redTrueEntity_trueEntity[codim], false));
      if (codim == 0) { continue; }
      redE_redTE_helper[codim].reset(
               new ParallelCSRMatrix(redEntity_redTrueEntity[codim], false));
   }
}

ParMatrix
Redistributor::BuildRedEntToTrueEnt(const ParallelCSRMatrix& elem_trueEntity) const
{
    matred::ParMatrix elem_trueEnt(elem_trueEntity, false);
    auto redElem_trueEnt = matred::Mult(redTrueEntity_trueEntity[0], elem_trueEnt);
    return matred::BuildRedistributedEntityToTrueEntity(redElem_trueEnt);
}

ParMatrix
Redistributor::BuildNewEntTrueEnt(const ParMatrix& redEntity_trueEntity) const
{
    auto tE_redE = matred::Transpose(redEntity_trueEntity);
    auto redE_tE_redE = matred::Mult(redEntity_trueEntity, tE_redE);
    return matred::BuildEntityToTrueEntity(redE_tE_redE);
}

ParMatrix
Redistributor::BuildRedTrueEntTrueEnt(
        const ParMatrix& redEntity_redTrueEntity,
        const ParMatrix& redEntity_trueEntity) const
{
    auto redTE_redE = matred::Transpose(redEntity_redTrueEntity);
    auto out = matred::Mult(redTE_redE, redEntity_trueEntity);
    out = 1.0;
    return out;
}

std::shared_ptr<AgglomeratedTopology>
Redistributor::Redistribute(const AgglomeratedTopology& topo) const
{
   // Redistribute the current topology to another set of processors
   // TODO: currently only elements and faces are redistributed, need to
   // generalize the code to cover edges and vertices as well.
   auto out = make_shared<AgglomeratedTopology>(topo.Comm_, topo.nCodim_);
   out->nDim_ = topo.nDim_;

   // Redistribute core topological data
   const int num_redElem = TrueEntityRedistribution(0).NumRows();
   out->EntityTrueEntity(0).SetUp(num_redElem);

   auto& redist0 = *redTE_TE_helper[0];
   auto& redist1 = TrueEntityRedistribution(1);

   auto redE_redTE = make_unique<ParallelCSRMatrix>(*redE_redTE_helper[1], false);
   unique_ptr<ParallelCSRMatrix> redE_TE(ParMult(redE_redTE.get(), &redist1));
   unique_ptr<ParallelCSRMatrix> TE_redE(redE_TE->Transpose());

   Array<int> redE_starts(3);
   Array<int> redTE_starts(3);
   redE_starts[0] = redE_redTE->RowPart()[0];
   redE_starts[1] = redE_redTE->RowPart()[1];
   redE_starts[2] = redE_redTE->M();
   redTE_starts[0] = redE_redTE->ColPart()[0];
   redTE_starts[1] = redE_redTE->ColPart()[1];
   redTE_starts[2] = redE_redTE->N();

   out->entityTrueEntity[1]->SetUp(
            redE_starts, redTE_starts, std::move(redE_redTE));

   // Redistribute other remaining data
   auto trueB = Assemble(topo.EntityTrueEntity(0), *topo.B_[0], topo.EntityTrueEntity(1));

   unique_ptr<ParallelCSRMatrix> tmpB(ParMult(&redist0, trueB.get()));
   unique_ptr<ParallelCSRMatrix> redB(ParMult(tmpB.get(), TE_redE.get()));

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

   unique_ptr<ParallelCSRMatrix> tE_bdrattr(ParMult(tE_E.get(), &e_bdrattr));
   unique_ptr<ParallelCSRMatrix> redE_bdrattr(ParMult(redE_TE.get(), tE_bdrattr.get()));

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
   redist0.BooleanMult(1.0, topo.element_attribute, 0.0, out->element_attribute);

   out->Weights_[0]->SetSize(num_redElem);
   redist0.Mult(topo.Weight(0), out->Weight(0));

   auto trueFacetWeight = topo.TrueWeight(1);
   out->Weights_[1]->SetSize(out->B_[0]->NumCols());
   redE_TE->Mult(*trueFacetWeight, *(out->Weights_[1]));

   return out;
}

std::unique_ptr<DofHandler> Redistributor::Redistribute(
        const DofHandler& dof,
        const std::shared_ptr<AgglomeratedTopology>& redist_topo) const
{
    auto& nonconst_dof = const_cast<DofHandler&>(dof);
    auto dof_alg = dynamic_cast<DofHandlerALG*>(&nonconst_dof);
    PARELAG_ASSERT(dof_alg); // only DofHandlerALG can be redistributed currently

    const int max_codim_base = dof.GetMaxCodimensionBaseForDof();
    unique_ptr<DofHandlerALG> out = make_unique<DofHandlerALG>(max_codim_base, redist_topo);

    auto& elem_dof = const_cast<SerialCSRMatrix&>(dof.GetEntityDofTable(AgglomeratedTopology::ELEMENT));
    auto& dof_trueDof = dof.GetDofTrueDof();
    auto elem_trueDof = Assemble(dof.GetEntityTrueEntity(0), elem_dof, dof_trueDof);

    auto redDof_trueDof = BuildRedEntToTrueEnt(*elem_trueDof);
    auto redDof_redTrueDof = BuildNewEntTrueEnt(redDof_trueDof);
    auto redTrueDof_trueDof = BuildRedTrueEntTrueEnt(redDof_redTrueDof, redDof_trueDof);
    auto trueDof_redTrueDof = matred::Transpose(redTrueDof_trueDof);
    ParallelCSRMatrix tD_redTD(trueDof_redTrueDof, false);

    for (int i = 1; i < max_codim_base; ++i)
    {
        auto codim = static_cast<AgglomeratedTopology::Entity>(i);
        auto& ent_dof = const_cast<SerialCSRMatrix&>(dof.GetEntityDofTable(codim));
        auto trueEnt_trueDof = Assemble(dof.GetEntityTrueEntity(i), ent_dof, dof_trueDof);

        unique_ptr<ParallelCSRMatrix> redTrueEnt_redTrueDof(
                    RAP(&TrueEntityRedistribution(i), trueEnt_trueDof.get(), &tD_redTD));

        SerialCSRMatrix redTrueEnt_redTrueDof_diag;
        redTrueEnt_redTrueDof->GetDiag(redTrueEnt_redTrueDof_diag); // TODO:distribute true entities and dofs
        out->entity_dof[i].reset(new SerialCSRMatrix(redTrueEnt_redTrueDof_diag)); // TODO: just steal
        out->finalized[i] = true;

        // need to convert to double and then convert the results back
        Array<int> num_int_dofs_null(dof_alg->entity_NumberOfInteriorDofsNullSpace[i].data(),
                                     dof_alg->GetNumberEntities(codim));
        Array<int> red_num_int_dofs_null(out->entity_NumberOfInteriorDofsNullSpace[i].data(),
                                         out->GetNumberEntities(codim));

        redTE_TE_helper[i]->BooleanMult(1.0, num_int_dofs_null, 0.0, red_num_int_dofs_null);

    }


}

} // namespace parelag
