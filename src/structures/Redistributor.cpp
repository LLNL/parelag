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
#include "utilities/mpiUtils.hpp"


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
     redTrueDof_trueDof(topo.Dimensions()+1),
     redDof_trueDof(topo.Dimensions()+1)
{
   // TODO: entities in codimension > 1 (need to adjust B)
   auto elem_redProc = matred::EntityToProcessor(topo.GetComm(), elem_redist_procs);
   auto redProc_elem = matred::Transpose(elem_redProc);
   auto redElem_elem = matred::BuildRedistributedEntityToTrueEntity(redProc_elem);
   redTrueEntity_trueEntity[0] = Move(redElem_elem); // elems are always "true"

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


unique_ptr<ParallelCSRMatrix>
Redistributor::BuildRepeatedDofToTrueDof(const DofHandler& dof, int codim_)
{
   auto codim = static_cast<AgglomeratedTopology::Entity>(codim_);
   auto dof_TrueDof = dof.GetDofTrueDof().get_entity_trueEntity();
   auto comm = dof_TrueDof->GetComm();

   auto& RDof_dof = dof.GetrDofDofTable(codim);
   mfem::Array<int> RDof_starts;
   int num_RDofs = RDof_dof.NumRows();
   ParPartialSums_AssumedPartitionCheck(comm, num_RDofs, RDof_starts);

   unique_ptr<ParallelCSRMatrix> RDof_TDof(
            dof_TrueDof->LeftDiagMult(RDof_dof, RDof_starts));
   RDof_TDof->CopyRowStarts();
   return RDof_TDof;
}


void Redistributor::RedistributeRepeatedDofs(const DofHandler& dof,
                                             DofHandler& redist_dof)
{
   auto redD_redTD = redist_dof.GetDofTrueDof().get_entity_trueEntity();
   auto dof_TDof = dof.GetDofTrueDof().get_entity_trueEntity();
   auto comm = redD_redTD->GetComm();
   const int max_codim = dof.GetMaxCodimensionBaseForDof();

   for (int i = 1; i < max_codim; ++i)
   {

      auto e_tE = dof.GetEntityTrueEntity(i).get_entity_trueEntity();
   //   auto& redE_redTE = redist_dof.GetEntityTrueEntity(i).get_entity_trueEntity();

      auto codim = static_cast<AgglomeratedTopology::Entity>(i);
      auto& ent_dof = dof.entity_dof[codim];
      auto& redEnt_redDof = redist_dof.entity_dof[codim];

//      auto& ent_rdof = dof.GetEntityRDofTable(codim);
//      auto& redEnt_redRDof = redist_dof.GetEntityRDofTable(codim);
      auto& redRDof_redDof = redist_dof.GetrDofDofTable(codim);

//      auto tE_tD = Assemble(dof.GetEntityTrueEntity(i), ent_dof, dof_trueDof);
//      auto redEnt_redDof = RAP(*redEntity_trueEntity[i], *tE_tD, *tD_redD);

//      SerialCSRMatrix redEnt_redDof_diag;
//      redEnt_redDof->GetDiag(redEnt_redDof_diag);
//      out->entity_dof[i].reset(new SerialCSRMatrix(redEnt_redDof_diag));
//      out->finalized[i] = true;


//      unique_ptr<SerialCSRMatrix> redRDof_redDof(Transpose(redDof_redRDof));

      mfem::Array<int> redRDof_starts;
      int num_redRDofs = redRDof_redDof.NumRows();
      ParPartialSums_AssumedPartitionCheck(comm, num_redRDofs, redRDof_starts);

      unique_ptr<ParallelCSRMatrix> redRDof_redTD(
               redD_redTD->LeftDiagMult(redRDof_redDof, redRDof_starts));




//      SerialCSRMatrix redE_TE_diag, redE_TE_offd;
//      HYPRE_Int * trash_map;
//      redEntity_trueEntity[i]->GetDiag(redE_TE_diag);
//      redEntity_trueEntity[i]->GetOffd(redE_TE_offd, trash_map);


      auto& RDof_dof = dof.GetrDofDofTable(codim);
      mfem::Array<int> RDof_starts;
      int num_RDofs = RDof_dof.NumRows();
      ParPartialSums_AssumedPartitionCheck(comm, num_RDofs, RDof_starts);

      unique_ptr<ParallelCSRMatrix> RDof_TD(
               dof_TDof->LeftDiagMult(RDof_dof, RDof_starts));



//      auto redRDof_TD = Mult(*redRDof_redTD, *redTrueDof_trueDof[max_codim]);

//      for (int ent = 0; ent < ent_dof->NumRows(); ++ ent)
//      {
//         for (int j = 0; j < ent_dof->RowSize(ent); ++j)
//         {

//         }
//      }
   }

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

   auto redElem_redTrueElem = out->EntityTrueEntity(0).get_entity_trueEntity();
   redEntity_trueEntity[0] = Mult(*redElem_redTrueElem, *redTrueEntity_trueEntity[0]);

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

    const int dim = redist_topo->Dimensions();
    const int max_codim = dof.GetMaxCodimensionBaseForDof();
    const int jform = dim - max_codim;
    auto out = make_unique<DofHandlerALG>(max_codim, redist_topo);

    auto elem = AgglomeratedTopology::ELEMENT;
    auto& elem_dof = const_cast<SerialCSRMatrix&>(dof.GetEntityDofTable(elem));
    auto& dof_trueDof = dof.GetDofTrueDof();
    auto elem_trueDof = Assemble(dof.GetEntityTrueEntity(0), elem_dof, dof_trueDof);

    redDof_trueDof[jform] = BuildRedEntToTrueEnt(*elem_trueDof);
    auto redDof_redTrueDof = BuildRedEntToRedTrueEnt(*redDof_trueDof[jform]);
    redTrueDof_trueDof[jform] = // TODO: prabably need to take case of rdof ordering
          BuildRedTrueEntToTrueEnt(*redDof_redTrueDof, *redDof_trueDof[jform]);

    std::unique_ptr<ParallelCSRMatrix> tD_redD(redDof_trueDof[jform]->Transpose());
    out->dofTrueDof.SetUp(std::move(redDof_redTrueDof));

    for (int i = 0; i < max_codim+1; ++i)
    {
       std::unique_ptr<ParallelCSRMatrix> tE_tD;
       if (i > 0)
       {
          auto codim = static_cast<AgglomeratedTopology::Entity>(i);
          auto& ent_dof = const_cast<SerialCSRMatrix&>(dof.GetEntityDofTable(codim));
          tE_tD = IgnoreNonLocalRange(dof.GetEntityTrueEntity(i), ent_dof, dof_trueDof);
       }
       else
       {
          tE_tD.reset(elem_trueDof.release());
       }

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


std::shared_ptr<DeRhamSequenceAlg> Redistributor::Redistribute(
      const DeRhamSequence& sequence,
      const std::shared_ptr<AgglomeratedTopology>& redist_topo)
{
   const int dim = redist_topo->Dimensions();
   const int num_forms = sequence.GetNumberOfForms();
   auto redist_seq = std::make_shared<DeRhamSequenceAlg>(redist_topo, num_forms);
   redist_seq->Targets_.resize(num_forms);

   for (int codim = 0; codim < num_forms; ++codim)
   {
      const int jform = num_forms-codim-1;
      if (jform < sequence.jformStart_) { break; }

      auto type = static_cast<AgglomeratedTopology::Entity>(codim);

//      auto& redTE_tE = TrueEntityRedistribution(codim);
      auto& dof_handler = *sequence.Dof_[jform];

      redist_seq->Dof_[jform] = Redistribute(dof_handler, redist_topo);
      auto& redD_tD = RedistributedDofToTrueDof(jform);

      if (jform != (num_forms - 1))
      {
         auto trueD = sequence.ComputeTrueD(jform);
         auto redD1_tD1 = RedistributedDofToTrueDof(jform+1);
         unique_ptr<ParallelCSRMatrix> tD_redD(redD_tD.Transpose());
         auto redD = RAP(redD1_tD1, *trueD, *tD_redD);

         SerialCSRMatrix redD_diag;
         redD->GetDiag(redD_diag);
         redist_seq->D_[jform].reset(new SerialCSRMatrix(redD_diag));
      }

      // redistribution of M is taken out of this look

      const int true_size = dof_handler.GetDofTrueDof().GetTrueLocalSize();
      auto& Targets = *(sequence.Targets_[jform]);
      MultiVector trueTargets(Targets.NumberOfVectors(), true_size);
      trueTargets = 0.0;
      dof_handler.AssembleGlobalMultiVector(type, Targets, trueTargets);

      const int redist_size = redist_seq->Dof_[jform]->GetNDofs();
      redist_seq->Targets_[jform].reset(
               new MultiVector(trueTargets.NumberOfVectors(), redist_size));
      Mult(redD_tD, trueTargets, *(redist_seq->Targets_[jform]));
   }


   for (int codim = 0; codim < num_forms; ++codim)
   {
      const int jform = num_forms-codim-1;
      if (jform < sequence.jformStart_) { break; }

      auto type = static_cast<AgglomeratedTopology::Entity>(codim);

      for (int j = sequence.jformStart_; j <= jform; ++j)
      {
         const int idx = (dim-j)*(num_forms-j)/2 + codim;
         //              M_[idx]; // TODO: need to match with rdof, shared enetities need to combine first

         auto& dof_handler = *sequence.Dof_[j];

         auto RD_TD = BuildRepeatedDofToTrueDof(dof_handler, codim);
         auto redRD_redTD = BuildRepeatedDofToTrueDof(*redist_seq->Dof_[j], codim);
         unique_ptr<ParallelCSRMatrix> redTD_redRD(redRD_redTD->Transpose());
         unique_ptr<ParallelCSRMatrix> tD_redTD(redTrueDof_trueDof[j]->Transpose());


         SerialCSRMatrix M(*const_cast<DeRhamSequence&>(sequence).GetM(type, j));
         ParallelCSRMatrix pM(RD_TD->GetComm(), RD_TD->M(), RD_TD->RowPart(), &M);
         auto trueM = IgnoreNonLocalRange(*RD_TD, pM, *RD_TD);

         unique_ptr<ParallelCSRMatrix> red_trueM(
                  mfem::RAP(tD_redTD.get(), trueM.get(), tD_redTD.get()));

         unique_ptr<ParallelCSRMatrix> redM(mfem::RAP(red_trueM.get(), redTD_redRD.get()));

         SerialCSRMatrix redM_diag;
         redM->GetDiag(redM_diag);
         redist_seq->M_[idx].reset(new SerialCSRMatrix(redM_diag));
      }
   }

   auto redTD_tD = TrueDofRedistribution(dim);
   redist_seq->L2_const_rep_.SetSize(redTD_tD.NumRows());
   redTD_tD.Mult(sequence.L2_const_rep_, redist_seq->L2_const_rep_);

   return redist_seq;
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

   for (int i = 0; i < Ax.Size(); ++i)
   {
      Ax[i] = Ax_vec[i];
   }
}

} // namespace parelag
