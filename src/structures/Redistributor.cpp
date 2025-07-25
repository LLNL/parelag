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
#include "hypreExtension/hypreExtension.hpp"

#ifdef ParELAG_ENABLE_MATRED

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

std::vector<int> RedistributeElements(
      ParallelCSRMatrix& elem_face, int& num_redist_procs)
{
    MPI_Comm comm = elem_face.GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);

    mfem::Array<int> proc_starts, perm_rowstarts;
    int num_procs_loc = elem_face.NumRows() > 0 ? 1 : 0;
    ParPartialSums_AssumedPartitionCheck(comm, num_procs_loc, proc_starts);

    int num_procs = proc_starts.Last();
    ParPartialSums_AssumedPartitionCheck(comm, num_procs, perm_rowstarts);

    SerialCSRMatrix proc_elem(num_procs_loc, elem_face.NumRows());
    if (num_procs_loc == 1)
    {
       for (int j = 0 ; j < proc_elem.NumCols(); ++j)
       {
           proc_elem.Set(0, j, 1.0);
       }
    }
    proc_elem.Finalize();

    unique_ptr<ParallelCSRMatrix> proc_face(
             elem_face.LeftDiagMult(proc_elem, proc_starts));
    unique_ptr<ParallelCSRMatrix> face_proc(proc_face->Transpose());
    auto proc_proc = Mult(*proc_face, *face_proc, false);

    mfem::Array<HYPRE_Int> proc_colmap(num_procs-num_procs_loc);

    SerialCSRMatrix perm_diag(num_procs, num_procs_loc);
    SerialCSRMatrix perm_offd(num_procs, num_procs-num_procs_loc);
    int offd_proc_count = 0;
    for (int i = 0 ; i < num_procs; ++i)
    {
       if (i == myid)
       {
          perm_diag.Set(i, 0, 1.0);
       }
       else
       {
          perm_offd.Set(i, offd_proc_count, 1.0);
          proc_colmap[offd_proc_count++] = i;
       }
    }
    perm_diag.Finalize();
    perm_offd.Finalize();

    int num_perm_rows = perm_rowstarts.Last();
    ParallelCSRMatrix permute(comm, num_perm_rows, num_procs, perm_rowstarts,
                              proc_starts, &perm_diag, &perm_offd, proc_colmap);

    unique_ptr<ParallelCSRMatrix> permuteT(permute.Transpose());
    auto permProc_permProc = parelag::RAP(permute, *proc_proc, *permuteT);

    SerialCSRMatrix globProc_globProc;
    permProc_permProc->GetDiag(globProc_globProc);

    std::vector<int> out(elem_face.NumRows());
    if (elem_face.NumRows() > 0)
    {
        PARELAG_ASSERT_DEBUG(IsConnected(globProc_globProc));

        mfem::Array<int> partition;
        MetisGraphPartitioner partitioner;
        partitioner.setParELAGDefaultMetisOptions();
        partitioner.setParELAGDefaultFlags(globProc_globProc.NumRows()/num_redist_procs);
        partitioner.doPartition(globProc_globProc, num_redist_procs, partition);

        PARELAG_ASSERT(myid < partition.Size());
        std::fill_n(out.begin(), elem_face.NumRows(), partition[myid]);
    }
    return out;
}

Redistributor::Redistributor(
      const AgglomeratedTopology& topo, const std::vector<int>& elem_redist_procs)
   : redTrueEntity_trueEntity(topo.Codimensions()+1),
     redEntity_trueEntity(topo.Codimensions()+1),
     redTrueDof_trueDof(topo.Dimensions()+1),
     redDof_trueDof(topo.Dimensions()+1)
{
   Init(topo, elem_redist_procs);
}

Redistributor::Redistributor(const AgglomeratedTopology& topo, int& num_redist_procs)
   : redTrueEntity_trueEntity(topo.Codimensions()+1),
     redEntity_trueEntity(topo.Codimensions()+1),
     redTrueDof_trueDof(topo.Dimensions()+1),
     redDof_trueDof(topo.Dimensions()+1)
{
   auto elem_redist_procs = RedistributeElements(topo.TrueB(0), num_redist_procs);
   Init(topo, elem_redist_procs);
}

void Redistributor::Init(
   const AgglomeratedTopology& topo, const std::vector<int>& elem_redist_procs)
{
   // TODO: entities in codimension > 1 (need to adjust B)
   auto elem_redProc = matred::EntityToProcessor(topo.GetComm(), elem_redist_procs);
   auto redProc_elem = matred::Transpose(elem_redProc);
   auto redElem_elem = matred::BuildRedistributedEntityToTrueEntity(redProc_elem);
   redTrueEntity_trueEntity[0] = Move(redElem_elem); // elems are always "true"

   auto elem_trueEntity = const_cast<AgglomeratedTopology&>(topo).TrueB(0);
   redEntity_trueEntity[1] = BuildRedEntToTrueEnt(elem_trueEntity);

   redist_topo = Redistribute(topo);
}

unique_ptr<ParallelCSRMatrix>
Redistributor::BuildRedEntToTrueEnt(const ParallelCSRMatrix& elem_tE) const
{
    auto redElem_tE = Mult(*redTrueEntity_trueEntity[0], elem_tE);
    hypre_DropSmallEntries(*redElem_tE, 1e-6);

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
    unique_ptr<ParallelCSRMatrix> out = Mult(*redTE_redE, redE_tE, true);
    hypre_DropSmallEntries(*out, 1e-6);
    *out = 1.0;
    return out;
}


unique_ptr<ParallelCSRMatrix>
Redistributor::BuildRepeatedDofToTrueDof(const DofHandler& dof, int codim)
{
   auto type = static_cast<AgglomeratedTopology::Entity>(codim);
   auto dof_TrueDof = dof.GetDofTrueDof().get_entity_trueEntity();
   auto comm = dof_TrueDof->GetComm();

   auto& RDof_dof = dof.GetrDofDofTable(type);
   mfem::Array<int> RDof_starts;
   int num_RDofs = RDof_dof.NumRows();
   ParPartialSums_AssumedPartitionCheck(comm, num_RDofs, RDof_starts);

   unique_ptr<ParallelCSRMatrix> RDof_TDof(
            dof_TrueDof->LeftDiagMult(RDof_dof, RDof_starts));
   RDof_TDof->CopyRowStarts();
   return RDof_TDof;
}

unique_ptr<ParallelCSRMatrix>
Redistributor::BuildRepeatedDofRedistribution(const AgglomeratedTopology& topo,
                                              const DofHandler& dof,
                                              const DofHandler& redist_dof,
                                              int codim, int jform)
{
   auto type = static_cast<AgglomeratedTopology::Entity>(codim);
   auto& redTE_TE = *redTrueEntity_trueEntity[codim];
   auto comm = redTE_TE.GetComm();

   auto E_RD = ToParMatrix(comm, dof.GetEntityRDofTable(type));
   auto E_TE = topo.EntityTrueEntity(codim).get_entity_trueEntity();
   unique_ptr<ParallelCSRMatrix> TE_E(E_TE->Transpose());
   auto TE_RD = Mult(*TE_E, *E_RD);

   auto redE_redRD = ToParMatrix(comm, redist_dof.GetEntityRDofTable(type));
   auto redE_redTE = redist_topo->EntityTrueEntity(codim).get_entity_trueEntity();
   unique_ptr<ParallelCSRMatrix> redRD_redE(redE_redRD->Transpose());
   auto redRD_redTE = Mult(*redRD_redE, *redE_redTE);

   auto redRD_TE_RD = RAP(*redRD_redTE, redTE_TE, *TE_RD);

   // Find intersection of redRDof_TrueElem_RDof and redRDof_TrueDof_RDof
   auto RD_TD = BuildRepeatedDofToTrueDof(dof, codim);
   unique_ptr<ParallelCSRMatrix> TD_RD(RD_TD->Transpose());
   auto redRD_redTD = BuildRepeatedDofToTrueDof(redist_dof, codim);
   auto redRD_TD_RD = RAP(*redRD_redTD, *redTrueDof_trueDof[jform], *TD_RD);

   mfem::SparseMatrix redRD_TE_RD_diag, redRD_TE_RD_offd;
   mfem::SparseMatrix redRD_TD_RD_diag, redRD_TD_RD_offd;
   HYPRE_BigInt *redRD_TE_RD_colmap, *redRD_TD_RD_colmap;

   redRD_TE_RD->GetDiag(redRD_TE_RD_diag);
   redRD_TE_RD->GetOffd(redRD_TE_RD_offd, redRD_TE_RD_colmap);
   redRD_TD_RD->GetDiag(redRD_TD_RD_diag);
   redRD_TD_RD->GetOffd(redRD_TD_RD_offd, redRD_TD_RD_colmap);

   HYPRE_BigInt * out_colmap = new HYPRE_BigInt[redRD_TE_RD_offd.NumCols()];
   std::copy_n(redRD_TE_RD_colmap, redRD_TE_RD_offd.NumCols(), out_colmap);

   auto out_diag = new SerialCSRMatrix(redRD_TE_RD_diag.NumRows(),
                                       redRD_TE_RD_diag.NumCols());
   auto out_offd = new SerialCSRMatrix(redRD_TE_RD_offd.NumRows(),
                                       redRD_TE_RD_offd.NumCols());
   for (int i = 0; i < redRD_TE_RD->NumRows(); ++i)
   {
       if (redRD_TE_RD_diag.RowSize(i) > 0)
       {
           for (int j = 0; j < redRD_TE_RD_diag.RowSize(i); ++j)
           {
               int RD = redRD_TE_RD_diag.GetRowColumns(i)[j];
               mfem::Array<int> RDs(redRD_TD_RD_diag.GetRowColumns(i),
                                    redRD_TD_RD_diag.RowSize(i));
               if (RDs.Find(RD) != -1)
               {
                   out_diag->Add(i, RD, 1.0);
                   break;
               }
           }
           assert(out_diag->RowSize(i) == 1);
       }
       else
       {
           assert(redRD_TE_RD_offd.RowSize(i) > 0);
           for (int j = 0; j < redRD_TE_RD_offd.RowSize(i); ++j)
           {
               int RD = redRD_TE_RD_offd.GetRowColumns(i)[j];
               HYPRE_Int RD_global = redRD_TE_RD_colmap[RD];
               mfem::Array<int> RDs(redRD_TD_RD_offd.RowSize(i));
               for (int k = 0; k < RDs.Size(); ++k)
               {
                   RDs[k] = redRD_TD_RD_colmap[redRD_TD_RD_offd.GetRowColumns(i)[k]];
               }
               if (RDs.Find(RD_global) != -1)
               {
                   out_offd->Add(i, RD, 1.0);
                   break;
               }
           }
           assert(out_offd->RowSize(i) == 1);
       }
   }
   out_diag->Finalize();
   out_offd->Finalize();

   auto redRD_RD = make_unique<ParallelCSRMatrix>(
               comm, redE_redRD->N(), E_RD->N(), redE_redRD->ColPart(),
               E_RD->ColPart(), out_diag, out_offd, out_colmap, true);
   redRD_RD->CopyRowStarts();
   redRD_RD->CopyColStarts();
   redRD_RD->SetOwnerFlags(3, 3, 1);
   return redRD_RD;
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
   hypre_DropSmallEntries(*redEntity_trueEntity[0], 1e-6);

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

std::unique_ptr<DofHandler> Redistributor::Redistribute(const DofHandler& dof)
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

    int myid;
    MPI_Comm_rank(redist_topo->GetComm(), &myid);

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
    }

    return out;
}

std::shared_ptr<DeRhamSequenceAlg>
Redistributor::Redistribute(const DeRhamSequence& sequence)
{
   const int dim = redist_topo->Dimensions();
   const int num_forms = sequence.GetNumberOfForms();
   auto redist_seq = std::make_shared<DeRhamSequenceAlg>(redist_topo, num_forms);
   redist_seq->Targets_.resize(num_forms);

   for (int codim = 0; codim < num_forms; ++codim)
   {
      const int jform = num_forms-codim-1;
      if (jform < sequence.jformStart_) { break; }
      DofHandler& dof_handler = *sequence.Dof_[jform];

      redist_seq->Dof_[jform] = Redistribute(dof_handler);
      auto& redD_tD = redDof_trueDof[jform];

      if (jform != (num_forms - 1))
      {
         auto trueD = sequence.ComputeTrueD(jform);
         unique_ptr<ParallelCSRMatrix> tD_redD(redD_tD->Transpose());
         auto redD = RAP(*redDof_trueDof[jform+1], *trueD, *tD_redD);

         SerialCSRMatrix redD_diag;
         redD->GetDiag(redD_diag);
         redist_seq->D_[jform].reset(new SerialCSRMatrix(redD_diag));
      }

      // redistribution of M is taken out of this look

      const int true_size = dof_handler.GetDofTrueDof().GetTrueLocalSize();
      auto& Targets = *(sequence.Targets_[jform]);
      MultiVector trueTargets(Targets.NumberOfVectors(), true_size);
      trueTargets = 0.0;
      dof_handler.GetDofTrueDof().IgnoreNonLocal(Targets, trueTargets);

      const int redist_size = redist_seq->Dof_[jform]->GetNDofs();
      redist_seq->Targets_[jform].reset(
               new MultiVector(trueTargets.NumberOfVectors(), redist_size));
      Mult(*redD_tD, trueTargets, *(redist_seq->Targets_[jform]));
   }

   for (int codim = 0; codim < num_forms; ++codim)
   {
      const int jform = num_forms-codim-1;
      if (jform < sequence.jformStart_) { break; }

      auto type = static_cast<AgglomeratedTopology::Entity>(codim);

      for (int j = sequence.jformStart_; j <= jform; ++j)
      {
         SerialCSRMatrix M(*const_cast<DeRhamSequence&>(sequence).GetM(type, j));
         auto pM = ToParMatrix(redist_topo->GetComm(), M);

         unique_ptr<ParallelCSRMatrix> redM;
         if (codim == 0) // redistribution of RDofs when codim=0 follows that of elements
         {
             auto topo = const_cast<DeRhamSequence&>(sequence).GetTopology();
             auto redRD_RD = BuildRepeatedDofRedistribution(
                     *topo, *sequence.Dof_[j], *redist_seq->Dof_[j], codim, j);
             unique_ptr<ParallelCSRMatrix> RD_redRD(redRD_RD->Transpose());
             redM = parelag::RAP(*redRD_RD, *pM, *RD_redRD);
         }
         else if (codim == 1) // codim-1 RDofs when jform=dim-2 are identified with true dofs
         {
             auto RD_TD = BuildRepeatedDofToTrueDof(*sequence.Dof_[j], codim);
             auto redRD_redTD = BuildRepeatedDofToTrueDof(*redist_seq->Dof_[j], codim);
             unique_ptr<ParallelCSRMatrix> redTD_redRD(redRD_redTD->Transpose());
             unique_ptr<ParallelCSRMatrix> tD_redTD(redTrueDof_trueDof[j]->Transpose());

             auto trueM = IgnoreNonLocalRange(*RD_TD, *pM, *RD_TD);

             unique_ptr<ParallelCSRMatrix> red_trueM(
                      mfem::RAP(tD_redTD.get(), trueM.get(), tD_redTD.get()));
             redM.reset(mfem::RAP(red_trueM.get(), redTD_redRD.get()));
         }
         else
         {
             PARELAG_TEST_FOR_EXCEPTION(
                     true, std::runtime_error,
                     "redistribution of M when codim > 1 is not implemented.");
         }

         SerialCSRMatrix redM_diag;
         redM->GetDiag(redM_diag);

         const int idx = (dim-j)*(num_forms-j)/2 + codim;
         redist_seq->M_[idx].reset(new SerialCSRMatrix(redM_diag));
      }
   }

   auto redTD_tD = TrueDofRedistribution(dim);
   redist_seq->L2_const_rep_.SetSize(redTD_tD.NumRows());
   redTD_tD.Mult(sequence.L2_const_rep_, redist_seq->L2_const_rep_);

   redist_seq->SetSVDTol(sequence.GetSVDTol());

   return redist_seq;
}

} // namespace parelag

#endif // ParELAG_ENABLE_MATRED
