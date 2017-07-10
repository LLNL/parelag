/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#include "elag_structures.hpp"

ProcessorCoarsening::ProcessorCoarsening(MPI_Comm comm) : comm(comm)
{
	right_identity = NULL;
	coarse = NULL;
	fine = NULL;
}

void ProcessorCoarsening::SetUpSameOrder(Array<int> & current_global_offsets_, Array<int> & desired_global_offsets_)
{
	current_global_offsets_.Copy(current_global_offsets);
	desired_global_offsets_.Copy(desired_global_offsets);
	int N = current_global_offsets[current_global_offsets.Size()-1];
	right_identity = hypre_IdentityParCSRMatrixOffsets(comm, N, current_global_offsets, desired_global_offsets);
}

void ProcessorCoarsening::SetUpReorder(Array<int> & partitioning, int new_num_procs,
													Array<int> & current_global_offsets_)
{
	current_global_offsets_.Copy(current_global_offsets);
	int N = current_global_offsets.Size();
	int num_entities = current_global_offsets[N-1];
		
	hypre_ParCSRMatrix * left_identity = create_reordering(partitioning, num_entities, 
																			 new_num_procs, current_global_offsets);
	// hypre_ParCSRMatrixTranspose(left_identity, &right_identity,1);	
	hypre_ParCSRMatrixTranspose2(left_identity, &right_identity);
	hypre_ParCSRMatrixDestroy(left_identity);
}


ProcessorCoarsening::~ProcessorCoarsening()
{
	if (right_identity)
		hypre_ParCSRMatrixDestroy(right_identity);
}

// not clear this should be a method of ProcessorCoarsening
// for the moment we are just putting each processor sequentially on a column
//   see also TopologyTable::TransposeOrientation() which has a similar purpose but
//   is implemented quite differently
hypre_ParCSRMatrix * ProcessorCoarsening::transpose_partitioning(Array<int> & partitioning, 
																					  int global_num_entities, int new_num_procs,
																					  int * entity_starts)
{
	int pid;
	int np;
	MPI_Comm_rank(comm, &pid);
	MPI_Comm_size(comm, &np);

	elag_assert(partitioning.Max() < new_num_procs);

	Array<int> col_starts;
	int first_col_diag, last_col_diag;
	if (HYPRE_AssumedPartitionCheck())
	{
		col_starts.SetSize(3);
		if (pid < new_num_procs) 
		{
			col_starts[0] = pid;
			col_starts[1] = pid+1;
			col_starts[2] = new_num_procs;
		} 
		else
		{
			col_starts[0] = new_num_procs;
			col_starts[1] = new_num_procs;
			col_starts[2] = new_num_procs;
		}
		first_col_diag = col_starts[0];
		last_col_diag = col_starts[1] - 1;
	}
	else
	{
		col_starts.SetSize(np+1);
		col_starts[0] = 0;
		for (int i=1; i<new_num_procs; ++i)
		{
			col_starts[i] = i;
		}
		for (int i=new_num_procs; i<np+1; ++i)
		{
			col_starts[i] = new_num_procs;
		}
		first_col_diag = col_starts[pid];
		last_col_diag = col_starts[pid+1] - 1;
	}

	// if (pid == 0) std::cout << "Making a global entity_processor matrix with " 
	//								<< global_num_entities << " rows and " 
	//								<< new_num_procs << " columns." << std::endl;

	// make local matrix with no diag/offd distinction
	int local_num_rows = partitioning.Size();
	hypre_CSRMatrix * local = hypre_CSRMatrixCreate(local_num_rows, new_num_procs, local_num_rows);
	hypre_CSRMatrixInitialize(local);
	HYPRE_Int * local_i = hypre_CSRMatrixI(local);
	HYPRE_Int * local_j = hypre_CSRMatrixJ(local);
	double * local_a = hypre_CSRMatrixData(local);
	local_i[0] = 0;
	int numlocal = 0;
	for (int i=0; i<local_num_rows; ++i)
	{
		local_i[i+1] = i+1;
		local_j[i] = partitioning[i];
		if (local_j[i] == pid) numlocal++;
		local_a[i] = 1.0;
	}
	// msg.str("");
	// msg << "[" << pid << "] local->num_nonzeros = " << local->num_nonzeros << std::endl;
	// SerializedOutput(comm, std::cout, msg.str() );
	hypre_ParCSRMatrix * entity_processor = hypre_ParCSRMatrixCreate(comm, global_num_entities,
																						  new_num_procs, entity_starts,
																						  col_starts, local_num_rows - numlocal,
																						  numlocal, local_num_rows - numlocal);
	hypre_ParCSRMatrixSetRowStartsOwner(entity_processor, 0);
	hypre_ParCSRMatrixSetColStartsOwner(entity_processor, 0);

	// NB first_col_diag, last_col_diag are INCLUSIVE, not C-style!!!
	GenerateDiagAndOffd(local, entity_processor, first_col_diag, last_col_diag);
	hypre_CSRMatrixDestroy(local);
	
	hypre_ParCSRMatrix * out;
	hypre_ParCSRMatrixTranspose2(entity_processor, &out);
	hypre_ParCSRMatrixDestroy(entity_processor);

	return out;
}

hypre_ParCSRMatrix * ProcessorCoarsening::explode_processor_entity(hypre_ParCSRMatrix * processor_entity)
{
	int pid;
	int np;
	MPI_Comm_rank(comm, &pid);
	MPI_Comm_size(comm, &np);

	hypre_CSRMatrix * in_diag = hypre_ParCSRMatrixDiag(processor_entity);
	hypre_CSRMatrix * in_offd = hypre_ParCSRMatrixOffd(processor_entity);
	HYPRE_Int * in_i_diag = hypre_CSRMatrixI(in_diag);
	HYPRE_Int * in_j_diag = hypre_CSRMatrixJ(in_diag);
	HYPRE_Int * in_i_offd = hypre_CSRMatrixI(in_offd);
	HYPRE_Int * in_j_offd = hypre_CSRMatrixJ(in_offd);

	HYPRE_Int * in_col_starts = hypre_ParCSRMatrixColStarts(processor_entity);
	int in_diag_num_cols;
	if (HYPRE_AssumedPartitionCheck())
	{
		in_diag_num_cols = in_col_starts[1] - in_col_starts[0];
	}
	else
	{
		in_diag_num_cols = in_col_starts[pid+1] - in_col_starts[pid];
	}
	int in_diag_num_nonzeros = hypre_ParCSRMatrixDiag(processor_entity)->num_nonzeros;
	int in_offd_num_nonzeros = hypre_ParCSRMatrixOffd(processor_entity)->num_nonzeros;
	int * in_col_map = hypre_ParCSRMatrixColMapOffd(processor_entity);

	int in_offd_num_cols = hypre_ParCSRMatrixOffd(processor_entity)->num_cols;
	int N = hypre_ParCSRMatrixGlobalNumCols(processor_entity);

	int numlocalrows = in_diag_num_nonzeros + in_offd_num_nonzeros;
	ParPartialSums_AssumedPartitionCheck(comm, numlocalrows, desired_global_offsets);

	hypre_ParCSRMatrix * out = hypre_ParCSRMatrixCreate(comm, N, N,
																		 desired_global_offsets, 
																		 in_col_starts, // col_starts is from processor_entity, row_starts from entity_processor, which is entity_starts, which is the original partitioning from the beginning
																		 in_offd_num_cols,
																		 in_diag_num_nonzeros,
																		 in_offd_num_nonzeros);
	hypre_ParCSRMatrixSetRowStartsOwner(out, 0); 
	hypre_ParCSRMatrixSetColStartsOwner(out, 0); 
	hypre_ParCSRMatrixSetDataOwner(out, 1);
	hypre_ParCSRMatrixSetColStartsOwner(processor_entity, 0); 

	hypre_ParCSRMatrixInitialize(out);

	int * out_col_map = hypre_ParCSRMatrixColMapOffd(out);
	HYPRE_Int * out_i_diag = hypre_CSRMatrixI(out->diag);
	HYPRE_Int * out_j_diag = hypre_CSRMatrixJ(out->diag);
	double    * out_a_diag = hypre_CSRMatrixData(out->diag);
	HYPRE_Int * out_i_offd = hypre_CSRMatrixI(out->offd);
	HYPRE_Int * out_j_offd = hypre_CSRMatrixJ(out->offd);
	double    * out_a_offd = hypre_CSRMatrixData(out->offd);

	int diag_num_nonzeros = 0;
	int offd_num_nonzeros = 0;
	out_i_diag[0] = 0;
	out_i_offd[0] = 0;
	elag_assert(in_diag->num_rows == in_offd->num_rows);
	for (int i=0; i<in_diag->num_rows; ++i)
	{
		for (int k=in_i_diag[i]; k<in_i_diag[i+1]; ++k)
		{
			out_j_diag[diag_num_nonzeros] = in_j_diag[k];
			out_a_diag[diag_num_nonzeros] = 1.0;
			diag_num_nonzeros++;
			out_i_diag[diag_num_nonzeros + offd_num_nonzeros] = diag_num_nonzeros;
			out_i_offd[diag_num_nonzeros + offd_num_nonzeros] = offd_num_nonzeros;
		}
		for (int k=in_i_offd[i]; k<in_i_offd[i+1]; ++k)
		{
			out_j_offd[offd_num_nonzeros] = offd_num_nonzeros;
			out_col_map[offd_num_nonzeros] = in_col_map[in_j_offd[k]];
			out_a_offd[offd_num_nonzeros] = 1.0;
			offd_num_nonzeros++;
			out_i_diag[diag_num_nonzeros + offd_num_nonzeros] = diag_num_nonzeros;
			out_i_offd[diag_num_nonzeros + offd_num_nonzeros] = offd_num_nonzeros;
		}
	}
	elag_assert(diag_num_nonzeros == in_diag_num_nonzeros);
	elag_assert(offd_num_nonzeros == in_offd_num_nonzeros);

	return out;
}

// TODO: some of these arguments are actually owned by the object, not necessary
hypre_ParCSRMatrix * ProcessorCoarsening::create_reordering(Array<int> & partitioning, int global_num_entities,
																				int new_num_procs, int * entity_starts)
{
	// HypreParMatrix * processor_entity = TransposeOrientation(partitioning, new_num_procs);
	hypre_ParCSRMatrix * processor_entity = transpose_partitioning(partitioning, global_num_entities,
																						new_num_procs, entity_starts);
	hypre_ParCSRMatrix * reordering = explode_processor_entity(processor_entity);
	hypre_ParCSRMatrixDestroy(processor_entity);
	return reordering;
}

HypreParMatrix * RepartitionMatrix(ProcessorCoarsening &pc, HypreParMatrix &A)
{
	elag_assert(A.M() == A.N());

	hypre_ParCSRMatrix *hout;
	hypre_BoomerAMGBuildCoarseOperator(pc.right_identity, A, pc.right_identity, &hout);

	hypre_ParCSRMatrixSetRowStartsOwner(hout, 0); 
	hypre_ParCSRMatrixSetColStartsOwner(hout, 0); 

	HypreParMatrix *out = new HypreParMatrix(hout);

	return out;
}
	
