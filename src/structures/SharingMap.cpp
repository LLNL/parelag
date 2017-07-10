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
#include "../amge/elag_amge.hpp"
#include "general/sort_pairs.hpp"
#include <cmath>


SharingMap::SharingMap(MPI_Comm comm_):
	comm(comm_),
	assumedPID(0),
	assumedNumProc(2),
	entity_start(0),
	trueEntity_start(0),
	entity_trueEntity(static_cast<hypre_ParCSRMatrix*>(NULL)),
	entity_trueEntity_entity(static_cast<hypre_ParCSRMatrix*>(NULL)),
	helpData(0),
	helpTrueData(0),
	xtrue(static_cast<hypre_ParVector *>(NULL)),
	x(static_cast<hypre_ParVector *>(NULL))
{
	if(!HYPRE_AssumedPartitionCheck())
	{
		MPI_Comm_rank(comm, &assumedPID);
		MPI_Comm_size(comm, &assumedNumProc);
	}

	entity_start.SetSize(assumedNumProc+1);
	trueEntity_start.SetSize(assumedNumProc+1);
}

void SharingMap::SetUp(Array<int> & entityStart, Array<int> & trueEntityStart, hypre_ParCSRMatrix * entity_trueEntity_)
{
	elag_trace_enter_block("SharingMap::SetUp(entityStart, trueEntityStart, entity_trueEntity_)");
	if(&entity_start != &entityStart)
		entityStart.Copy(entity_start);
	if(&trueEntity_start != &trueEntityStart)
		trueEntityStart.Copy(trueEntity_start);
	entity_trueEntity = entity_trueEntity_;

	elag_assert( hypre_ParCSRMatrixOwnsRowStarts(entity_trueEntity) == 0 );
	elag_assert( hypre_ParCSRMatrixOwnsColStarts(entity_trueEntity) == 0 );

	hypre_ParCSRMatrixRowStarts(entity_trueEntity) = entity_start.GetData();
	hypre_ParCSRMatrixColStarts(entity_trueEntity) = trueEntity_start.GetData();

	hypre_ParCSRMatrix * tE_e;
	elag_trace("Transpose entity_trueEntity");
	hypre_ParCSRMatrixTranspose2(entity_trueEntity, &tE_e);
	elag_trace("Transpose entity_trueEntity - done!");
	elag_assert( hypre_ParCSRMatrixOwnsRowStarts(tE_e) == 0 );
	elag_assert( hypre_ParCSRMatrixOwnsColStarts(tE_e) == 0 );
	elag_trace("Compute entity_trueEntity_entity");
	entity_trueEntity_entity = hypre_ParMatmul(entity_trueEntity, tE_e);
	elag_trace("Compute entity_trueEntity_entity - done!");
	hypre_ParCSRMatrixDestroy( tE_e );

	elag_assert( hypre_ParCSRMatrixOwnsRowStarts(entity_trueEntity_entity) == 0 );
	elag_assert( hypre_ParCSRMatrixOwnsColStarts(entity_trueEntity_entity) == 0 );

	resetHypreParVectors();
	storeSharedEntitiesIds();
	elag_trace_leave_block("SharingMap::SetUp(entityStart, trueEntityStart, entity_trueEntity_)");
}

void SharingMap::resetHypreParVectors()
{
	hypre_ParVectorDestroy(xtrue);
	hypre_ParVectorDestroy(x);

	xtrue = hypre_ParVectorCreate( comm, trueEntity_start[assumedNumProc], trueEntity_start.GetData() );
	x     = hypre_ParVectorCreate( comm, entity_start[assumedNumProc], entity_start.GetData() );

	hypre_ParVectorSetPartitioningOwner(xtrue, 0);
	hypre_ParVectorSetPartitioningOwner(x,     0);
	hypre_SeqVectorSetDataOwner( hypre_ParVectorLocalVector(xtrue), 0 );
	hypre_SeqVectorSetDataOwner( hypre_ParVectorLocalVector(x), 0 );

	//Trick: we would like xtrue and x to be initialized, but still have null data...
	// So we first give xtrue and x some pointers, then we initialize the vectors, and finally we reset the data pointer to null.
	double  a;
	hypre_VectorData( hypre_ParVectorLocalVector(xtrue) ) = &a;
	hypre_VectorData( hypre_ParVectorLocalVector(x) ) = &a;
	hypre_ParVectorInitialize(xtrue);
	hypre_ParVectorInitialize(x);

	hypre_VectorData( hypre_ParVectorLocalVector(xtrue) ) = NULL;
	hypre_VectorData( hypre_ParVectorLocalVector(x) ) = NULL;
}

void SharingMap::storeSharedEntitiesIds()
{
	int nLocEntities = GetLocalSize();

	hypre_CSRMatrix * offd_entity_trueEntity_entity = hypre_ParCSRMatrixOffd(entity_trueEntity_entity);
	hypre_CSRMatrix * offd_entity_trueEntity = hypre_ParCSRMatrixOffd(entity_trueEntity);

	int nSharedIds = 0;
	{
		int * i_offd = hypre_CSRMatrixI(offd_entity_trueEntity_entity);
		for(int i(0); i < nLocEntities; ++i)
			if(i_offd[i+1] - i_offd[i]) ++nSharedIds;
	}

	int nSharedNotOwnedIds = 0;
	{
		int * i_offd = hypre_CSRMatrixI(offd_entity_trueEntity);
		for(int i(0); i < nLocEntities; ++i)
			if(i_offd[i+1] - i_offd[i]) ++nSharedNotOwnedIds;
	}

	nOwnedSharedEntities = nSharedIds - nSharedNotOwnedIds;

	sharedEntityIds.SetSize(nSharedIds);
	int offset_not_owned = nOwnedSharedEntities;
	int owned_counter = 0;
	int notowned_counter = 0;

	for(int i(0); i < nLocEntities; ++i)
	{
		switch(IsShared(i))
		{
			case -1:
				sharedEntityIds[offset_not_owned + notowned_counter] = i;
				++notowned_counter;
				break;
			case 0:
				break;
			case 1:
				sharedEntityIds[owned_counter] = i;
				++owned_counter;
				break;
			default:
				elag_error_msg(1,"The Impossible Has Happened!");
				break;
		}
	}
	elag_assert(owned_counter == nOwnedSharedEntities);
	elag_assert(notowned_counter == nSharedNotOwnedIds);
}

void SharingMap::SetUp(ParMesh * pmesh, int codim)
{
	elag_trace_enter_block("SharingMap::SetUp(pmesh = " << pmesh << ", codim = " << codim << ")" );
	FiniteElementCollection * fecColl;
	int ndim = pmesh->Dimension();
	switch(codim)
	{
	case 0:
		fecColl = new L2_FECollection(0, ndim );
		break;
	case 1:
		fecColl = new RT_FECollection(0, ndim );
		break;
	case 2:
		fecColl = (ndim == 2) ?
				  static_cast<FiniteElementCollection *>(new H1_FECollection(1, ndim )) :
				  static_cast<FiniteElementCollection *>(new ND_FECollection(1, ndim ));
		break;
	case 3:
		elag_assert(ndim == 3);
		fecColl = new H1_FECollection(1, ndim );
		break;
	default:
		elag_error_msg(1, "Wrong Codimension number \n");
		fecColl = static_cast<FiniteElementCollection *>(NULL);
		break;
	}

	ParFiniteElementSpace * feSpace = new ParFiniteElementSpace(pmesh, fecColl);
	SetUp(feSpace, 1);

	delete feSpace;
	delete fecColl;
	elag_trace_leave_block("SharingMap::SetUp(pmesh = " << pmesh << ", codim = " << codim << ")" );
}

void SharingMap::SetUp(ParFiniteElementSpace * fes, int useDofSign)
{
	elag_trace_enter_block("SharingMap::SetUp(fes = " << fes << ", useDofSign = " << useDofSign << ")");
	Array<int> estart(fes->GetDofOffsets(), assumedNumProc+1);
	Array<int> etstart(fes->GetTrueDofOffsets(), assumedNumProc+1);
	elag_trace("Get the dotTrueDof matrix from fes");
	hypre_ParCSRMatrix * mat = fes->Dof_TrueDof_Matrix()->StealData();
	elag_trace("Get the dotTrueDof matrix from fes - done");
	if(useDofSign)
	{
		hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd( mat );
		HYPRE_Int * i_offd = hypre_CSRMatrixI(offd);
		double * a_offd = hypre_CSRMatrixData(offd);
		int nrows = hypre_CSRMatrixNumRows(offd);
		for(int i = 0; i < nrows; ++i)
		{
			elag_assert( (i_offd[i+1] - i_offd[i]) < 2);
			if( (i_offd[i+1] - i_offd[i]) != 0 && fes->GetDofSign(i) == -1)
				a_offd[i_offd[i]] *= -1.;
		}

		hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag( mat );
		HYPRE_Int * i_diag = hypre_CSRMatrixI(diag);
		double * a_diag = hypre_CSRMatrixData(diag);
		for(int i = 0; i < nrows; ++i)
		{
			elag_assert( (i_diag[i+1] - i_diag[i]) < 2);
			if( (i_diag[i+1] - i_diag[i]) != 0 && fes->GetDofSign(i) == -1)
				a_diag[i_diag[i]] *= -1.;
		}

	}
	SetUp(estart, etstart, mat );
	elag_trace_leave_block("SharingMap::SetUp(fes = " << fes << ", useDofSign = " << useDofSign << ")");
}

void SharingMap::SetUp(int localSize)
{

	ParPartialSums_AssumedPartitionCheck(comm, localSize, entity_start);

	trueEntity_start.MakeRef(entity_start);

	entity_trueEntity = hypre_IdentityParCSRMatrix( comm, entity_start.Last(), entity_start.GetData() );

	hypre_ParCSRMatrix * tE_e;
	hypre_ParCSRMatrixTranspose2(entity_trueEntity, &tE_e);
	elag_assert( hypre_ParCSRMatrixOwnsRowStarts(tE_e) == 0 );
	elag_assert( hypre_ParCSRMatrixOwnsColStarts(tE_e) == 0 );
	entity_trueEntity_entity = hypre_ParMatmul(entity_trueEntity, tE_e);
	hypre_ParCSRMatrixDestroy(tE_e);

	elag_assert( hypre_ParCSRMatrixOwnsRowStarts(entity_trueEntity_entity) == 0 );
	elag_assert( hypre_ParCSRMatrixOwnsColStarts(entity_trueEntity_entity) == 0 );

	resetHypreParVectors();
	storeSharedEntitiesIds();
}

void SharingMap::SetUp(SerialCSRMatrix & e_AE, SharingMap & e_Te)
{
	elag_assert( e_AE.Size() == e_Te.GetLocalSize() );
	int localSize = e_AE.Width();
	ParPartialSums_AssumedPartitionCheck(comm, localSize, entity_start);

	ParallelCSRMatrix e_AEd(e_Te.comm, e_Te.entity_start.Last(), entity_start.Last(),
			                e_Te.entity_start.GetData(), entity_start.GetData(), &e_AE);

	hypre_BoomerAMGBuildCoarseOperator(e_AEd, e_Te.entity_trueEntity_entity, e_AEd, &entity_trueEntity_entity );
	//hypre_BoomerAMG gives by default ownership of the column/row partitioning to the result. We don't want that :)
	hypre_ParCSRMatrixSetRowStartsOwner(entity_trueEntity_entity,0);
	hypre_ParCSRMatrixSetColStartsOwner(entity_trueEntity_entity,0);

	hypre_ParCSRDataTransformationSign(entity_trueEntity_entity);

	entity_trueEntity = ParUnique(entity_trueEntity_entity, trueEntity_start);


	resetHypreParVectors();
	storeSharedEntitiesIds();

	elag_assert( DebugCheck() == 0 );
}

int SharingMap::DebugCheck()
{
	hypre_CSRMatrix * diag_ee = hypre_ParCSRMatrixDiag(entity_trueEntity_entity);

	elag_assert( hypre_CSRMatrixNumNonzeros(diag_ee) == hypre_CSRMatrixNumRows(diag_ee) );
	elag_assert( hypre_CSRMatrixNumNonzeros(diag_ee) == hypre_CSRMatrixNumCols(diag_ee) );

	double * data_diag_ee = hypre_CSRMatrixData(diag_ee);
	for(int i = 0; i <hypre_CSRMatrixNumNonzeros(diag_ee); ++i)
		elag_assert( fabs(data_diag_ee[i] - 1.) < 1e-9 );

	hypre_ParCSRMatrix * tmp;
	hypre_ParCSRMatrixTranspose2(entity_trueEntity, &tmp);
	hypre_ParCSRMatrix * e_T_e = hypre_ParMatmul(entity_trueEntity, tmp);

	int ierr = hypre_ParCSRMatrixCompare(e_T_e, entity_trueEntity_entity, 1e-9, 1);

	if(ierr)
	{
		hypre_ParCSRMatrixPrintIJ(entity_trueEntity, 0, 0, "e_Te");
		hypre_ParCSRMatrixPrintIJ(entity_trueEntity_entity, 0, 0, "e_Te_e");
	}

	hypre_ParCSRMatrixDestroy(e_T_e);
	hypre_ParCSRMatrixDestroy(tmp);

	return ierr;
}

void SharingMap::SetUp(DofHandler & dof)
{
	elag_error_msg(1, "void SharingMap::SetUp(DofHandler & dof) is deprecated!");
	std::cout<<"WARNING!! SharingMap::SetUp(DofHandler & dof) only works if I have 1 dof for each entity!!!\n";
	DofHandlerALG & dof_alg = dynamic_cast<DofHandlerALG &>(dof);
	int baseCodim = dof.GetMaxCodimensionBaseForDof();
	int localSize = dof_alg.GetNDofs();
	ParPartialSums_AssumedPartitionCheck(comm, localSize, entity_start);

	for(int i(0); i <= baseCodim; ++i)
	{
		SerialCSRMatrix * e_idof =
			dof_alg.GetEntityInternalDofTable(static_cast<AgglomeratedTopology::EntityByCodim>(i) );
		e_idof->SetWidth( localSize );
		hypre_ParCSRMatrix * help = dof.GetEntityTrueEntity(i).ParMatmultAtA(*e_idof, entity_start);

		if(entity_trueEntity_entity)
		{
			hypre_ParCSRMatrix * tmp  = entity_trueEntity_entity;
			hypre_ParCSRMatrixAdd(help, tmp, &entity_trueEntity_entity);
			hypre_ParCSRMatrixDestroy( tmp );
			hypre_ParCSRMatrixDestroy( help );
		}
		else
			entity_trueEntity_entity = help;


		delete e_idof;
	}

	entity_trueEntity = ParUnique(entity_trueEntity_entity, trueEntity_start);
	resetHypreParVectors();
	storeSharedEntitiesIds();

	elag_assert( DebugCheck() == 0 );
}

void SharingMap::SetUp(const SerialCSRMatrix & Pi, const SharingMap & fdof_fdof, const SerialCSRMatrix & P)
{
	elag_assert( Pi.Size() == P.Width() );
	elag_assert( Pi.Width() == P.Size() );
	elag_assert( Pi.Width() == fdof_fdof.GetLocalSize() );
	elag_assert( P.Size() == fdof_fdof.GetLocalSize() );

	int localSize = Pi.Size();
	ParPartialSums_AssumedPartitionCheck(comm, localSize, entity_start);

	entity_trueEntity_entity = fdof_fdof.ParMatmultAB(Pi, P, entity_start, entity_start );
	hypre_ParCSRMatrixDeleteZeros(entity_trueEntity_entity, 1e-6);

	entity_trueEntity = ParUnique(entity_trueEntity_entity, trueEntity_start);

	resetHypreParVectors();
	storeSharedEntitiesIds();

	elag_assert( DebugCheck() == 0 );
}

void SharingMap::SetUp(SharingMap & original, SparseMatrix & e_idof)
{
	elag_assert( original.GetLocalSize() == e_idof.Size() );


	int * i_e_idof = e_idof.GetI();
	int * j_e_idof = e_idof.GetJ();
	int nnz_e_idof = e_idof.NumNonZeroElems();

	Vector ones_e_idof(nnz_e_idof);
	ones_e_idof = 1.;
	Vector tag(nnz_e_idof);
	tag = -1.;
	for(int i(0); i < e_idof.Size(); ++i)
		for(int jpos = i_e_idof[i]; jpos != i_e_idof[i+1]; ++jpos)
		{
//			elag_assert( fabs( tag(jpos) + 1. ) < 1e-9 );
			tag(jpos) = jpos - i_e_idof[i] + 1;
		}

	SparseMatrix e_idof_ones(i_e_idof, j_e_idof, ones_e_idof, e_idof.Size(), e_idof.Width() );
	SparseMatrix e_idof_tag(i_e_idof, j_e_idof, tag, e_idof.Size(), e_idof.Width() );


	// Prepare the matrix with all ones
	hypre_ParCSRMatrix * e_tE_e = original.entity_trueEntity_entity;
	hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(e_tE_e);
	hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd(e_tE_e);

	double * diag_data = hypre_CSRMatrixData(diag);
	double * diag_offd = hypre_CSRMatrixData(offd);

	int nnz = std::max(hypre_CSRMatrixNumNonzeros(diag), hypre_CSRMatrixNumNonzeros(offd));
	Vector ones(nnz);
	ones = 1.0;

	hypre_CSRMatrixData(diag) = ones.GetData();
	hypre_CSRMatrixData(offd) = ones.GetData();


	//Do the actual computation
	int nDofs = e_idof.Width();
        ParPartialSums_AssumedPartitionCheck(comm, nDofs, entity_start); 
        hypre_ParCSRMatrix * ones_tag = original.ParMatmultAtB(e_idof_ones, e_idof_tag, entity_start, entity_start);
        hypre_ParCSRMatrix * tag_ones = original.ParMatmultAtB(e_idof_tag, e_idof_ones, entity_start, entity_start);
        int ierr = hypre_ParCSRMatrixKeepEqualEntries(ones_tag, tag_ones);
        elag_assert(ierr == 0);
        entity_trueEntity_entity = ones_tag;
        ones_tag = static_cast<hypre_ParCSRMatrix *>(NULL);
        entity_trueEntity = ParUnique(entity_trueEntity_entity, trueEntity_start);

        hypre_ParCSRMatrixDestroy(tag_ones);

        resetHypreParVectors();
        storeSharedEntitiesIds();

        elag_assert( DebugCheck() == 0 );

	//RESTORE ORIGINAL POINTERS 

	e_idof_ones.LoseData();
	e_idof_tag.LoseData();
	hypre_CSRMatrixData(diag) = diag_data;
	hypre_CSRMatrixData(offd) = diag_offd;


}

SharingMap::~SharingMap()
{
	hypre_ParCSRMatrixDestroy(entity_trueEntity);
	hypre_ParCSRMatrixDestroy(entity_trueEntity_entity);
	hypre_ParVectorDestroy(x);
	hypre_ParVectorDestroy(xtrue);
}

int SharingMap::Synchronize(Array<int> & data) const
{
	elag_assert( GetLocalSize() == data.Size() );
	Array<int> trueData( GetTrueLocalSize() );

	int ierr;
	ierr = IgnoreNonLocal(data, trueData);

	elag_assert( ierr == 0 );

	ierr = Distribute(trueData, data);

	elag_assert( ierr == 0 );

	return ierr;
}

int SharingMap::Distribute(const Array<int> & trueData, Array<int> & data) const
{
	elag_assert( GetTrueLocalSize() == trueData.Size() );
	elag_assert( GetLocalSize() == data.Size() );

#if 1
	helpTrueData.SetSize(GetTrueLocalSize());
	helpData.SetSize( GetLocalSize() );

	for(int i = 0; i < trueData.Size(); ++i)
		helpTrueData(i) = static_cast<double>( trueData[i] );

	 hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(entity_trueEntity);
	 hypre_CSRMatrix * offd = hypre_ParCSRMatrixOffd(entity_trueEntity);

	 double * diag_data = hypre_CSRMatrixData(diag);
	 double * diag_offd = hypre_CSRMatrixData(offd);

	 int nnz = std::max(hypre_CSRMatrixNumNonzeros(diag), hypre_CSRMatrixNumNonzeros(offd));
	 Vector ones(nnz);
	 ones = 1.0;

	 hypre_CSRMatrixData(diag) = ones.GetData();
	 hypre_CSRMatrixData(offd) = ones.GetData();

	int ierr = Distribute(helpTrueData, helpData);

	hypre_CSRMatrixData(diag) = diag_data;
	hypre_CSRMatrixData(offd) = diag_offd;

	round(helpData, data);

#else
	int ierr = hypre_ParCSRMatrixMatvecBoolInt(1, entity_trueEntity, const_cast<int *>( trueData.GetData() ), 0, data.GetData() );
#endif
	return ierr;
}

int SharingMap::Distribute(const Vector & trueData, Vector & data) const
{

	hypre_VectorData( hypre_ParVectorLocalVector(xtrue) ) = trueData.GetData();
	hypre_VectorData( hypre_ParVectorLocalVector(x) )     = data.GetData();

	int ierr = hypre_ParCSRMatrixMatvec(1., entity_trueEntity, xtrue, 0., x);

	hypre_VectorData( hypre_ParVectorLocalVector(xtrue) ) = NULL;
	hypre_VectorData( hypre_ParVectorLocalVector(x) )     = NULL;

	return ierr;
}

int SharingMap::IgnoreNonLocal(const Array<int> & data, Array<int> & trueData) const
{
	elag_assert( GetTrueLocalSize() == trueData.Size() );
	elag_assert( GetLocalSize() == data.Size() );
#if 0
	helpTrueData.SetSize(GetTrueLocalSize());
	helpData.SetSize( GetLocalSize() );

	for(int i = 0; i < data.Size(); ++i)
		helpData(i) = static_cast<double>( data[i] );

	int ierr = IgnoreNonLocal(helpData, helpTrueData);

	for(int i = 0; i < trueData.Size(); ++i)
		trueData[i] = static_cast<int>( helpTrueData(i) + 0.5 );
#else
     hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(entity_trueEntity);

     // Check that all entries are 1.
     {
         double * v = hypre_CSRMatrixData(diag);
         int nnz = hypre_CSRMatrixNumNonzeros(diag);
         for(double * end = v+nnz; v < end; ++v)
            elag_assert(fabs( fabs(*v) - 1.) < 1.e-10 );
     }
     const int * i_diag = hypre_CSRMatrixI(diag);
     const int * j_diag = hypre_CSRMatrixJ(diag);
     int nrows = hypre_CSRMatrixNumRows(diag);
     trueData = 0;

     int val = 0;
     for(int i = 0; i < nrows; ++i)
     {
        val = data[i];
        for(int jpos = i_diag[i]; jpos < i_diag[i+1]; ++jpos)
             trueData[ j_diag[jpos] ] += val;
     }

     int ierr = 0;
#endif
	return ierr;

}

int SharingMap::IgnoreNonLocal(const Vector & data, Vector & trueData) const
{
	hypre_VectorData( hypre_ParVectorLocalVector(xtrue) ) = trueData.GetData();
	hypre_VectorData( hypre_ParVectorLocalVector(x) )     = data.GetData();

	hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag(entity_trueEntity);

	int ierr = hypre_CSRMatrixMatvecT(1., diag, hypre_ParVectorLocalVector( x ), 0., hypre_ParVectorLocalVector( xtrue ));

	hypre_VectorData( hypre_ParVectorLocalVector(xtrue) ) = NULL;
	hypre_VectorData( hypre_ParVectorLocalVector(x) )     = NULL;

	return ierr;
}

int SharingMap::Assemble(const Array<int> & data, Array<int> & trueData) const
{
	elag_assert( GetTrueLocalSize() == trueData.Size() );
	elag_assert( GetLocalSize() == data.Size() );

#if 1
	helpTrueData.SetSize(GetTrueLocalSize());
	helpData.SetSize( GetLocalSize() );

	for(int i = 0; i < data.Size(); ++i)
		helpData(i) = static_cast<double>( data[i] );

	int ierr = Assemble(helpData, helpTrueData);

	round(helpTrueData, trueData);

#else
	int ierr = hypre_ParCSRMatrixMatvecTBoolInt(1., entity_trueEntity, x, 0., xtrue);
#endif

	return ierr;

}

int SharingMap::Assemble(const Vector & data, Vector & trueData) const
{
	hypre_VectorData( hypre_ParVectorLocalVector(xtrue) ) = trueData.GetData();
	hypre_VectorData( hypre_ParVectorLocalVector(x) )     = data.GetData();

	int ierr = hypre_ParCSRMatrixMatvecT(1., entity_trueEntity, x, 0., xtrue);

	hypre_VectorData( hypre_ParVectorLocalVector(xtrue) ) = NULL;
	hypre_VectorData( hypre_ParVectorLocalVector(x) )     = NULL;

	return ierr;
}

hypre_ParCSRMatrix * SharingMap::ParMatmultAB(const SerialCSRMatrix & A, const SerialCSRMatrix & B, const Array<int> & row_starts_A, const Array<int> & col_starts_B) const
{

	elag_assert( A.Size() == row_starts_A[assumedPID+1] - row_starts_A[assumedPID]);
	elag_assert( A.Width() == GetLocalSize() );
	elag_assert( B.Size() == GetLocalSize()  );
	elag_assert( B.Width() == col_starts_B[assumedPID+1] - col_starts_B[assumedPID]);

	hypre_ParCSRMatrix * out;

	ParallelCSRMatrix Ad(comm, row_starts_A.Last(), entity_start.Last(),
						const_cast<int *>(row_starts_A.GetData()),
			            const_cast<int *>(entity_start.GetData()),
			            const_cast<SerialCSRMatrix *>(&A) );

	ParallelCSRMatrix Bd(comm, entity_start.Last(), col_starts_B.Last(),
			                   const_cast<int *>(entity_start.GetData()),
			                   const_cast<int *>(col_starts_B.GetData()),
			                   const_cast<SerialCSRMatrix *>(&B) );

	hypre_ParCSRMatrix * tmp = hypre_ParMatmul(Ad, entity_trueEntity_entity);
	hypre_MatvecCommPkgCreate(tmp);
	out = hypre_ParMatmul(tmp,Bd);
	hypre_MatvecCommPkgCreate(out);
	hypre_ParCSRMatrixDestroy(tmp);

	//hypre_BoomerAMG gives by default ownership of the column/row partitioning to the result. We don't want that :)
	hypre_ParCSRMatrixSetRowStartsOwner(out,0);
	hypre_ParCSRMatrixSetColStartsOwner(out,0);

	return out;

}

hypre_ParCSRMatrix * SharingMap::ParMatmultAtB(SerialCSRMatrix & A, SerialCSRMatrix & B, const Array<int> & col_starts_A, const Array<int> & col_starts_B) const
{
	hypre_ParCSRMatrix * out;

	ParallelCSRMatrix Ad(comm, entity_start.Last(), col_starts_A.Last(),
			                   const_cast<int *>(entity_start.GetData()),
			                   const_cast<int *>(col_starts_A.GetData()), &A);

	ParallelCSRMatrix Bd(comm, entity_start.Last(), col_starts_B.Last(),
			                   const_cast<int *>(entity_start.GetData()),
			                   const_cast<int *>(col_starts_B.GetData()), &B);

	hypre_BoomerAMGBuildCoarseOperator(Ad, entity_trueEntity_entity, Bd, &out );
	//hypre_BoomerAMG gives by default ownership of the column/row partitioning to the result. We don't want that :)
	hypre_ParCSRMatrixSetRowStartsOwner(out,0);
	hypre_ParCSRMatrixSetColStartsOwner(out,0);

	return out;
}

hypre_ParCSRMatrix * SharingMap::ParMatmultAtA(SerialCSRMatrix & A, const Array<int> & col_starts_A) const
{
	elag_assert(A.Size() == GetLocalSize() );

	hypre_ParCSRMatrix * out;

	ParallelCSRMatrix Ad(comm, entity_start.Last(), col_starts_A.Last(),
			                   const_cast<int *>(entity_start.GetData()),
			                   const_cast<int *>(col_starts_A.GetData()), &A);

	hypre_BoomerAMGBuildCoarseOperator(Ad, entity_trueEntity_entity, Ad, &out );
	//hypre_BoomerAMG gives by default ownership of the column/row partitioning to the result. We don't want that :)
	hypre_ParCSRMatrixSetRowStartsOwner(out,0);
	hypre_ParCSRMatrixSetColStartsOwner(out,0);

	return out;
}

int SharingMap::IsShared(int localId) const
{
	hypre_CSRMatrix * diag_entity_trueEntity = hypre_ParCSRMatrixDiag(entity_trueEntity);
	hypre_CSRMatrix * offd_entity_trueEntity_entity = hypre_ParCSRMatrixOffd(entity_trueEntity_entity);

	HYPRE_Int * i_diag_entity_trueEntity = hypre_CSRMatrixI(diag_entity_trueEntity);
	HYPRE_Int * i_offd_entity_trueEntity_entity = hypre_CSRMatrixI(offd_entity_trueEntity_entity);

	int is_shared = 0;

	if( i_offd_entity_trueEntity_entity[localId+1] - i_offd_entity_trueEntity_entity[localId])
		is_shared = (i_diag_entity_trueEntity[localId+1] - i_diag_entity_trueEntity[localId]) ? 1 : -1;

	return is_shared;
}

int SharingMap::GetNumberShared() const
{
	return sharedEntityIds.Size();
}

int SharingMap::GetNumberSharedOwned() const
{
	return nOwnedSharedEntities;
}

int SharingMap::GetNumberSharedNotOwned() const
{
	return sharedEntityIds.Size() - nOwnedSharedEntities;
}

const Array<int> & SharingMap::SharedEntitiesId() const
{
	return sharedEntityIds;
}
void SharingMap::ViewOwnedSharedEntitiesId(Array<int> & ownedSharedId)
{
	ownedSharedId.MakeRef(sharedEntityIds.GetData(), nOwnedSharedEntities);
}
void SharingMap::ViewNotOwnedSharedEntitiesId(Array<int> & notOwnedSharedId)
{
	notOwnedSharedId.MakeRef(sharedEntityIds.GetData() + nOwnedSharedEntities, sharedEntityIds.Size() - nOwnedSharedEntities);
}

SerialCSRMatrix * Distribute(SharingMap & range, ParallelCSRMatrix & A, SharingMap & domain)
{
	elag_error_msg(NOT_IMPLEMENTED_YET, "Not Implemented Yet");
	return (SerialCSRMatrix *)(NULL);
}

ParallelCSRMatrix * IgnoreNonLocalRange(const SharingMap & range, SerialCSRMatrix & A,
													 const SharingMap & domain)
{
	elag_assert(range.comm == domain.comm);
	elag_assert( A.Size() == range.GetLocalSize() );
	elag_assert( A.Width() == domain.GetLocalSize() );

	ParallelCSRMatrix Bdiag(domain.comm, range.GetGlobalSize(), domain.GetGlobalSize(),
			const_cast<int *>(range.entity_start.GetData()),
			const_cast<int *>(domain.entity_start.GetData()), &A);
	hypre_ParCSRMatrix * out;

	hypre_ParCSRMatrixAssembly( range.entity_trueEntity, Bdiag, domain.entity_trueEntity, 1, &out);

	elag_assert(hypre_ParCSRMatrixOwnsRowStarts(out) == 0 );
	elag_assert(hypre_ParCSRMatrixOwnsColStarts(out) == 0 );

	return new ParallelCSRMatrix(out);
}

ParallelCSRMatrix * Assemble(const SharingMap & range, SerialCSRMatrix & A, const SharingMap & domain)
{
	elag_assert(range.comm == domain.comm);
	elag_assert(A.Size() == range.GetLocalSize() );
	elag_assert(A.Width() == domain.GetLocalSize() );

	ParallelCSRMatrix Bdiag(domain.comm, range.GetGlobalSize(), domain.GetGlobalSize(),
			const_cast<int *>(range.entity_start.GetData()),
			const_cast<int *>(domain.entity_start.GetData()), &A);
	hypre_ParCSRMatrix * out;

//	hypre_ParCSRMatrixAssembly( range.entity_trueEntity, Bdiag, domain.entity_trueEntity, 0, &out);
	hypre_BoomerAMGBuildCoarseOperator(range.entity_trueEntity,Bdiag,domain.entity_trueEntity,&out);
	hypre_ParCSRMatrixSetNumNonzeros(out);
    /* Warning: hypre_BoomerAMGBuildCoarseOperator steals the col_starts
       from P (even if it does not own them)! */
	hypre_ParCSRMatrixSetRowStartsOwner(out,0);
	hypre_ParCSRMatrixSetColStartsOwner(out,0);

	return new ParallelCSRMatrix(out);

}

SerialCSRMatrix * AssembleNonLocal(SharingMap & range, SerialCSRMatrix & A, SharingMap & domain)
{
	elag_assert(range.comm = domain.comm);

	// NOTE: We need to handle both the case in which range and domain are the same or different objects.
	// To avoid Segmentation Faults it is important that Step 1-2-3 are performed in such order.

	// (1) Extract the diag of the matrix.
	hypre_CSRMatrix * dom_diag = hypre_ParCSRMatrixDiag(domain.entity_trueEntity_entity);
	hypre_CSRMatrix * ran_diag = hypre_ParCSRMatrixDiag(range.entity_trueEntity_entity);

	// (2) Create a zeros block to replace the diag block.
	hypre_CSRMatrix * dom_dzeros = hypre_ZerosCSRMatrix(domain.GetLocalSize(), domain.GetLocalSize());
	hypre_CSRMatrix * ran_dzeros = hypre_ZerosCSRMatrix(range.GetLocalSize(), range.GetLocalSize());

	// (3) Replace the diag blocks and set to one the entries in the offd
	hypre_ParCSRMatrixDiag(domain.entity_trueEntity_entity) = dom_dzeros;
	hypre_ParCSRMatrixDiag(range.entity_trueEntity_entity)  = ran_dzeros;


	ParallelCSRMatrix Bdiag(domain.comm, range.GetGlobalSize(), domain.GetGlobalSize(),
									range.entity_start.GetData(), domain.entity_start.GetData(), &A);

	hypre_ParCSRMatrix * offd_A, *offd_A_offd;

	offd_A = hypre_ParMatmul(range.entity_trueEntity_entity, Bdiag);
	offd_A_offd = hypre_ParMatmul(offd_A, domain.entity_trueEntity_entity);

	hypre_CSRMatrix * res = hypre_ParCSRMatrixDiag(offd_A_offd);
	SerialCSRMatrix tmp(hypre_CSRMatrixI(res), hypre_CSRMatrixJ(res), hypre_CSRMatrixData(res),
			              hypre_CSRMatrixNumRows(res), hypre_CSRMatrixNumCols(res));

	SerialCSRMatrix * out = Add(A, tmp);
	tmp.LoseData();

	hypre_ParCSRMatrixDestroy(offd_A);
	hypre_ParCSRMatrixDestroy(offd_A_offd);

	hypre_ParCSRMatrixDiag(domain.entity_trueEntity_entity) = dom_diag;
	hypre_ParCSRMatrixDiag(range.entity_trueEntity_entity) = ran_diag;

	hypre_CSRMatrixDestroy(dom_dzeros);
	hypre_CSRMatrixDestroy(ran_dzeros);


	return out;

}

void SharingMap::round(const Vector & d, Array<int> & a) const
{
	elag_assert(d.Size() == a.Size());
	const double * dd = d.GetData();
	int * aa = a.GetData();

	int n = d.Size();

	for(int i = 0; i < n; ++i)
		aa[i] = static_cast<int>( ::round( dd[i] ) );
}
