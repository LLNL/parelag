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

#ifndef _Topology_HPP_
#define _Topology_HPP_

//! @class
/*!
 * @brief A simple structure to describe an (agglomerated) topology
 *
 * The topology of a <i>n</i>-dimensional mesh is described by <i>n</i> oriented tables (boundary operators).
 * For example in 3-D we have
 * <ul>
 * <li> <tt> B[0]</tt> is the <tt> element_facet</tt> table
 * <li> <tt> B[1]</tt> is the <tt> facet_ridge</tt> table
 * <li> <tt> B[2]</tt> is the <tt> ridge_peak</tt> table
 * </ul>
 *
 *
 *
 * The boundary operators B define a complex on the mesh entities that is dual to the finite element complex
 *
 *\verbatim
 * Continuous Lagragian --Grad--> Nedelec --Curl--> Raviart Thomas --Div--> Piecewise Disc.
 *       vertices       <--B_2--  edges   <--B_1---   faces        <--B_0--     elements
 *\endverbatim
 *
 * The boundary operators B_i can be computed from by using the DiscreteInterpolators DivergenceInterpolator, CurlInterpolator, GradientInterpolator in MFEM,
 * and by normalizing to +-1 the entries of those matrices.
 *
 * The following relationship should be verified:
 * B[0]*B[1] = 0
 * B[1]*B[2] = 0
 *
 * We also introduce weights for element, facets, ridgets. Such weights are used to obtain better shaped agglomerates in a multilevel framework.
 *
 */

class AgglomeratedTopology
{
public:
	typedef Vector            TopologyArray;
	typedef ParallelCSRMatrix TopologyParTable;
	typedef double            TopologyDataType;

	//! Entities by Codimension
	enum EntityByCodim{ ELEMENT = 0, FACET = 1, RIDGE = 2, PEAK = 3 };
	AgglomeratedTopology(MPI_Comm comm, int ncodim_);
	AgglomeratedTopology(ParMesh * pmesh, int ncodim_);
	~AgglomeratedTopology();

	inline TopologyTable & B(int i)
	{
		elag_assert(0 <= i && ncodim > i);
		elag_assert(B_[i] != NULL );

		return *(B_[i]);
	}

	inline TopologyTable & AEntityEntity(int i)
	{
		elag_assert(0 <= i && ncodim >= i);
		elag_assert(AEntity_entity[i] != NULL );

		return *(AEntity_entity[i]);
	}

	inline Array<int> & Partitioning()
	{
		return partitioning;
	}

	inline SharingMap & EntityTrueEntity(int i)
	{
		elag_assert(0 <= i && ncodim >= i);
		elag_assert(entityTrueEntity[i] != NULL );

		return *(entityTrueEntity[i]);
	}
	inline const SharingMap & EntityTrueEntity(int i) const
	{
		elag_assert(0 <= i && ncodim >= i);
		elag_assert(entityTrueEntity[i] != NULL );

		return *(entityTrueEntity[i]);
	}

	inline int GetNumberSharedEntities(int i) const
	{
		elag_assert(0 <= i && ncodim >= i);
		elag_assert(entityTrueEntity[i] != NULL );

		return entityTrueEntity[i]->GetNumberShared();
	}

	inline void GetOwnedSharedEntities(int i, Array<int> & sharedEntitiesId)
	{
		elag_assert(0 <= i && ncodim >= i);
		elag_assert(entityTrueEntity[i] != NULL );

		return entityTrueEntity[i]->ViewOwnedSharedEntitiesId(sharedEntitiesId);
	}

	inline void GetNotOwnedSharedEntities(int i, Array<int> & sharedEntitiesId)
	{
		elag_assert(0 <= i && ncodim >= i);
		elag_assert(entityTrueEntity[i] != NULL );

		return entityTrueEntity[i]->ViewNotOwnedSharedEntitiesId(sharedEntitiesId);
	}

	inline TopologyArray & Weight(int i)
	{
		elag_assert(0 <= i && ncodim >= i);
		elag_assert(w[i] != NULL );
		return *(w[i]);
	}

	TopologyParTable & TrueB(int i);
	TopologyArray * TrueWeight(int i);

	inline Array<int> & ElementAttribute()
	{
		return element_attribute;
	}

	inline TopologyTable & FacetBdrAttribute()
	{
		return *facet_bdrAttribute;
	}
	inline const TopologyTable & FacetBdrAttribute() const
	{
		return *facet_bdrAttribute;
	}

	inline int Dimensions() const
	{
		return ndim;
	}
	inline int Codimensions() const
	{
		return ncodim;
	}

	MPI_Comm GetComm() const { return comm;}

	int GetNumberLocalEntities(EntityByCodim id) const        { return (id <= ncodim) ? entityTrueEntity[id]->GetLocalSize() : 0; }
	int GetNumberGlobalTrueEntities( EntityByCodim id ) const { return (id <= ncodim) ? entityTrueEntity[id]->GetTrueGlobalSize() : 0; }

	void BuildConnectivity();
	const BooleanMatrix & GetConnectivity(int range, int domain) const;
	void GetBoundaryOfEntity(EntityByCodim type, EntityByCodim type_bdr, int ientity, Array<int> & bdr_entity) const;

	//! Computes the LocalElementElementTable that can be used to generate partitions with metis.
	// Note that the diagonal of this matrix will not be non-zeros which metis-wrapper will have
	// to drop before calling the partitioner.
	// Offdiagonal entries are negative if orientation is used.
	SerialCSRMatrix    * LocalElementElementTable();
	//! Computes the GlobalElementElementTable that can be used to generate partitions with parmetis.
	// Note that the diagonal of this matrix will not be non-zeros which parmetis-wrapper will have
	// to drop before calling the partitioner.
	// Offdiagonal entries are negative if orientation is used.
	ParallelCSRMatrix * GlobalElementElementTable();

	//! Coarsering routine to generate an agglomerated topology from the current topology.
	/**
	 * We will use this routine when agglomerates are local to the processor.
	 *
	 * -- partitioning is the array that tells which element belongs to each partition.
	 *    Partitions may be empty or disconnected.
	 * -- checkTopology: if 1 we will check that each agglomerated entity has valid topology.
	 *                   if 0 we will only check for connected components.
	 * -- preserveMaterialInterfaces: if 1 we will make sure that partitions do not span across material interfaces.
	 *                                if 0 we will discard material interfaces for the coarse topology.
	 */
	AgglomeratedTopology * CoarsenLocalPartitioning(Array<int> & partitioning, int checkTopology, int preserveMaterialInterfaces);

	//! Coarsering routine to generate an agglomerated topology from the current topology.
	/**
	 * We will use this routine when agglomerates may span across processors.
	 *
	 */
	AgglomeratedTopology * CoarsenGlobalPartitioning(Array<int> & partitioning, int checkTopology, int preserveMaterialInterfaces);


	AgglomeratedTopology * UniformRefinement();

	void ShowMe(std::ostream & os);

	AgglomeratedTopology * FinestTopology();

	AgglomeratedTopology * finerTopology;
	AgglomeratedTopology * coarserTopology;

protected:

	void generateTopology(ParMesh * pmesh);

	void initializeWeights(Mesh * mesh);
	void generateFacetBdrAttributeTable(Mesh * mesh);

	TopologyTable * generate_AF_f_ForUniformRefinement();
	TopologyTable * generate_AR_r_ForUniformRefinement();
	TopologyTable * generate_AP_p_ForUniformRefinement();

	MPI_Comm comm;
	//! Number of space Dimensions
	int ndim;
	//! Number of needed codimensions (ncodim <= ndim)
	int ncodim;
	//! Pointer to the parallel mesh. Only valid for the finest grid topology.
	ParMesh * pmesh;
	/**
	 * Topological Tables that describe the relationship entity_entityBoundary:
	 * B[0] = element_facet
	 * B[1] = facet_ridge
	 * B[2] = ridge_peak
	 *
	 * The size of B is at most ncodim.
	 */
	Array<TopologyTable *> B_;
	/**
	 * Build other connectivities tables as required.
	 * Initially allocated with 0 size.
	 *
	 * First row:
	 * conn(ELEMENT, FACET) = B(0);
	 * conn(ELEMENT, RIDGE) = bool( B(0) ) * bool( B(1) );
	 * conn(ELEMENT, PEAK)  = conn(ELEMENT, RIDGE) * bool( B(2) );
	 *
	 * Second row:
	 * conn(FACET, RIDGE) = B(1);
	 * conn(FACET, PEAK)  = bool( B(1) ) * bool( B(2) );
	 *
	 * Third row:
	 * conn(RIDGE, PEAK) = B(2);
	 */
	Array2D<BooleanMatrix *> conn;
	/**
	 * Weights for each entity:
	 * w[0][i] = weight of element i.
	 * w[1][i] = weight of facet i
	 * w[2][i] = weight of ridge i
	 *
	 * The size of w is at most ncodim.
	 */
	Array<TopologyArray *> w;

	//! attribute (i.e. material subdomains) for each elements. It has size 0, if no attributes.
	Array<int> element_attribute;
	//! attribute (i.e. bc labels) for each facets. Internal facets are marked with -1. It has size 0, if no attributes.
	TopologyTable * facet_bdrAttribute;

	//! SharingMap for each entity. Size ncodim+1.
	Array<SharingMap *> entityTrueEntity;

	//@name Agglomeration information
	//@{
	/** If globalAgglomeration == 1, then agglomerates may span across processors.
	 *  If globalAgglomeration == 0, then agglomerates will be local to the processor.
	 */
	int globalAgglomeration;
	/**
	 * Partitioning vector for each elements it define to which partition it belongs to.
	 */
	Array<int> partitioning;
	/**
	 * AgglomeratedEntity to entity table.
	 * This table is filled only if globalAgglomeration == 0.
	 */
	Array<TopologyTable *> AEntity_entity;
	/**
	 * AgglomeratedTrueEntity to trueEntity table.
	 * This table is filled only if globalAgglomeration == 1.
	 */
	Array<TopologyParTable *> ATrueEntity_trueEntity;

	/**
	 * @class ExtraTopologyTables
	 * This class is a container for temporary tables that are used during the coarsering.
	 * After the coarse topology is created one can safely eliminate this tables.
	 */
	class ExtraTopologyTables
	{
	public:
		ExtraTopologyTables(int ncodim);
		void Clean();
		~ExtraTopologyTables();

		SerialCSRMatrix * elem_elem_local;
		int owns_elem_elem_local;
		Array<TopologyTable *> Bt_;
		Array<int> owns_Bt;
		ParallelCSRMatrix * elem_elem_global;
		int owns_elem_elem_global;
		Array<AgglomeratedTopology::TopologyParTable *> trueB_;
		Array<int> owns_trueB;
		Array<AgglomeratedTopology::TopologyParTable *> trueBt_;
		Array<int> owns_trueBt;
	};
	ExtraTopologyTables workspace;
	//@}
};

void ShowTopologyAgglomeratedElements(AgglomeratedTopology * topo, ParMesh * mesh);
void ShowTopologyAgglomeratedFacets(AgglomeratedTopology * topo, ParMesh * mesh);
void ShowTopologyBdrFacets(AgglomeratedTopology * topo, ParMesh * mesh);
void ShowAgglomeratedTopology3D(AgglomeratedTopology * topo, ParMesh * mesh);

#endif
