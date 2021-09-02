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

#ifndef _Topology_HPP_
#define _Topology_HPP_

#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <mfem.hpp>

#include "structures/BooleanMatrix.hpp"
#include "structures/SharingMap.hpp"
#include "topology/TopologyTable.hpp"
#include "utilities/elagError.hpp"

namespace parelag
{
//! @class
/*!
 * @brief A structure to describe an (agglomerated) topology
 *
 * The topology of a <i>n</i>-dimensional mesh is described by
 * <i>n</i> oriented tables (boundary operators).
 * For example in 3-D we have
 * <ul>
 * <li> <tt> B[0]</tt> is the <tt> element_facet</tt> table
 * <li> <tt> B[1]</tt> is the <tt> facet_ridge</tt> table
 * <li> <tt> B[2]</tt> is the <tt> ridge_peak</tt> table
 * </ul>
 *
 *
 * The boundary operators B define a complex on the mesh entities that
 * is dual to the finite element complex
 *
 *\verbatim
 * Continuous Lagragian --Grad--> Nedelec --Curl--> Raviart Thomas --Div--> Piecewise Disc.
 *       vertices       --B_2-->  edges   --B_1--->   faces        --B_0-->     elements
 *\endverbatim
 *
 * The boundary operators B_i can be computed from by using the
 * DiscreteInterpolators, DivergenceInterpolator, CurlInterpolator,
 * GradientInterpolator in MFEM, and by normalizing to +-1 the entries
 * of those matrices.
 *
 * The following relationship should be verified:
 * B[0]*B[1] = 0
 * B[1]*B[2] = 0
 *
 * We also introduce weights for element, facets, ridgets. Such
 * weights are used to obtain better shaped agglomerates in a
 * multilevel framework.
 *
 */
//template <typename LocalIndexT = int, typename GlobalIndexT = LocalIndex>
class AgglomeratedTopology :
        public std::enable_shared_from_this<AgglomeratedTopology>
{
public:
    using array_t = mfem::Vector;
    using par_table_t = mfem::HypreParMatrix;

    using local_index_type = int;
    using global_index_type = int;

    using data_type = double;

    //! Entities by Codimension
    enum Entity {ELEMENT = 0, FACET = 1, RIDGE = 2, PEAK = 3};

public:
    /// Constructors and Destructor
    ///@{

    /// Construct a topology from an MPI communicator
    ///
    /// This constructor is useful for constructing coarse topologies
    /// for which no mesh is explicitly available.
    AgglomeratedTopology(MPI_Comm comm,int ncodim);

    /// Construct a topology from a ParMesh object
    ///
    /// This constructor is useful for constructing the fine-grid
    /// topology from a valid MFEM distributed mesh.
    AgglomeratedTopology(
        const std::shared_ptr<mfem::ParMesh>& pmesh,int ncodim);

    /// Destructor
    ~AgglomeratedTopology() = default;
    ///@}
    /// Non-owning attribute accessors
    ///@{

    /// Get the boundary operator B_[codim] mapping codimension "codim"
    /// to codimension "codim+1".
    TopologyTable & GetB(int codim)
    {
        elag_assert(0 <= codim && nCodim_ > codim);
        elag_assert(B_[codim]);

        return *(B_[codim]);
    }

    /// Get the boundary operator B[codim] mapping codimension "codim"
    /// to codimension "codim+1". (const version)
    const TopologyTable & GetB(int codim) const
    {
        elag_assert(0 <= codim && nCodim_ > codim);
        elag_assert(B_[codim]);

        return *(B_[codim]);
    }

    /// Get the (agglomerated entity)_(entity) table
    TopologyTable & AEntityEntity(int codim)
    {
        elag_assert(0 <= codim && nCodim_ >= codim);
        elag_assert(AEntity_entity.size() > 0);
        elag_assert(AEntity_entity[codim]);

        return *(AEntity_entity[codim]);
    }

    /// Get the (agglomerated entity)_(entity) table (const version)
    const TopologyTable & AEntityEntity(int codim) const
    {
        elag_assert(0 <= codim && nCodim_ >= codim);
        elag_assert(AEntity_entity.size() > 0);
        elag_assert(AEntity_entity[codim]);

        return *(AEntity_entity[codim]);
    }

    /// Get the (agglomerated entity)_(entity) table
    par_table_t & ATrueEntityTrueEntity(int codim)
    {
        elag_assert(0 <= codim && nCodim_ >= codim);
        elag_assert(ATrueEntity_trueEntity.size() > 0);
        elag_assert(ATrueEntity_trueEntity[codim]);

        return *(ATrueEntity_trueEntity[codim]);
    }

    /// Get the (agglomerated entity)_(entity) table (const version)
    const par_table_t & ATrueEntityTrueEntity(int codim) const
    {
        elag_assert(0 <= codim && nCodim_ >= codim);
        elag_assert(ATrueEntity_trueEntity.size() > 0);
        elag_assert(ATrueEntity_trueEntity[codim]);

        return *(ATrueEntity_trueEntity[codim]);
    }

    /// Get a reference to the partitioning
    mfem::Array<int>& Partitioning() noexcept
    {
        mfem_partitioning.MakeRef(Partition_.data(),Partition_.size());
        return mfem_partitioning;
    }

    /// Get a reference to the partitioning (const version)
    //const mfem::Array<int>& Partitioning() const noexcept
    //{ return partitioning; }

    /// Get the Entity_TrueEntity table for a given codimension
    SharingMap & EntityTrueEntity(int codim)
    {
        elag_assert(0 <= codim && nCodim_ >= codim);
        elag_assert(entityTrueEntity[codim]);

        return *(entityTrueEntity[codim]);
    }

    /// Get the Entity_TrueEntity table for a given codimension (const
    /// version)
    const SharingMap & EntityTrueEntity(int codim) const
    {
        elag_assert(0 <= codim && nCodim_ >= codim);
        elag_assert(entityTrueEntity[codim]);

        return *(entityTrueEntity[codim]);
    }

    /// Get a view of the shared entites in the given codimension that
    /// are owned
    void GetOwnedSharedEntities(
        int codim,mfem::Array<int> & sharedEntitiesId) const
    {
        elag_assert(0 <= codim && nCodim_ >= codim);
        elag_assert(entityTrueEntity[codim]);

        return entityTrueEntity[codim]->
            ViewOwnedSharedEntitiesId(sharedEntitiesId);
    }

    /// Get a view of the shared entities in the given codimension
    /// that are not owned
    void GetNotOwnedSharedEntities(
        int codim,mfem::Array<int> & sharedEntitiesId) const
    {
        elag_assert(0 <= codim && nCodim_ >= codim);
        elag_assert(entityTrueEntity[codim]);

        return entityTrueEntity[codim]->
            ViewNotOwnedSharedEntitiesId(sharedEntitiesId);
    }

    /// Return the weight array for a given codimension
    array_t & Weight(int codim)
    {
        elag_assert(0 <= codim && nCodim_ >= codim);
        elag_assert( Weights_[codim] );
        return *(Weights_[codim]);
    }

    /// Return the weight array for a given codimension (const version
    const array_t & Weight(int codim) const
    {
        elag_assert(0 <= codim && nCodim_ >= codim);
        elag_assert( Weights_[codim] );
        return *(Weights_[codim]);
    }

    /// Return the parallel version of B[codim]
    par_table_t & TrueB(int codim) const;

    /// Return the weight vector for the TrueEntity
    std::unique_ptr<array_t> TrueWeight(int codim) const;

    /// Return a reference to the array of element attributes
    mfem::Array<int> & ElementAttribute() noexcept { return element_attribute; }

    /// Return a reference tp the array of element attributes (const version)
    const mfem::Array<int>& ElementAttribute() const noexcept
    {return element_attribute;}

    /// Return a reference to the Facet boundary attribute table
    TopologyTable & FacetBdrAttribute() noexcept { return *facet_bdrAttribute; }

    /// Return a reference to the Facet boundary attribute table (const version)
    const TopologyTable & FacetBdrAttribute() const noexcept
    { return *facet_bdrAttribute; }

    //@}
    /// \name Information queries
    //@{

    /// Dimension of the underlying topology
    int Dimensions() const noexcept { return nDim_; }

    /// Number of codimensions
    int Codimensions() const noexcept { return nCodim_; }

    /// Number of shared entities associated with a given codimension
    int GetNumberSharedEntities(int codim) const
    {
        elag_assert(0 <= codim && nCodim_ >= codim);
        elag_assert(entityTrueEntity[codim]);

        return entityTrueEntity[codim]->GetNumberShared();
    }

    /// Get the underlying communicator object
    MPI_Comm GetComm() const noexcept
    {
        return Comm_;
    }

    /// Number of local entities of given type
    int GetNumberLocalEntities(Entity id) const noexcept
    {
        return (id <= nCodim_) ? entityTrueEntity[id]->GetLocalSize() : 0;
    }

    /// Number of global entities of given type
    int GetNumberGlobalTrueEntities(Entity id) const noexcept
    {
        return (id <= nCodim_) ? entityTrueEntity[id]->GetTrueGlobalSize() : 0;
    }
    //@}
    /// \name Unsorted functions
    //@{

    /// Builds all of the relevant connectivity tables
    void BuildConnectivity();

    /// Get the connectivity between two codimensions
    const BooleanMatrix & GetConnectivity(
        int range_codim,int domain_codim) const;

    /// Get the boundary of a given instance of a given entity in
    /// terms of another entity (e.g. the boundary of "ELEMENT" with
    /// index "ientity" in terms of "FACETS"
    void GetBoundaryOfEntity(
        Entity type,Entity type_bdr,
        int ientity,mfem::Array<int>& bdr_entity) const;

    /// Computes the LocalElementElementTable that can be used to
    /// generate partitions with metis.
    ///
    /// Note that the diagonal of this matrix will not be non-zeros
    /// which metis-wrapper will have to drop before calling the
    /// partitioner.
    ///
    /// Off-diagonal entries are negative if orientation is used.
    mfem::SparseMatrix * LocalElementElementTable();

    /// Computes the GlobalElementElementTable that can be used to
    /// generate partitions with parmetis.
    ///
    /// Note that the diagonal of this matrix will not be non-zeros
    /// which parmetis-wrapper will have to drop before calling the
    /// partitioner.
    ///
    /// Off-diagonal entries are negative if orientation is used.
    mfem::HypreParMatrix * GlobalElementElementTable();

    /// Coarsening routine to generate an agglomerated topology from
    /// the current topology.
    ///
    /// We will use this routine when agglomerates are local to the
    /// processor.
    ///
    /// -- partitioning is the array that tells which element belongs
    ///    to each partition.
    ///
    ///    Partitions may be empty or disconnected.
    ///
    /// -- checkTopology: if 'true', we check that each agglomerated
    ///                   entity has valid topology; if 'false', we
    ///                   only check for connected components and
    ///                   empty partitions.
    ///
    /// -- preserveMaterialInterfaces: if 'true', we will make sure
    ///                                that partitions do not span
    ///                                across material interfaces; if
    ///                                'false', we will discard material
    ///                                interfaces for the coarse
    ///                                topology.
    /// -- coarsefaces_algo: 0 for old variant, 2 for book's algorithm
    ///                      used for AFs (less memory consumption)
    std::shared_ptr<AgglomeratedTopology> CoarsenLocalPartitioning(
        mfem::Array<int> & partitioning,
        bool checkTopology,
        bool preserveMaterialInterfaces, int coarsefaces_algo = 0);

    /// Coarsening routine to generate an agglomerated topology from
    /// the current topology.
    ///
    /// We will use this routine when agglomerates may span across
    /// processors.
    ///
    /// \warning NOT IMPLEMENTED
    std::shared_ptr<AgglomeratedTopology> CoarsenGlobalPartitioning(
        mfem::Array<int> & partitioning,
        bool checkTopology,
        bool preserveMaterialInterfaces);

    /// Split agglomerates that are deemed "bad" into agglomerates
    /// that are... "not bad"? (Hopefully)
    void DeAgglomerateBadAgglomeratedEntities(
        mfem::Array<int> & isbad, int icodim);

    /// Given a coarse face, this routine returns the agglomerated
    /// elements sharing that coarse face
    void GetFacetNeighbors(int iAF, int & el1, int & el2);

    std::shared_ptr<AgglomeratedTopology> UniformRefinement(
        mfem::Array<mfem::FiniteElementCollection *> & fecs,
        std::vector<std::unique_ptr<mfem::SparseMatrix>>& Pg);

    std::shared_ptr<AgglomeratedTopology> UniformRefinement();

    void ShowMe(std::ostream & os);

    AgglomeratedTopology * FinestTopology();

    std::shared_ptr<AgglomeratedTopology> FinerTopology() const noexcept
    {
        return FinerTopology_.lock();
    }

    std::shared_ptr<AgglomeratedTopology> CoarserTopology() const noexcept
    {
        return CoarserTopology_.lock();
    }

    bool PerformGlobalAgglomeration() const { return globalAgglomeration; }

protected:

    std::weak_ptr<AgglomeratedTopology> FinerTopology_;
    std::weak_ptr<AgglomeratedTopology> CoarserTopology_;

    void generateTopology(mfem::ParMesh& pmesh);
    void initializeWeights(const mfem::Mesh& mesh);
    void generateFacetBdrAttributeTable(const mfem::Mesh& mesh);

    std::unique_ptr<TopologyTable> generate_AF_f_ForUniformRefinement();
    std::unique_ptr<TopologyTable> generate_AR_r_ForUniformRefinement();
    std::unique_ptr<TopologyTable> generate_AP_p_ForUniformRefinement();

    /// Implements a different algorithm for generating coarse facets.
    /// The algorithm is taken from the book by P. Vassilevski.
    /// requires that coarser topology (for which AFs are to constructed) already
    /// has entity_trueEntity, and a given AE_fc relation table
    std::unique_ptr<TopologyTable> ComputeCoarseFacets(std::shared_ptr<AgglomeratedTopology> CoarseTopology,
            TopologyTable &AE_fc);
    //This function will check the topology of codimension icodim and fix the AEntity_entity table (and its transpose fc_AF) if needed.
    void CheckHFacetsTopology(int icodim, std::unique_ptr<TopologyTable> & fc_AF);

    /// MPI communicator
    MPI_Comm Comm_;

    /// Number of space Dimensions
    int nDim_;

    /// Number of needed codimensions (ncodim <= ndim)
    int nCodim_;

    /// Pointer to the parallel mesh. Only valid for the finest grid topology.
    std::shared_ptr<mfem::ParMesh> pMesh_;

    /// Topological Tables that describe the relationship entity_entityBoundary:
    /// B[0] = element_facet
    /// B[1] = facet_ridge
    /// B[2] = ridge_peak
    ///
    /// The size of B_ is at most ncodim.
    std::vector<std::unique_ptr<TopologyTable>> B_;
    //std::vector<TopologyTable> B_;

    /// Build other connectivities tables as required.
    /// Initially allocated with 0 size.
    ///
    /// First row:
    /// conn(ELEMENT, FACET) = B(0);
    /// conn(ELEMENT, RIDGE) = bool( B(0) )/// bool( B(1) );
    /// conn(ELEMENT, PEAK)  = conn(ELEMENT, RIDGE)/// bool( B(2) );
    ///
    /// Second row:
    /// conn(FACET, RIDGE) = B(1);
    /// conn(FACET, PEAK)  = bool( B(1) )/// bool( B(2) );
    ///
    /// Third row:
    /// conn(RIDGE, PEAK) = B(2);
    ///
    /// Now we unroll this into a 1D array row-wise. Thus,
    ///
    /// conn(ELEMENT,FACET) --> Conn_[0];
    /// conn(ELEMENT,RIDGE) --> Conn_[1];
    /// conn(ELEMENT,PEAK)  --> Conn_[2];
    /// conn(FACET,RIDGE)   --> Conn_[3];
    /// conn(FACET,PEAK)    --> Conn_[4];
    /// conn(RIDGE,PEAK)    --> Conn_[5];
    ///
    /// To access conn(i,j), one needs to access
    ///
    /// Conn_[ i*(5-i)/2 + j - 1 ]
    ///
    /// To be extra careful, one might assert 0 <= i < j <= nCodim_
    std::array<std::unique_ptr<BooleanMatrix>,6> Conn_;

    ///If used, this stores for each row (corresponding to a face),
    ///the agglomerated elements sharing that face
    ///
    ///Should be extended to store for facet_element(:,2) -1 if
    ///boundary and -2 if processor boundary
    mfem::Array2D<int> facet_element;

    /// Weights for each entity:
    /// w[0][i] = weight of element i.
    /// w[1][i] = weight of facet i
    /// w[2][i] = weight of ridge i
    ///
    /// The size of w is at most nCodim_.
    std::vector<std::unique_ptr<array_t>> Weights_;

    /// attribute (i.e. material subdomains) for each elements. It has
    /// size 0, if no attributes.
    mfem::Array<int> element_attribute;

    /// attribute (i.e. bc labels) for each facets. Internal facets
    /// are marked with -1. It has size 0, if no attributes.
    std::unique_ptr<TopologyTable> facet_bdrAttribute;

    /// SharingMap for each entity. Size nCodim_+1.
    std::vector<std::unique_ptr<SharingMap>> entityTrueEntity;

    /// \name Agglomeration information
    ///@{
    ///
    /// If globalAgglomeration == true, then agglomerates may span across
    /// processors. --> NOT SUPPORTED (TODO: trying to support it)
    ///
    /// If globalAgglomeration == false, then agglomerates will be local to
    /// the processor. --> ONLY REAL OPTION
    ///
    bool globalAgglomeration = false;

    /// Partitioning vector. For each element, it defines to which
    /// partition it belongs.
    mfem::Array<int> mfem_partitioning;
    std::vector<int> Partition_;

    /// AgglomeratedEntity to entity table.
    /// This table is filled only if globalAgglomeration == false.
    std::vector<std::unique_ptr<TopologyTable>> AEntity_entity;

    /// AgglomeratedTrueEntity to trueEntity table.
    /// This table is filled only if globalAgglomeration == true.
    std::vector<std::unique_ptr<par_table_t>> ATrueEntity_trueEntity;

    /// \class ExtraTopologyTables
    /// \brief A container for temporary tables
    ///
    /// Used during coarsening.  After the coarse topology is created,
    /// one can safely eliminate these tables.
    // FIXME (trb 04/13/2016): I don't know that this is necessary...
    mutable struct ExtraTopologyTables
    {
        using par_table_t = AgglomeratedTopology::par_table_t;

        ExtraTopologyTables(int ncodim)
            : elem_elem_local{nullptr}, Bt_(ncodim),
              elem_elem_global{nullptr}, trueB_(ncodim)
        {}
        ~ExtraTopologyTables() = default;

        std::unique_ptr<mfem::SparseMatrix> elem_elem_local;
        std::vector<std::unique_ptr<TopologyTable>> Bt_;
        std::unique_ptr<mfem::HypreParMatrix> elem_elem_global;

        // trb (12/10/15): It didn't appear that "owns_trueB[i]" could
        // ever be false. Hence, the array is no more and these are
        // unique_ptrs.
        std::vector<std::unique_ptr<par_table_t>> trueB_;

        // trb (12/10/15): Also, trueBt_ was never ever used in all of
        // parelag; likewise with owns_trueBt, so they are gone now.
    }  workspace;
    ///@}

private:

    void buildFacetElementArray();

    void setCoarseElementAttributes(
        const mfem::Array<int> & fine_attr,
        mfem::Array<int> & coarse_attr);
};

/// Draw each agglomerated element in a different color
void ShowTopologyAgglomeratedElements(
    AgglomeratedTopology * topo,
    mfem::ParMesh * mesh,
    std::ofstream * file=nullptr);

/// @todo not implemented
void ShowTopologyAgglomeratedFacets(
    AgglomeratedTopology * topo,
    mfem::ParMesh * mesh);

void ShowTopologyBdrFacets(
    AgglomeratedTopology * topo,
    mfem::ParMesh * mesh);

/// I have no idea what this actually shows us
void ShowAgglomeratedTopology3D(
    AgglomeratedTopology * topo,
    mfem::ParMesh * mesh);

}//namespace parelag
#endif
