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

#ifndef SEQUENCE_HIERARCHY_HPP_
#define SEQUENCE_HIERARCHY_HPP_

#include <amge/DeRhamSequenceFE.hpp>
#include <utilities/ParELAG_SimpleXMLParameterListReader.hpp>

namespace parelag {

using namespace std;
using namespace mfem;

/** \class SequenceHierarchy
 *  \brief A wrapper for the construction of a hierarchy of DeRhamSequences.
 *
 *  Based on the given parameters, this class helps you to construct a hierarchy
 *  of DeRhamSequences by first performing geometric coarsening (based on mfem
 *  refinements), and then algebraic coarsening (based on METIS).
 *
 *  If a particular level has too few elements in some processors, the level
 *  will be redistributed to a subset of processors before further coarsening.
 */
class SequenceHierarchy
{
    // TODO (aschaf 09/21/22) Describe topo_ and seq_ layout
    /**
     * The topologies and sequences are stored based on which communicator C_k they belong to, 
     * with k the communicator or redistribution index. Hence, if multiple communicators are used,
     * as is the case when we coarsen beyond the number of processors and want to have multiple 
     * coarse copies. 
     * 
     * 
     */
    vector<vector<shared_ptr<AgglomeratedTopology>>> topo_;
    vector<vector<shared_ptr<DeRhamSequence>>> seq_;
    // 
    vector<unique_ptr<MultiRedistributor>> redistributors_;
    // TODO (aschaf 09/08/22): decide on a canonical way to store this information
    // tag indicating in which subgroup the level belongs in relation to the previous level
    vector<int> mycopy_;
    vector<bool> level_is_redistributed_;
    bool is_redistributed_;
    vector<int> num_copies_;
    /**
     * @brief Redistribution index \f$ k_\ell \f$
     * 
     * 
     */
    vector<int> redistribution_index;
    /// communicators C_k, C_0 is usually MPI_COMM_WORLD
    vector<MPI_Comm> comms_;
    vector<int> num_global_copies_;

    MPI_Comm comm_;
    std::shared_ptr<mfem::ParMesh> mesh_;
    ParameterList params_;
    bool verbose_;

    bool mass_is_assembled_;

    const AgglomeratedTopology::Entity elem_t_ = AgglomeratedTopology::ELEMENT;

    void GeometricCoarsenings(const Array<int>& num_elems, int dim);

    int MinNonzeroNumLocalElements(int level, int zero_replace);

    int ser_ref_levels; // number of serial refinements (to undo)
    Array<int> partitioning_permutation; // inverse map on how the elements are distributed along the processors (not owned)
public:

    /** \brief Constructor.
     *
     *  Set up the fine level DeRhamSequence of the hierarchy. The coarse levels
     *  will be constructed by coarsening the fine level recursively.
     *
     *  \param mesh The mesh where the fine level DeRhamSequence is defined on.
     *
     *  \param params A list of parameters that dictates the coarsening process.
     *         In particular, the following parameters can be supplied:
     *         * Hierarchy levels (default is 2):
     *              Desired number of levels of DeRhamSequences to be built.
     *
     *         * Finite element order (default is 0):
     *              Order of finite element spaces in the fine-level sequence.
     *
     *         * Upscaling order (default is 0):
     *              Polynomial order for global interpolation targets in the
     *              coarse-level sequences.
     *
     *         * Hierarchy coarsening factor (default is 8):
     *              Average number of elements in agglomerates.
     *
     *         * Processor coarsening factor (default is 2):
     *              Intended number of processors to be grouped together
     *              whenever redistribution is triggered.
     *
     *         * Local elements threshold (default is 80):
     *              If the number of local elements is less than this threshold
     *              on a particular level, the level will first be redistributed
     *              and then coarsened.
     *
     *         * Global elements threshold (default is 10):
     *              If the number of global elements is less than this threshold
     *              on a particular level, the coarsening process will be
     *              terminiated earlier.
     *
     *         * SVD tolerance (default is 1e-6):
     *              Cutoff value for (relative singular values) for selecting
     *              singular vectors in the coarsening steps. The smaller the
     *              tolerance, the more singular vectors will be selected.
     *
     *  \param verbose Whether to print the progress of coarsening.
     */
    SequenceHierarchy(shared_ptr<ParMesh> mesh, ParameterList params, bool verbose=false);

    SequenceHierarchy(const SequenceHierarchy&) = delete;
    SequenceHierarchy& operator=(const SequenceHierarchy&) = delete;

    /// Construct the hierarchy via recursive coarsening.
    /// \param num_elements contains the number of elements in the first few
    /// finer levels (including the finest). These numbers are obtained from
    /// mfem (parallel) refinements. num_elements[0] is number of elements on
    /// the finest level. Geometric coarsening will be performed until all the
    /// numbers in num_elements have been used. Further coarsenings are algebraic.
    void Build(const Array<int>& num_elements, const SequenceHierarchy &other_sequence_hierarchy);
    void Build(const Array<int>& num_elements)
    {
        // vector <shared_ptr<DeRhamSequence>> dummy(0);
        Build(num_elements, *this);
    }
    void Build(vector<int> num_elements)
    {
        Array<int> num_elems(num_elements.data(), num_elements.size());
        Build(num_elems);
    }

    /// Construct the hierarchy without geometric coarsening.
    void Build() { Build(vector<int>(0)); }

    const vector<shared_ptr<DeRhamSequence>>& GetDeRhamSequences(int k = 0) const;

    /// A portal to DeRhamSequenceFE::ReplaceMassIntegrator.
    /// This is for setting coefficients for mass matrices in the fine sequence.
    /// Use recompute_mass to control when to general the element mass matrices.
    /// Typically you want to generate them after all the coefficients are set.
    template<typename T>
    void SetCoefficient(int form, T& coef, bool recompute_mass)
    {
        unique_ptr<BilinearFormIntegrator> integ;
        if (form == 0 || form == mesh_->Dimension())
        {
            integ.reset(new MassIntegrator(coef));
        }
        else
        {
            integ.reset(new VectorFEMassIntegrator(coef));
        }

        ReplaceMassIntegrator(form, move(integ), recompute_mass);
    }

    void ReplaceMassIntegrator(int form,
                               unique_ptr<BilinearFormIntegrator> integ,
                               bool recompute_mass);


    int GetMyCopy(int level) const noexcept
    {
        return mycopy_[level];
    }

    // vector<bool> GetRedistributionBoolVector() const noexcept
    // {
    //     return redistribution_;
    // }

    const vector<MPI_Comm> & GetComms() const noexcept
    {
        return comms_;
    }

    MPI_Comm GetComm(int k) const
    {
        return comms_[k];
    }

    /// Returns the number of groups within MPI_COMM_WORLD for redistribution index k
    int GetNumGlobalCopies(int k) const
    {
        return num_global_copies_[k];
    }

    const vector<int> & GetNumGlobalCopies() const
    {
        return num_global_copies_;
    }

    /// Returns the number of groups within the parent communicator
    int GetNumCopies(int level) const
    {
        return num_copies_[level];
    }

    inline bool IsRedistributed() const noexcept
    {
        return is_redistributed_;
    }

    inline bool IsRedistributed(int level) const
    {
        return level_is_redistributed_[level];
    }

    bool IsVectorRedistributed(int level, int jform, const mfem::Vector &x);

    unique_ptr<ParallelCSRMatrix> RedistributeParMatrix(int level, int jform, const ParallelCSRMatrix *mat);

    /// Redistribute copies of a parallel matrix \p mat , where domain and range are from different sequence hiearchies.
    /// Both hierarchies \b MUST share the same communicator layout!
    unique_ptr<ParallelCSRMatrix> RedistributeParMatrix(int level, int jform, const ParallelCSRMatrix *mat, const SequenceHierarchy &range_hierarchy);

    /// Redistribute copies of a parallel vector \p x on level \p level
    void RedistributeVector(int level, int jform, const Vector &x, Vector &redist_x);

    int GetRedistributionIndex(int level) const
    {
        return redistribution_index[level];
    }

    const std::vector<int> & GetRedistributionIndices() const
    {
        return redistribution_index;
    }

    int GetNumRedistributions() const
    {
        return redistribution_index.back();
    }

    // FIXME (aschaf 09/08/22) Find a better name for these functions
    /// Applies the TrueP matrix from a given level, projecting from level l to l+1
    /// If multiple copies are available, it redistributes a copy of the result to
    /// each group of processors. 

    void ApplyTruePi(int l, int jform, const Vector &x, Vector &y);

    void ApplyTruePTranspose(int level, int jform, const Vector &x, Vector &y);

    /**
     * @brief 
     * 
     * @param level 
     * @param k redistribution index
     * @param groupid which group in communicator C_k
     * @param jform 
     * @param true_v 
     */
    void ShowTrueData(int level, int k, int groupid, int jform, MultiVector & true_v);

    /**
     * @brief Set the number of serial refinements as well as the parallel partioning of the final serial refinement. Does not assume ownership of the array.
     * 
     * @param serial_refinements number of serial refinements before distributing the mesh
     * @param partitioning_permutation map of the initial distribution to the original ordering
     */
    void SetSerialRefinementInfo(int serial_refinements, const Array<int>& partitioning_permutation)
    {
        ser_ref_levels = serial_refinements;
        this->partitioning_permutation.MakeRef(partitioning_permutation);
    }
};

}

#endif /* SEQUENCE_HIERARCHY_HPP_ */
