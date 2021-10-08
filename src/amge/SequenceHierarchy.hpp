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

#ifndef SEQUENCE_HPP_
#define SEQUENCE_HPP_

#include <amge/DeRhamSequence.hpp>

namespace parelag {

using std::shared_ptr;
using std::unique_ptr;

//enum PartitionType { MFEMRefined, METIS };

//struct MFEM_refine_info
//{
//    std::vector<int> num_elements;
//};

class SequenceHierarchy
{
    std::vector<shared_ptr<AgglomeratedTopology>> topo_;
    std::vector<shared_ptr<DeRhamSequence>> seq_;

    MPI_Comm comm_;
    bool verbose_;

    const AgglomeratedTopology::Entity elem_t_ = AgglomeratedTopology::ELEMENT;

    void GeometricPartitionings(const std::vector<int> &num_elems, int dim);

    int MinNonzeroNumLocalElements(int level, int zero_replace);
public:

    /// @param num_local_elems_threshold if the number of local elements is less
    /// than this threshold on a particular level, the level will first be
    /// redistributed and then coarsened.
    /// @param num_global_elems_threshold if the number of glocal elements is
    /// less than this threshold on a particular level, the coarsening process
    /// will be terminiated even if num_levels has not been reached.
    SequenceHierarchy(const std::shared_ptr<mfem::ParMesh>& mesh,
                      int num_levels,
                      const std::vector<int>& num_elements,
                      int elem_coarsening_factor,
                      int proc_coarsening_factor=2,
                      int num_local_elems_threshold=5,
                      int num_global_elems_threshold=5,
                      bool verbose=false);

    SequenceHierarchy(const std::shared_ptr<mfem::ParMesh>& mesh,
                      int num_levels,
                      int elem_coarsening_factor,
                      int proc_coarsening_factor=2,
                      int num_local_elems_threshold=5,
                      int num_global_elems_threshold=5,
                      bool verbose=false)
        : SequenceHierarchy(mesh, num_levels, std::vector<int>(0),
                            elem_coarsening_factor, proc_coarsening_factor,
                            num_local_elems_threshold,
                            num_global_elems_threshold, verbose)
    { }

    const std::vector<shared_ptr<DeRhamSequence>>& GetDeRhamSequences() const { return seq_; }

};



}

#endif /* SEQUENCE_HPP_ */
