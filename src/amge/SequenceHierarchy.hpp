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

class SequenceHierarchy
{
    vector<shared_ptr<AgglomeratedTopology>> topo_;
    vector<shared_ptr<DeRhamSequence>> seq_;

    MPI_Comm comm_;
    std::shared_ptr<mfem::ParMesh> mesh_;
    ParameterList params_;
    bool verbose_;

    const AgglomeratedTopology::Entity elem_t_ = AgglomeratedTopology::ELEMENT;

    void GeometricPartitionings(const vector<int>& num_elems, int dim);

    int MinNonzeroNumLocalElements(int level, int zero_replace);
public:

    /// @param num_local_elems_threshold if the number of local elements is less
    /// than this threshold on a particular level, the level will first be
    /// redistributed and then coarsened.
    /// @param num_global_elems_threshold if the number of glocal elements is
    /// less than this threshold on a particular level, the coarsening process
    /// will be terminiated even if num_levels has not been reached.
    SequenceHierarchy(shared_ptr<ParMesh> mesh, ParameterList params, bool verbose=false);

    void Coarsen(const vector<int>& num_elements);

    void Coarsen() { Coarsen(vector<int>(0)); }

    const vector<shared_ptr<DeRhamSequence>>& GetDeRhamSequences() const { return seq_; }

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

        DeRhamSequenceFE * seq = seq_[0]->FemSequence();
        seq->ReplaceMassIntegrator(elem_t_, form, move(integ), recompute_mass);
    }
};

}

#endif /* SEQUENCE_HIERARCHY_HPP_ */
