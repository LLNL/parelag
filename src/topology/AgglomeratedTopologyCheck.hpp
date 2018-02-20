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
#ifndef AGGLOMERATEDTOPOLOGYCHECK_HPP_
#define AGGLOMERATEDTOPOLOGYCHECK_HPP_

#include <iostream>

#include "topology/Topology.hpp"

namespace parelag
{
class AgglomeratedTopologyCheck {
public:
    /** Marks bad agglomerated entities detected by Betti numbers
        Returns true if there are any bad agglomerates */
    static bool MarkBadAgglomeratedEntities(int codim,
                                            AgglomeratedTopology & topo,
                                            mfem::Array<int> & isbad);

    /** Implements same logic as MarkBadAgglomeratedEntities, but prints 
        out results, telling you about disconnected components, holes,
        tunnels, etc. */
    static void ShowBadAgglomeratedEntities(int codim,
                                            AgglomeratedTopology & topo,
                                            std::ostream & os);

private:
    /// Don't allow creating an instance of this object
    AgglomeratedTopologyCheck() {}

    /** Marks agglomerated faces (or edges) that don't have two
        adjacent elements (or faces). These are not detected by Betti
        numbers */
    static void additionalTopologyCheck(
        int codim, AgglomeratedTopology & topo, mfem::Array<int> & isbad,
        bool verbose=false, std::ostream & os=std::cout);

    /**
       Returns in bettiNumbers(iAE, k) the the kth Betti number of
       agglomerated entity number iAE.

       Calculated as

         dim(ker(d^k)) - dim(ran(d^{k-1}))

       where d^k are local boundary operators corresponding to local
       versions of topo->B[] with different numbering.

       In three dimensions:

       b_0 = number of connected components [should be 1]
       b_1 = number of tunnels (ie, torus) [should be 0]
       b_2 = number of voids [should be 0]
    */
    static void computeBettiNumbersAgglomeratedEntities(
        int codim, AgglomeratedTopology & topo,
        mfem::DenseMatrix & bettiNumbers);

    static void showBadAgglomeratedElements(
        mfem::DenseMatrix & bettiNumbers,
        AgglomeratedTopology & topo, std::ostream & os);

    static void showBadAgglomeratedFacets(
        mfem::DenseMatrix & bettiNumbers,
        AgglomeratedTopology & topo, std::ostream & os);

    static void showBadAgglomeratedRidges(
        mfem::DenseMatrix & bettiNumbers,
        AgglomeratedTopology & topo, std::ostream & os);
};
}//namespace parelag
#endif /* AGGLOMERATEDTOPOLOGYCHECK_HPP_ */
