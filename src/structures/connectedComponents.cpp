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

#include "connectedComponents.hpp"

#include "utilities/elagError.hpp"

namespace parelag
{
using namespace mfem;

/// TODO should this warn (optionally) when it finds disconnected components?
int connectedComponents(Array<int> & partitioning, const SerialCSRMatrix & conn)
{
    if (partitioning.Size() == 0) { return 0; }

    elag_assert(partitioning.Size() == conn.Height() );
    elag_assert(partitioning.Size() == conn.Width() );
    int num_nodes = conn.Size();
    int num_part(partitioning.Max()+1);

    Array<int> component(num_nodes);
    component = -1;
    Array<int> offset_comp(num_part+1);
    offset_comp = 0;
    Array<int> num_comp(offset_comp.GetData()+1, num_part);
    int i, j, k;
    const int *i_table, *j_table;

    i_table = conn.GetI();
    j_table = conn.GetJ();

    Array<int> vertex_stack(num_nodes);
    int stack_p, stack_top_p, node;

    stack_p = 0;
    stack_top_p = 0;  // points to the first unused element in the stack
    for (node = 0; node < num_nodes; node++)
    {
        if (partitioning[node] < 0)
            continue;

        if (component[node] >= 0)
            continue;

        component[node] = num_comp[partitioning[node]]++;

        vertex_stack[stack_top_p++] = node;

        for ( ; stack_p < stack_top_p; stack_p++)
        {
            i = vertex_stack[stack_p];
            if (partitioning[i] < 0)
                continue;

            for (j = i_table[i]; j < i_table[i+1]; j++)
            {
                k = j_table[j];
                if (partitioning[k] == partitioning[i] )
                {
                    if (component[k] < 0)
                    {
                        component[k] = component[i];
                        vertex_stack[stack_top_p++] = k;
                    }
                    elag_assert(component[k] == component[i]);
                }
            }
        }
    }
    offset_comp.PartialSum();

    for(int i(0); i < num_nodes; ++i)
        partitioning[i] = offset_comp[partitioning[i]] + component[i];

    elag_assert(partitioning.Max()+1 == offset_comp.Last());
    return offset_comp.Last();
}

int connectedComponents(Array<int> & partitioning, SerialCSRMatrix const&,
                        Array<int> const&)
{
    std::cout << "WARNING: this form of connectedComponents not implemented yet."
              << std::endl;
    return partitioning.Max() + 1;
}

}//namespace parelag
