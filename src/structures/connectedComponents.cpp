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

int connectedComponents(Array<int> & partitioning, SerialCSRMatrix & conn )
{
	elag_assert(partitioning.Size() == conn.Size() );
	int num_nodes    = conn.Size();
	int num_part(partitioning.Max()+1);

	Array<int> component(num_nodes);
	component = -1;
	Array<int> offset_comp(num_part+1);
	offset_comp = 0;
	Array<int> num_comp(offset_comp.GetData()+1, num_part);
	int i, j, k;
	const int *i_table, *j_table;

	i_table = const_cast<int *>(conn.GetI());
	j_table = const_cast<int *>(conn.GetJ());


   Array<int> vertex_stack(num_nodes);
   int stack_p, stack_top_p, node;

   stack_p = 0;
   stack_top_p = 0;  // points to the first unused element in the stack
	   for (node = 0; node < num_nodes; node++)
	   {

		  if(partitioning[node] < 0)
			  continue;

	      if (component[node] >= 0)
	         continue;

	      component[node] = num_comp[partitioning[node]]++;

	      vertex_stack[stack_top_p++] = node;

	      for ( ; stack_p < stack_top_p; stack_p++)
	      {
	         i = vertex_stack[stack_p];
	         if(partitioning[i] < 0)
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
	               else if (component[k] != component[i])
	               {
	                  mfem_error("FindPartitioningComponents");
	               }
	            }
	         }
	      }
	   }

	offset_comp.PartialSum();

	for(int i(0); i < num_nodes; ++i)
		partitioning[i] = offset_comp[partitioning[i]]+component[i];

	elag_assert(partitioning.Max()+1 == offset_comp.Last());

	return offset_comp.Last();
}

int connectedComponents(Array<int> & partitioning, SerialCSRMatrix & conn, Array<int> & materialSubdomains )
{
	std::cout << "WARNING: connectedComponents not implemented yet.\n";
	return partitioning.Max()+1;
}

