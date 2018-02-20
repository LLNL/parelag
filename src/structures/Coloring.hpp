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

#include <mfem.hpp>

namespace parelag
{
template<typename IJ_STRUCT>
int GetElementColoring(mfem::Array<int> &colors, const int el0, const IJ_STRUCT & el_el )
{
    int num_el = el_el.Size(), stack_p, stack_top_p, max_num_col;
    mfem::Array<int> el_stack(num_el);

    const int *i_el_el = el_el.GetI();
    const int *j_el_el = el_el.GetJ();

    colors.SetSize(num_el);
    colors = -2;
    max_num_col = 1;
    stack_p = stack_top_p = 0;
    for (int el = el0; stack_top_p < num_el; el=(el+1)%num_el)
    {
        if (colors[el] != -2)
            continue;

        colors[el] = -1;
        el_stack[stack_top_p++] = el;

        for ( ; stack_p < stack_top_p; stack_p++)
        {
            int i = el_stack[stack_p];
            int num_nb = i_el_el[i+1] - i_el_el[i]-1;   // assuming non-zeros on diagonal
            if (max_num_col < num_nb + 1)
                max_num_col = num_nb + 1;
            for (int j = i_el_el[i]; j < i_el_el[i+1]; j++)
            {
                int k = j_el_el[j];
                if (j==i)
                    continue; // skip self-interaction
                if (colors[k] == -2)
                {
                    colors[k] = -1;
                    el_stack[stack_top_p++] = k;
                }
            }
        }
    }

    mfem::Array<int> col_marker(max_num_col);

    int out = 0;

    for (stack_p = 0; stack_p < stack_top_p; stack_p++)
    {
        int i = el_stack[stack_p], col;
        col_marker = 0;
        for (int j = i_el_el[i]; j < i_el_el[i+1]; j++)
        {
            if ( j_el_el[j] == i)
                continue;          // skip self-interaction
            col = colors[j_el_el[j]];
            if (col != -1)
                col_marker[col] = 1;
        }

        for (col = 0; col < max_num_col; col++)
            if (col_marker[col] == 0)
                break;

        colors[i] = col;

        if(out < col)
            out = col;

    }

    return out+1;
}
}//namespace parelag
