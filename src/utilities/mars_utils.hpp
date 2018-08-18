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

#ifndef PARELAG_MARSUTILS_HPP_
#define PARELAG_MARSUTILS_HPP_

#include <mfem.hpp>
#include <mars.hpp>
#include <map>

namespace parelag {

template<mars::Integer Dim>
static void convert(const mfem::Mesh &in, mars::Mesh<Dim, Dim> &out)
{
    //TODO: Handle element attribute.

    using Point = mars::Vector<mars::Real, Dim>;
    using Elem = mars::Simplex<Dim, Dim>;

    mfem::Array<int> vertices;

    out.clear();
    out.reserve(in.GetNE(), in.GetNV());

    for (int i=0; i < in.GetNE(); ++i)
    {

        Elem elem;
        auto &e = *in.GetElement(i);
        PARELAG_ASSERT(e.GetNVertices() == Dim+1);
        e.GetVertices(vertices);
        for (int k=0; k < e.GetNVertices(); ++k)
            elem.nodes[k] = vertices[k];
        out.add_elem(elem);
    }

    for (int i=0; i < in.GetNV(); ++i)
    {
        Point p;
        auto in_p = in.GetVertex(i);
        for (int d=0; d < Dim; ++d)
            p(d) = in_p[d];
        out.add_point(p);
    }

    // Can copy it from mfem
    out.update_dual_graph();

    std::map<mars::Side<Dim>, mars::Integer> b_to_index;
    mars::Side<Dim> temp;

    for (int i=0; i < in.GetNBE(); ++i)
    {
        Point p;
        auto &be = *in.GetBdrElement(i);
        be.GetVertices(vertices);
        for (int k = 0; k < be.GetNVertices(); ++k)
            temp[k] = vertices[k];
        temp.fix_ordering();
        b_to_index[temp] = i;
    }

    mars::Simplex<Dim, Dim-1> side;
    for (mars::Integer i=0; i < out.n_elements(); ++i)
    {
        if (!out.is_active(i) || !out.is_boundary(i))
            continue;
        auto &e = out.elem(i);
        for (mars::Integer k = 0; k < n_sides(e); ++k)
        {
            e.side(k, side);
            temp = mars::Side<Dim>(side.nodes);
            auto it = b_to_index.find(temp);
            if (it == b_to_index.end())
                continue;
            auto bdr_index = it->second;
            auto attr = in.GetBdrAttribute(bdr_index);
            e.side_tags[k] = attr;
        }
    }
}

template<mars::Integer Dim>
static std::shared_ptr<mfem::Mesh> convert(const mars::Mesh<Dim, Dim> &in, int geom, int bdr_geom)
{
    using Point = mars::Vector<mars::Real, Dim>;
    const auto n_active_elements = in.n_active_elements();
    PARELAG_ASSERT(n_active_elements > 0);

    std::shared_ptr<mfem::Mesh> temp = std::make_shared<mfem::Mesh>(Dim, in.n_nodes(), n_active_elements, in.n_boundary_sides());

    for (mars::Integer i=0; i < in.n_nodes(); ++i)
        temp->AddVertex(&in.point(i)[0]);

    mars::Simplex<Dim, Dim-1> side;
    mars::Integer el_index = 0;
    for (mars::Integer i=0; i < in.n_elements(); ++i)
    {
        if (!in.is_active(i))
            continue;

        auto *e = temp->NewElement(geom);
        std::vector<int> vertices;
        vertices.insert(vertices.begin(), in.elem(i).nodes.begin(), in.elem(i).nodes.end());
        e->SetVertices(&vertices[0]);
        e->SetAttribute(1);
        temp->AddElement(e);

        if (in.is_boundary(i))
        {
            const auto &me = in.elem(i);
            for (mars::Integer k=0; k < n_sides(me); ++k)
            {
                if (!in.is_boundary(i, k) || me.side_tags[k] < 0)
                    continue;

                me.side(k, side);
                auto *be = temp->NewElement(bdr_geom);
                vertices.insert(vertices.begin(), side.nodes.begin(), side.nodes.end());
                be->SetVertices(&vertices[0]);
                be->SetAttribute(me.side_tags[k]);
                temp->AddBdrElement(be);
            }
        }
    }

//    if (Dim == 4)
//        temp->PrepareFinalize4D();
    temp->FinalizeTopology();
    temp->Finalize();

    return temp;
}

template<mars::Integer Dim>
static int *refine(mars::Mesh<Dim, Dim> &in)
{
    mars::Bisection<mars::Mesh<Dim, Dim>> b(in);
    b.uniform_refine(1);
    const auto n_active_elements = in.n_active_elements();
    PARELAG_ASSERT(n_active_elements > 0);
    int *partitioning = new int[n_active_elements];

    mars::Integer el_index = 0;
    for (mars::Integer i=0; i < in.n_elements(); ++i)
    {
        PARELAG_ASSERT(in.elem(i).id == i);
        if (!in.is_active(i))
            continue;

        int part = in.elem(i).parent_id;
        //part = (part == mars::INVALID_INDEX ? in.elem(i).id : part);
        PARELAG_ASSERT(part != mars::INVALID_INDEX);
        while (in.elem(part).parent_id != mars::INVALID_INDEX)
            part = in.elem(part).parent_id;
        PARELAG_ASSERT(el_index < n_active_elements);
        partitioning[el_index++] = part;
    }
    in.clean_up();
    PARELAG_ASSERT(el_index == in.n_active_elements() && in.n_active_elements() == in.n_elements());

    return partitioning;
}

}

#endif /* PARELAG_MARSUTILS_HPP_ */
