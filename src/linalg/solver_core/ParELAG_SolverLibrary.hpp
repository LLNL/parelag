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


#ifndef PARELAG_SOLVERLIBRARY_HPP_
#define PARELAG_SOLVERLIBRARY_HPP_

#include <unordered_map>

#include "linalg/solver_core/ParELAG_SolverFactory.hpp"

#include "utilities/ParELAG_Factory.hpp"
#include "utilities/ParELAG_ParameterList.hpp"

namespace parelag
{

/** \class SolverLibrary
 *  \brief A runtime collection of known solvers.
 *
 *  This class is essentially a SolverFactory Factory with some extra
 *  features. From the point of view of this class, a "solver" is a
 *  type coupled with some parameters specific to that type. This
 *  class is basically a map from user-defined solver names to
 *  (type,parameters) pairs.
 *
 *  When a user requests the SolverFactory for some solver with given
 *  name "my solver name", this checks for a solver called "my solver
 *  name" in its map. If it finds one, it creates an instance of the
 *  SolverFactory corresponding to "my solver name"'s type string and
 *  initializes it with "my solver name"'s parameters.
 *
 *  There are a few known solver types builtin. These are added to the
 *  library automatically when Initialize is called. Additional solver
 *  types may be added by calling
 *  "AddNewSolverFactory(type,builder)". For example:
 *
 *      class SuperCoolSolverFactory : public SolverFactor { ... };
 *
 *      int main(int argc, char** argv)
 *      {
 *          ...
 *
 *          std::shared_ptr<SolverLibrary> lib = SolverLibrary::CreateLibrary();
 *          lib->AddNewSolverFactory("SuperCool",SolverFactoryCreator<SuperCoolSolverFactory>{});
 *
 *          ...
 *      }
 *
 *  Notice that the constructors are private. This is because we
 *  inherit from std::enable_shared_from_this. Such inheritance is
 *  required for building nested preconditioners that need access to
 *  the solver library to build their subsolvers. To enable
 *  construction, there is a static CreateLibrary() function that
 *  allows default construction or construction with a ParameterList
 *  but ensures that the library can only be managed through a
 *  std::shared_ptr.
 */
class SolverLibrary : public std::enable_shared_from_this<SolverLibrary>
{
private:

    using id_type = std::string;
    using builder_type = std::function<std::unique_ptr<SolverFactory>
                                       (ParameterList const&,
                                        std::shared_ptr<const SolverLibrary>)>;
    using factory_factory_type = Factory<SolverFactory,id_type,builder_type>;

public:

    /** \brief Creates a shared_ptr-managed SolverLibrary */
    template <typename... Ts>
    static std::shared_ptr<SolverLibrary> CreateLibrary(Ts&&... Args)
    {
        return std::shared_ptr<SolverLibrary>{
            new SolverLibrary{std::forward<Ts>(Args)...}};
    }


    /** \brief Construct from a ParameterList description
     *
     *  \param pl  The parameter list describing the whole library.
     */
    void Initialize(ParameterList const& pl)
    {
        _default_solvers_initialize();

        for (auto it = pl.begin_sublist(); it != pl.end_sublist(); ++it)
        {
            auto add_success = AddSolver(it->first, *(it->second));
            PARELAG_ASSERT_DEBUG(add_success);
        }
    }


    /** \brief Get the SolverFactory object for the given solver name.
     *
     *  The returned object is fully initialized with corresponding
     *  ParameterList.
     *
     *  Unknown solver_names are handled by the same error-handling
     *  policy as the underlying Factory.
     *
     *  \param solver_name  The human-chosen name for the desired solver.
     *  \return  The fully-initialized SolverFactory object.
     */
    std::unique_ptr<SolverFactory>
    GetSolverFactory(id_type const& solver_name) const
    {
        // Go find the solver type and parameter list
        auto it = string_to_pl_map_.find(solver_name);

        if (it != string_to_pl_map_.end())
        {
            const auto& ret_pair = it->second;
            return solver_factory_factory_.CreateObject(
                ret_pair.first,ret_pair.second,shared_from_this());
        }

        return solver_factory_factory_.HandleUnknownId(solver_name);
    }


    /** \brief Add a new solver to the library.
     *
     *  This version expects that the input ParameterList has a string
     *  member named "Type" and a sublist called "Solver
     *  Parameters". Other entries are ignored.
     *
     *  \param name  The name given to the solver being added.
     *  \param pl    The parameter list describing the solver.
     *
     *  \return \c true if the solver was successfully added.
     */
    bool AddSolver(id_type name, ParameterList const& pl)
    {
        return AddSolver(std::move(name),
                         pl.Get<std::string>("Type"),
                         pl.Sublist("Solver Parameters"));
    }


    /** \brief Add a new solver of a given type to the library.
     *
     *  This version expects that the parameter list being passed in
     *  is equivalent to a "Solver Parameters" parameter list (though
     *  it need not have that name).
     *
     *  \param name  The name given to the solver being added.
     *  \param type  The type of the solver being added.
     *  \param pl    The parameter list describing the solver.
     *
     *  \return \c true if the solver was successfully added.
     */
    bool AddSolver(id_type name, id_type type, ParameterList solver_params)
    {
        return string_to_pl_map_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::move(name)),
            std::forward_as_tuple(
                std::piecewise_construct,
                std::forward_as_tuple(std::move(type)),
                std::forward_as_tuple(std::move(solver_params)))).second;
    }


    /** \brief Get a list of the names of registered solvers.
     *
     *  \return A list of the names of registered solvers.
     */
    std::list<std::string> GetSolverNames() const
    {
        std::list<std::string> ret;
        for (const auto& s : string_to_pl_map_)
            ret.emplace_back(s.first);
        return ret;
    }


    /** \brief Add a new SolverFactoryCreator to the library's factory.
     *
     *  \param id       The name given to the SolverFactory type
     *  \param builder  A functor taking a ParameterList and
     *                  std::shared_ptr<const SolverLibrary> and
     *                  returning std::unique_ptr<SolverFactory>
     *
     *  \return \c true if the SolverFactory was added successfully.
     */
    bool AddNewSolverFactory(id_type id,builder_type builder)
    {
        return solver_factory_factory_.RegisterBuilder(
            std::move(id),std::move(builder));
    }


    /** \brief Remove a SolverFactoryCreator from the library's factory.
     *
     *  \param id  The name given to the SolverFactory type to be removed.
     *
     *  \return \c true if the SolverFactory was removed successfully.
     */
    bool RemoveSolverFactory(id_type const& id)
    {
        return solver_factory_factory_.Deregister(id);
    }

    /** \brief Get a list of the names of registered solver types.
     *
     *  \return A list of names of register SolverFactories.
     */
    std::list<std::string> GetSolverFactoryNames() const
    {
        return solver_factory_factory_.GetRegisteredBuilderNames();
    }

private:

    /** \brief Default constructor */
    SolverLibrary() = default;

    /** \brief Constructor from a parameter list representation.
     *
     *  \param pl  A parameter list describing a collection of known
     *             solvers.
     */
    SolverLibrary(ParameterList const& pl)
    {
        Initialize(pl);
    }

    /** \brief Initialize the default known SolverFactories.
     *
     *  Currently known SolverFactory types are:
     *
     *  Name String                | SolverFactory Type
     *  -------------------------- | ---------------------------------
     *    "ADS"                    | ADSSolverFactory
     *    "AMGe"                   | AMGeSolverFactory
     *    "AMS"                    | AMSSolverFactory
     *    "Block GS"               | Block2x2GaussSeidelSolverFactory
     *    "Block Jacobi"           | Block2x2JacobiSolverFactory
     *    "Block LDU"              | Block2x2LDUSolverFactory
     *    "BoomerAMG"              | BoomerAMGSolverFactory
     *    "Bramble-Pasciak"        | BramblePasciakFactory
     *    "Direct"                 | DirectSolverFactory
     *    "Hiptmair"               | HiptmairSmootherFactory
     *    "Hybridization"          | HybridizationSolverFactory
     *    "Hypre"                  | HypreSmootherFactory
     *    "Krylov"                 | KrylovSolverFactory
     *    "Stationary Iteration"   | StationarySolverFactory
    */
    void _default_solvers_initialize();

private:

    /** \brief The actual SolverFactory factory object */
    factory_factory_type solver_factory_factory_;

    /** \brief Map of names to solver types and parameter list pairs. */
    std::unordered_map<id_type,std::pair<id_type,ParameterList>>
        string_to_pl_map_;

};// class SolverLibrary
}// namespace parelag
#endif /* PARELAG_SOLVERLIBRARY_HPP_ */
