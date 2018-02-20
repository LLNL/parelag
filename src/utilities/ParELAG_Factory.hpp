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


#ifndef PARELAG_FACTORY_HPP_
#define PARELAG_FACTORY_HPP_

#include <functional>
#include <list>
#include <memory>
#include <sstream>
#include <unordered_map>

#include "utilities/MemoryUtils.hpp"

namespace parelag
{

/** \class DefaultFactoryErrorPolicy
 *  \brief Defines how a factory will handle building objects of
 *         an unknown type.
 *
 *  The policy must define "HandleUnknownId(IdType const&)".
 *
 *  The default behavior is to throw an exception.
 *
 *  \tparam IdType The type used for IDs in the Factory.
 *  \tparam AbstractType The type built by the Factory.
 */
template <typename IdType, class AbstractType>
struct DefaultFactoryErrorPolicy
{
    /** \class unknown_id_error
     *  \brief Exception that is thrown when unknown ID encountered.
     */
    struct unknown_id_error : public std::runtime_error
    {
        /** \brief Constructor taking it's "what()" string. */
        unknown_id_error(const std::string& what_arg)
            : std::runtime_error(what_arg) {}
    };

    /** \brief Handles an unknown ID by throwing exception.
     *
     *  \throws unknown_id_error
     *
     *  \return Return type specified to satisfy interface; no object
     *          actually returned.
     */
    std::unique_ptr<AbstractType> HandleUnknownId(IdType const& id) const
    {
        std::ostringstream oss("Unknown type identifier ", std::ios_base::ate);
        oss << "\"" << id << "\"";
        throw unknown_id_error(oss.str());
    }
};// struct DefaultFactoryErrorPolicy


/** \class Factory
 *  \brief Generic factory template.
 *
 *  Instantiations of this template are registries that map some type
 *  of ID to some builder type that constructs things of a given
 *  (base) class. The user needs to specify these pieces of
 *  information (ID type, Builder type, and Abstract type) as well as
 *  a policy class for handling the situation in which a user requests
 *  an unknown ID.
 *
 *  \tparam AbstractType  The base class of the types being constructed.
 *  \tparam IdType        The index type used to differentiate concrete types.
 *  \tparam BuilderType   The functor type that builds concrete types.
 *  \tparam ErrorPolicy   The way errors are handled.
 */
template <class AbstractType,
          typename IdType,
          typename BuilderType = std::function<std::unique_ptr<AbstractType>()>,
          template <typename, class> class ErrorPolicy
          = DefaultFactoryErrorPolicy>
class Factory : public ErrorPolicy<IdType,AbstractType>
{
public:

    /** \brief The (base) type of the objects that this factory creates. */
    using abstract_type = AbstractType;

    /** \brief The type of IDs used in the map. */
    using id_type = IdType;

    /** \brief The builder functor type. */
    using builder_type = BuilderType;

    /** \brief The type of the underlying map. */
    using map_type = std::unordered_map<id_type,builder_type>;

public:

    /** \brief Register a new builder with the factory.
     *
     *  The new builder is keyed by \c id. If a builder in the map
     *  already exists with key \c id, then no change occurs.
     *
     *  \param id The unique identifier for the builder.
     *  \param builder The builder corresponding to \c id.
     *
     *  \return true if registration successful.
     */
    bool RegisterBuilder(id_type id, builder_type builder)
    {
        return _map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::move(id)),
            std::forward_as_tuple(std::move(builder))).second;
    }

    /** \brief Deregister an existing builder.
     *
     *  If the given id \c id does not correspond to a registered
     *  builder, then no change occurs.
     *
     *  \param id The unique identifier for the builder to be removed.
     *
     *  \return \c true if deregistration was successful.
     */
    bool Deregister(id_type const& id)
    {
        return (_map.erase(id) == 1);
    }

    /** \brief Construct a new object.
     *
     *  The additional arguments are forwarded to the builder.
     *
     *  Unknown values for \c id are handled by \c ErrorPolicy.
     *
     *  \param id The unique identifier for the builder to be called.
     *
     *  \return A newly-created object of the given AbstractType.
     */
    template <typename... Ts>
    std::unique_ptr<AbstractType> CreateObject(
        IdType const& id, Ts&&... Args) const
    {
        auto it = _map.find(id);
        if (it != _map.end())
            return (it->second)(std::forward<Ts>(Args)...);

        return this->HandleUnknownId(id);
    }

    /** \brief Get the names of all builders known to the factory.
     *
     *  This function primarily exists for debugging the factory.
     *
     *  \return A list of identifiers known to the factory.
     */
    std::list<id_type> GetRegisteredBuilderNames() const
    {
        std::list<id_type> names;
        for (const auto& x : _map)
            names.push_back(x.first);

        return names;
    }

private:

    /** \brief The map from identifiers to builders. */
    map_type _map;

};// class Factory
}// namespace parelag
#endif /* PARELAG_OBJECT_FACTORY_HPP_ */
