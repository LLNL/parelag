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


#ifndef PARELAG_PARAMETERLIST_HPP_
#define PARELAG_PARAMETERLIST_HPP_

#include <unordered_map>

#include "ParELAG_VariableContainer.hpp"

namespace parelag
{

/** \class ParameterList
 *  \brief An associative container that maps strings to type-erased
 *         members.
 */
class ParameterList
{
public:
    // type aliasing
    using key_type = std::string;
    using mapped_type = VariableContainer;
    using map_type = std::unordered_map<key_type,mapped_type>;

    using iterator = map_type::iterator;
    using const_iterator = map_type::const_iterator;
    using size_type = map_type::size_type;

    // Have to use unique_ptr here because GCC whines about recursively using
    // the ParameterList type.
    using sublist_map_type =
        std::unordered_map<key_type,std::unique_ptr<ParameterList>>;
    using sublist_iterator = sublist_map_type::iterator;
    using sublist_const_iterator = sublist_map_type::const_iterator;

public:

    /** \name Constructors and destructors */
    ///@{

    /** \brief Default constructor */
    ParameterList()
        : Map_{}, SublistMap_{}, Name_{}
    {}

    /** \brief Named constructor */
    ParameterList(std::string name)
        : Map_{}, SublistMap_{}, Name_{std::move(name)}
    {}

    /** \brief Copy constructor */
    ParameterList(const ParameterList& rhs)
        : Map_{rhs.Map_},
          SublistMap_{},
          Name_{rhs.Name_}
    {
        // Copy through the pointers
        for (auto const& sl : rhs.SublistMap_)
            SublistMap_[sl.first] = make_unique<ParameterList>(*sl.second);
    }

    /** \brief Copy assignment operator */
    ParameterList& operator=(const ParameterList& rhs)
    {
        Map_ = rhs.Map_;
        Name_ = rhs.Name_;
        for (const auto& sl : rhs.SublistMap_)
            SublistMap_[sl.first] = make_unique<ParameterList>(*sl.second);
        return *this;
    }

    /** \brief Default move constructors */
    ParameterList(ParameterList&&) = default;
    ParameterList& operator=(ParameterList&&) = default;

    /** \brief Destructor */
    virtual ~ParameterList() = default;

    ///@}
    /** \name Get attributes */
    ///@{

    /** \brief Get the name of the parameter list */
    const std::string& GetName() const noexcept
    {
        return Name_;
    }

    ///@}
    /** \name Set attributes */
    ///@{

    /** \brief Set the name of the parameter list */
    void SetName(std::string name)
    {
        Name_ = std::move(name);
    }

    ///@}
    /** \name Query the list */
    ///@{

    /** \brief Returns true if the key exists in the map, false otherwise */
    bool IsParameter(const key_type& name) const noexcept
    {
        return !(Map_.find(name) == Map_.end());
    }

    /** \brief Test if the key exists in the map with a nonempty container.
     *
     *  \returns true if the key exists in the map and has data.
     *  \returns false if the key does not exist or its container is empty.
     */
    bool IsValid(const key_type& name) const noexcept;

    /** \brief Returns true if the key is the name of a sublist */
    bool IsSublist(const key_type& name) const noexcept;

    ///@}
    /** \name Set parameters */
    ///@{

    /** \brief Add a parameter to the list or reset its value.
     *
     *  Set a parameter called "name" with value "val" of type "T". This
     *  will overwrite the parameter if it is already present in the
     *  map and create it if it is not.
     */
    template <typename T>
    void Set(const key_type& name, T&& val)
    {
        Map_[name] = std::forward<T>(val);
    }

    /** \brief Specialization of Set for char*.
     *
     *  The value is stored as an std::string.
     */
    void Set(const key_type& name, const char * val)
    {
        Map_[name] = std::string(val);
    }

    /** \brief Add a "sublist" to the parameter list. */
    void Set(const key_type& name, const ParameterList& val)
    {
        this->Sublist(name) = val;
    }

    ///@}
    /** \name Get parameters */
    ///@{

    /** \brief Get the data keyed by "name" and return it.
     *
     *  If "name" does not exist in the map, create it with the
     *  default_value. If "name" exists in the map but is not of type T,
     *  then throw bad_var_cast.
     */
    template <typename T>
    T& Get(const key_type& name, const T& default_value)
    {
        // FIXME (trb 02/10/16): This could be simplified by using
        // "insert" or "emplace" (preferably the latter) since no
        // insertion happens if the element exists and an iterator to
        // the entry, inserted or not, is always returned.
        auto iter = Map_.find(name);

        // Does not exist; create it
        if (iter == Map_.end())
        {
            Map_[name] = default_value;
            return Map_[name].GetData<T>();
        }

        return iter->second.GetData<T>();
    }

    /** \brief Get the data keyed by "name" when data is of type
     *         std::string.
     *
     *  Recall that parameters set with type "char*" are actually
     *  stored as std:: string objects.
     */
    std::string& Get(const key_type& name, const char default_value[])
    {
        return Get(name, std::string(default_value));
    }

    /** \brief Get the data keyed by "name".
     *
     *  \throws std::out_of_range if "name" does not exist in the map
     *  \trhows parelag::bad_var_cast if the data keyed by "name" is not of
     *              type T.
     */
    template <typename T>
    T& Get(const key_type& name)
    {
        return Map_.at(name).GetData<T>();
    }

    /** \brief Get the data keyed by "name". (Const version) */
    template <typename T>
    const T& Get(const key_type& name) const
    {
        return Map_.at(name).GetData<T>();
    }

    ///@}
    /** \name ParameterList interaction */
    ///@{

    /** \brief Merge the parameters in "other" into this list. */
    void Merge(const ParameterList& other);

    ///@}
    /** \name Sublist management */
    ///@{

    /** \brief Get a sublist with the given name.
     *
     *  If such a sublist exists, it is returned; if it does not
     *  exist, it is created and returned. If "name" exists as a
     *  parameter that is NOT a ParameterList, an exception is thrown.
    */
    ParameterList& Sublist(const std::string& name, bool must_exist = false);

    /** \brief Get the specified sublist; the list must exist. */
    const ParameterList& Sublist(const std::string& name) const;

    ///@}
    /** \name std::map-like things, for <algorithm> compatibility */
    ///@{

    /** \brief Get the number of parameters in the list. */
    size_type size() const noexcept
    {
        return Map_.size();
    }

    /** \brief Get iterator to the beginning of the map. */
    iterator begin() noexcept
    {
        return Map_.begin();
    }

    /** \brief Get constant iterator to the beginning of the map. */
    const_iterator begin() const noexcept
    {
        return Map_.begin();
    }

    /** \brief Get iterator to the end of the map */
    iterator end() noexcept
    {
        return Map_.end();
    }

    /** \brief Get constant iterator to the end of the map. */
    const_iterator end() const noexcept
    {
        return Map_.end();
    }

    /** \brief Get iterator to the beginning of the map. */
    sublist_iterator begin_sublist() noexcept
    {
        return SublistMap_.begin();
    }

    /** \brief Get constant iterator to the beginning of the map. */
    sublist_const_iterator begin_sublist() const noexcept
    {
        return SublistMap_.begin();
    }

    /** \brief Get iterator to the end of the map */
    sublist_iterator end_sublist() noexcept
    {
        return SublistMap_.end();
    }

    /** \brief Get constant iterator to the end of the map. */
    sublist_const_iterator end_sublist() const noexcept
    {
        return SublistMap_.end();
    }
    ///@}

    /** Print the parameter list (and sublists) to the stream */
    void Print(std::ostream& os, unsigned indent = 0) const noexcept;

private:

    /** \brief The map from strings to values */
    map_type Map_;

    /** \brief The map from strings to sublists */
    sublist_map_type SublistMap_;

    /** \brief The name of this parameter list */
    std::string Name_ = "";

};// class ParameterList

/** \brief Overloaded stream operator for ParameterLists. */
inline std::ostream& operator<<(std::ostream& os, const ParameterList& list)
{
    list.Print(os);
    return os;
}

}// namespace parelag
#endif /* PARELAG_PARAMETERLIST_HPP_ */
